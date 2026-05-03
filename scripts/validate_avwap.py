from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.foundation.volatility import add_atr
from src.indicators.foundation.value import add_anchored_vwap
from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state
from src.indicators.structure.bos import add_bos
from src.indicators.structure.choch import add_choch
from src.validation.indicators.value import validate_avwap

OUT_DIR = Path("notebooks/foundation")
PLOT_ROWS = 300
MIN_FAMILY_ACTIVE_ROWS = 25
FAMILY_SAMPLE_COUNT = 5
RUNS = (
    ("XAU_USD", "H1"),
    ("XAU_USD", "H4"),
)


def _print_summary(value: object, *, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                print(f"{prefix}{key}:")
                _print_summary(child, indent=indent + 2)
            else:
                print(f"{prefix}{key}: {child}")
        return
    print(f"{prefix}{value}")


def _as_tick_only(raw: pd.DataFrame) -> pd.DataFrame:
    if "tickVolume" in raw.columns and "volume" not in raw.columns:
        return raw.copy()
    out = raw.copy()
    if "tickVolume" not in out.columns:
        volume_idx = int(out.columns.get_loc("volume"))
        tick_volume = normalize_candle_schema(out, require_volume=True)["volume"]
        out = out.drop(columns=["volume"], errors="ignore")
        out.insert(volume_idx, "tickVolume", tick_volume.astype(float))
        return out
    return out.drop(columns=["volume"], errors="ignore")


def _as_volume_only(raw: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_candle_schema(raw, require_volume=True)
    out = raw.copy()
    if "tickVolume" in out.columns:
        tick_idx = int(out.columns.get_loc("tickVolume"))
        out = out.drop(columns=["tickVolume"], errors="ignore")
        out.insert(tick_idx, "volume", normalized["volume"].astype(float))
        return out
    out["volume"] = normalized["volume"].astype(float)
    return out


def _build(
    df: pd.DataFrame, *, anchor_idx: int, include_research_only: bool
) -> pd.DataFrame:
    return add_anchored_vwap(
        df,
        anchor_idx=anchor_idx,
        anchor_label="validation_window_start",
        anchor_class="live_safe",
        anchor_origin_idx=anchor_idx,
        anchor_confirm_idx=anchor_idx,
        anchor_live_from_idx=anchor_idx,
        include_research_only=include_research_only,
    )


def _build_anchor_family_context(df: pd.DataFrame) -> pd.DataFrame:
    out = add_atr(df)
    out = add_swings(out, window=6)
    out = add_trend_state(out)
    out = add_bos(out)
    out = add_choch(out)
    # The sweep anchor family was wired against the legacy detector schema
    # (``sweep_confirm_flag`` / ``sweep_detect_idx`` / ``sweep_direction``).
    # Migration to the canonical ``add_final_sweeps`` chain requires the
    # full research pipeline upstream and is tracked separately; the
    # sweep_detect_to_confirm_hybrid family is currently disabled here.
    return out


def _latest_index(mask: pd.Series) -> int | None:
    positions = np.flatnonzero(mask.to_numpy())
    return int(positions[-1]) if len(positions) else None


def _select_anchor_positions(
    candidates: np.ndarray,
    *,
    total_rows: int,
    sample_count: int = FAMILY_SAMPLE_COUNT,
    min_active_rows: int = MIN_FAMILY_ACTIVE_ROWS,
) -> np.ndarray:
    if len(candidates) == 0:
        return np.array([], dtype=int)
    eligible = candidates[(total_rows - candidates) >= min_active_rows]
    pool = eligible if len(eligible) else candidates
    return pool[-min(sample_count, len(pool)) :].astype(int)


def _build_family_frames(df: pd.DataFrame) -> dict[str, list[pd.DataFrame] | None]:
    context = _build_anchor_family_context(df)
    n = len(context)
    family_frames: dict[str, list[pd.DataFrame] | None] = {
        "confirmed_swing_hybrid": None,
        "bos_event_live_safe": None,
        "choch_event_live_safe": None,
        "sweep_detect_to_confirm_hybrid": None,
        "synthetic_sweep_detect_to_confirm_hybrid": None,
    }

    swing_mask = pd.to_numeric(
        context["swing_high_confirm_flag"], errors="coerce"
    ).fillna(0).eq(1) | pd.to_numeric(
        context["swing_low_confirm_flag"], errors="coerce"
    ).fillna(
        0
    ).eq(
        1
    )
    swing_candidates = np.flatnonzero(swing_mask.to_numpy())
    swing_confirm_positions = _select_anchor_positions(swing_candidates, total_rows=n)
    if len(swing_confirm_positions):
        frames: list[pd.DataFrame] = []
        for swing_confirm_idx in swing_confirm_positions:
            row = context.iloc[swing_confirm_idx]
            sh_flag = pd.to_numeric(row["swing_high_confirm_flag"], errors="coerce")
            if np.isfinite(sh_flag) and int(sh_flag) == 1:
                origin_idx = int(row["swing_high_confirm_origin_idx"])
                label = "confirmed_swing_high"
            else:
                origin_idx = int(row["swing_low_confirm_origin_idx"])
                label = "confirmed_swing_low"
            frames.append(
                add_anchored_vwap(
                    context,
                    anchor_idx=origin_idx,
                    anchor_label=label,
                    anchor_class="hybrid",
                    anchor_origin_idx=origin_idx,
                    anchor_confirm_idx=swing_confirm_idx,
                    anchor_live_from_idx=swing_confirm_idx,
                )
            )
        family_frames["confirmed_swing_hybrid"] = frames

    bos_mask = pd.to_numeric(context["bos_bull"], errors="coerce").fillna(0).eq(
        1
    ) | pd.to_numeric(context["bos_bear"], errors="coerce").fillna(0).eq(1)
    bos_candidates = np.flatnonzero(bos_mask.to_numpy())
    bos_positions = _select_anchor_positions(bos_candidates, total_rows=n)
    if len(bos_positions):
        frames = []
        for bos_idx in bos_positions:
            direction = int(
                pd.to_numeric(context.iloc[bos_idx]["bos_direction"], errors="coerce")
            )
            frames.append(
                add_anchored_vwap(
                    context,
                    anchor_idx=bos_idx,
                    anchor_label=(
                        "bos_bull_event" if direction == 1 else "bos_bear_event"
                    ),
                    anchor_class="live_safe",
                    anchor_origin_idx=bos_idx,
                    anchor_confirm_idx=bos_idx,
                    anchor_live_from_idx=bos_idx,
                )
            )
        family_frames["bos_event_live_safe"] = frames

    choch_mask = pd.to_numeric(context["choch_bull"], errors="coerce").fillna(0).eq(
        1
    ) | pd.to_numeric(context["choch_bear"], errors="coerce").fillna(0).eq(1)
    choch_candidates = np.flatnonzero(choch_mask.to_numpy())
    choch_positions = _select_anchor_positions(choch_candidates, total_rows=n)
    if len(choch_positions):
        frames = []
        for choch_idx in choch_positions:
            direction = int(
                pd.to_numeric(
                    context.iloc[choch_idx]["choch_direction"], errors="coerce"
                )
            )
            frames.append(
                add_anchored_vwap(
                    context,
                    anchor_idx=choch_idx,
                    anchor_label=(
                        "choch_bull_event" if direction == 1 else "choch_bear_event"
                    ),
                    anchor_class="live_safe",
                    anchor_origin_idx=choch_idx,
                    anchor_confirm_idx=choch_idx,
                    anchor_live_from_idx=choch_idx,
                )
            )
        family_frames["choch_event_live_safe"] = frames

    # sweep_detect_to_confirm_hybrid family disabled pending migration to
    # canonical sweep schema (see _build_anchor_family_context).

    if n >= 4:
        synthetic_offsets = (40, 80, 120)
        frames = []
        for active_rows in synthetic_offsets:
            synthetic_confirm_idx = max(n - active_rows, 3)
            synthetic_detect_idx = synthetic_confirm_idx - 3
            frames.append(
                add_anchored_vwap(
                    context,
                    anchor_idx=synthetic_detect_idx,
                    anchor_label="synthetic_sweep_bull_detect",
                    anchor_class="hybrid",
                    anchor_origin_idx=synthetic_detect_idx,
                    anchor_confirm_idx=synthetic_confirm_idx,
                    anchor_live_from_idx=synthetic_confirm_idx,
                )
            )
        family_frames["synthetic_sweep_detect_to_confirm_hybrid"] = frames

    return family_frames


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        raw = pd.read_parquet(data_file)

        anchor_idx = max(len(raw) - PLOT_ROWS, 0)
        tick_live = _build(
            _as_tick_only(raw), anchor_idx=anchor_idx, include_research_only=False
        )
        volume_live = _build(
            _as_volume_only(raw), anchor_idx=anchor_idx, include_research_only=False
        )
        source_parity_ok = bool(tick_live.equals(volume_live))

        live_df = tick_live
        research_df = _build(
            _as_tick_only(raw), anchor_idx=anchor_idx, include_research_only=True
        )
        family_frames = _build_family_frames(_as_tick_only(raw))
        plot_df = research_df.tail(PLOT_ROWS).copy()
        html_path = OUT_DIR / f"avwap_validation_{instrument}_{timeframe}.html"
        result = validate_avwap(
            plot_df,
            summary_df=research_df,
            live_df=live_df,
            family_frames=family_frames,
            outpath=html_path,
            title=f"AVWAP Validation — {instrument} {timeframe}",
            source_parity_ok=source_parity_ok,
        )

        print(f"\n=== AVWAP SUMMARY: {instrument} {timeframe} ===")
        _print_summary(result["summary"])
        print(f"Wrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
