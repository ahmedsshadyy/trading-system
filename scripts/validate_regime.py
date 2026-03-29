from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src.indicators.foundation.adx import add_adx
from src.indicators.foundation.ema import add_emas
from src.indicators.foundation.regime import add_regime
from src.indicators.foundation.session import add_session_features
from src.indicators.foundation.volatility import add_atr, add_bb_width
from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state
from src.validation.indicators.regime import validate_regime

OUT_DIR = Path("notebooks/foundation")
PLOT_ROWS = 300
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


def _make_regime_input_df(n: int = 12) -> pd.DataFrame:
    close = pd.Series(np.linspace(100.0, 105.0, n), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": close - 0.3,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "atr_14": 2.0,
            "adx_14": 25.0,
            "bb_width": 0.05,
            "bb_width_pct_rank_100": 0.50,
            "ema_20_slope": 0.2,
            "ema_20_slope_atr": 0.2,
            "trend_state": 1,
            "trend_confidence": 2,
            "hh_count": 3,
            "ll_count": 0,
            "trend_bias_state": 1,
            "trend_strength_ema": 0.5,
        }
    )


def _synthetic_fixture_summary() -> dict[str, object]:
    fixtures: dict[str, tuple[pd.DataFrame, int]] = {}

    ranging = _make_regime_input_df(6)
    ranging["adx_14"] = 12.0
    ranging["bb_width_pct_rank_100"] = 0.05
    ranging["ema_20_slope_atr"] = 0.02
    ranging["trend_state"] = 0
    ranging["trend_confidence"] = 0
    ranging["hh_count"] = 0
    ranging["ll_count"] = 0
    ranging["trend_bias_state"] = 0
    fixtures["clean_ranging_fixture"] = (ranging, 0)

    trending = _make_regime_input_df(6)
    trending["adx_14"] = 38.0
    trending["bb_width_pct_rank_100"] = 0.85
    trending["ema_20_slope_atr"] = 0.50
    trending["trend_state"] = 1
    trending["trend_confidence"] = 2
    trending["hh_count"] = 4
    fixtures["clean_trending_fixture"] = (trending, 2)

    mixed = _make_regime_input_df(6)
    mixed["adx_14"] = 26.0
    mixed["bb_width_pct_rank_100"] = 0.45
    mixed["ema_20_slope_atr"] = 0.15
    mixed["trend_state"] = 1
    mixed["trend_confidence"] = 0.5
    mixed["hh_count"] = 1
    mixed["ll_count"] = 0
    fixtures["mixed_fixture"] = (mixed, 1)

    sequence = pd.concat(
        [
            ranging.assign(
                timestamp=pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
            ),
            mixed.assign(
                timestamp=pd.date_range(
                    "2024-01-01 06:00", periods=6, freq="1h", tz="UTC"
                )
            ),
            trending.assign(
                timestamp=pd.date_range(
                    "2024-01-01 12:00", periods=6, freq="1h", tz="UTC"
                )
            ),
        ],
        ignore_index=True,
    )
    fixtures["transition_sequence_fixture"] = (sequence, -1)

    passed = 0
    total = 0
    details: dict[str, object] = {}
    for name, (fixture_df, expected_regime) in fixtures.items():
        total += 1
        result = add_regime(fixture_df)
        valid = pd.to_numeric(result["regime"], errors="coerce").dropna().astype(int)
        if name == "transition_sequence_fixture":
            ok = (
                result.loc[0, "regime_enter_ranging"] == 1
                and result.loc[6, "regime_enter_transitional"] == 1
                and result.loc[12, "regime_enter_trending"] == 1
                and result.loc[6, "bars_in_regime"] == 1
                and result.loc[12, "bars_in_regime"] == 1
            )
            observed = [
                int(x)
                for x in pd.to_numeric(result["regime"], errors="coerce")
                .dropna()
                .tolist()
            ]
        else:
            ok = (not valid.empty) and bool((valid == expected_regime).all())
            observed = sorted(valid.unique().tolist())
        passed += int(ok)
        details[name] = {"passed": bool(ok), "observed": observed}

    return {"passed": passed, "total": total, "details": details}


def _build_context(df: pd.DataFrame, *, include_research_only: bool) -> pd.DataFrame:
    out = add_atr(df)
    out = add_emas(out)
    out = add_adx(out)
    out = add_bb_width(out)
    out = add_swings(out, window=6)
    out = add_trend_state(out)

    ts = pd.to_datetime(out["timestamp"], utc=True)
    if ts.diff().median().total_seconds() < 86400:
        out = add_session_features(out, include_research_only=include_research_only)

    out = add_regime(out, include_research_only=include_research_only)
    return out


def _print_validation_sections(summary: dict[str, object]) -> None:
    ordered_keys = [
        "current_regime_snapshot",
        "downstream_caution_contract",
        "warmup",
        "value_counts",
        "alignment_rates",
        "per_regime_alignment",
        "extreme_consistency",
        "unaligned_decomposition",
        "extreme_misalignment_audit",
        "extreme_misalignment_profiles",
        "trend_state_confusion_matrix",
        "trend_bias_confusion_matrix",
        "transition_matrix",
        "flicker_diagnostics",
        "boundary_diagnostics",
        "raw_vs_stabilized_audit",
        "synthetic_fixture_summary",
    ]
    print(f"row_count: {summary['row_count']}")
    print(f"valid_regime_row_count: {summary['valid_regime_row_count']}")
    print(f"regime_change_count: {summary['regime_change_count']}")
    for key in ordered_keys:
        if key in summary:
            print(f"{key}:")
            _print_summary(summary[key], indent=2)
    print("checks:")
    _print_summary(summary["checks"], indent=2)
    print("summary_stats:")
    _print_summary(summary["summary_stats"], indent=2)
    if "regime_by_session" in summary:
        print("regime_by_session:")
        _print_summary(summary["regime_by_session"], indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_summary = _synthetic_fixture_summary()

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        raw = pd.read_parquet(data_file)

        live_df = _build_context(raw, include_research_only=False)
        research_df = _build_context(raw, include_research_only=True)
        plot_df = research_df.tail(PLOT_ROWS).copy()

        html_path = OUT_DIR / f"regime_validation_{instrument}_{timeframe}.html"
        result = validate_regime(
            plot_df,
            summary_df=research_df,
            live_df=live_df,
            research_df=research_df,
            outpath=html_path,
            title=f"Regime Validation — {instrument} {timeframe}",
            synthetic_summary=synthetic_summary,
        )

        print(f"\n=== REGIME SUMMARY: {instrument} {timeframe} ===")
        _print_validation_sections(result["summary"])
        print(f"Wrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
