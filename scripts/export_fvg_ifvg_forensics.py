from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.indicators.foundation.volatility import add_atr
from src.indicators.research.fvg_research import build_fvg_research_table
from src.indicators.smc.fvg import ALL_FVG_CORE_COLUMNS, collect_fvg_debug_tables
from src.indicators.smc.fvg_fill import FVG_FILL_COLUMNS, add_fvg_fill
from src.indicators.smc.ifvg import ALL_IFVG_COLUMNS, add_ifvg
from src.validation.indicators.fvg import (
    FORENSIC_WINDOW_START,
    _summarize_fill_selection_reconciliation,
)

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/smc/forensics_2026-01-11_plus")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(pd.to_datetime(value, utc=True))
    if value is None or pd.isna(value):
        return None
    return value


def _write_table(base_dir: Path, name: str, df: pd.DataFrame) -> dict[str, str]:
    csv_path = base_dir / f"{name}.csv"
    parquet_path = base_dir / f"{name}.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return {
        "csv": str(csv_path.resolve()),
        "parquet": str(parquet_path.resolve()),
    }


def _build_ifvg_detect_events(window_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for side in ("bull", "bear"):
        detect_mask = (
            pd.to_numeric(window_df[f"ifvg_{side}_detect_flag"], errors="coerce")
            .fillna(0)
            .eq(1)
        )
        scoped = window_df.loc[detect_mask, ["timestamp"]].copy()
        if scoped.empty:
            continue
        side_cols = [
            column for column in window_df.columns if column.startswith(f"ifvg_{side}_")
        ]
        side_frame = pd.concat(
            [scoped, window_df.loc[detect_mask, side_cols].copy()],
            axis=1,
        )
        side_frame.insert(1, "ifvg_side", side)
        side_frame = side_frame.rename(
            columns={
                column: column.replace(f"ifvg_{side}_", "ifvg_") for column in side_cols
            }
        )
        frames.append(side_frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ifvg_detect_idx", "ifvg_event_id", "ifvg_side"]
    )


def _build_fill_count_timeseries(
    *,
    active_members: pd.DataFrame,
    window_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in window_df.itertuples(index=False):
        row_idx = int(row.row_idx)
        scoped = active_members[active_members["row_idx"] == row_idx].copy()
        record: dict[str, object] = {
            "row_idx": row_idx,
            "timestamp": row.timestamp,
        }
        for side in ("bull", "bear"):
            side_members = scoped[scoped["side"] == side].copy()
            live_fill_pct = pd.to_numeric(
                side_members.get("fill_pct"), errors="coerce"
            ).fillna(0.0)
            historical_fill_pct = pd.to_numeric(
                side_members.get("historical_fill_pct"), errors="coerce"
            ).fillna(live_fill_pct)
            live_touch_count = pd.to_numeric(
                side_members.get("touch_count"), errors="coerce"
            ).fillna(0)
            historical_touch_count = pd.to_numeric(
                side_members.get("historical_touch_count"), errors="coerce"
            ).fillna(live_touch_count)

            untouched_mask = live_touch_count.eq(0) & live_fill_pct.eq(0.0)
            touched_only_mask = live_touch_count.gt(0) & live_fill_pct.eq(0.0)
            partial_fill_mask = live_fill_pct.gt(0.0)
            historical_untouched_mask = historical_touch_count.eq(
                0
            ) & historical_fill_pct.eq(0.0)
            historical_touched_only_mask = historical_touch_count.gt(
                0
            ) & historical_fill_pct.eq(0.0)
            historical_partial_fill_mask = historical_fill_pct.gt(0.0)

            record[f"{side}_active_member_count"] = int(len(side_members))
            record[f"{side}_untouched_count"] = int(untouched_mask.sum())
            record[f"{side}_touched_only_count"] = int(touched_only_mask.sum())
            record[f"{side}_partial_fill_count"] = int(partial_fill_mask.sum())
            record[f"{side}_any_fill_count"] = int(partial_fill_mask.sum())
            record[f"{side}_max_fill_pct"] = (
                float(live_fill_pct.max()) if not side_members.empty else np.nan
            )
            record[f"{side}_mean_fill_pct_active_members"] = (
                float(live_fill_pct.mean()) if not side_members.empty else np.nan
            )
            record[f"{side}_historical_untouched_count"] = int(
                historical_untouched_mask.sum()
            )
            record[f"{side}_historical_touched_only_count"] = int(
                historical_touched_only_mask.sum()
            )
            record[f"{side}_historical_partial_fill_count"] = int(
                historical_partial_fill_mask.sum()
            )
            record[f"{side}_historical_max_fill_pct"] = (
                float(historical_fill_pct.max()) if not side_members.empty else np.nan
            )
            record[f"{side}_historical_mean_fill_pct_active_members"] = (
                float(historical_fill_pct.mean()) if not side_members.empty else np.nan
            )

        record["total_active_member_count"] = int(
            record["bull_active_member_count"] + record["bear_active_member_count"]
        )
        record["total_partial_fill_count"] = int(
            record["bull_partial_fill_count"] + record["bear_partial_fill_count"]
        )
        record["total_touched_only_count"] = int(
            record["bull_touched_only_count"] + record["bear_touched_only_count"]
        )
        record["total_untouched_count"] = int(
            record["bull_untouched_count"] + record["bear_untouched_count"]
        )
        record["total_historical_partial_fill_count"] = int(
            record["bull_historical_partial_fill_count"]
            + record["bear_historical_partial_fill_count"]
        )
        rows.append(record)

    return pd.DataFrame(rows)


def _build_plot_surface_columns(df: pd.DataFrame) -> list[str]:
    columns = {
        "row_idx",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "fvg_bull_active_count",
        "fvg_bear_active_count",
        "fvg_bull_active",
        "fvg_bear_active",
    }
    for prefix in (
        "fvg_bull_active_",
        "fvg_bear_active_",
        "fvg_fill_bull_",
        "fvg_fill_bear_",
        "ifvg_bull_",
        "ifvg_bear_",
    ):
        columns.update(column for column in df.columns if column.startswith(prefix))
    columns.update(
        column for column in df.columns if column in IFVG_COMPAT_COLUMNS_PLACEHOLDER
    )
    ordered = [column for column in df.columns if column in columns]
    return ordered


IFVG_COMPAT_COLUMNS_PLACEHOLDER = {
    "ifvg_bull",
    "ifvg_bear",
    "ifvg_width_atr",
    "ifvg_bull_break",
    "ifvg_bear_break",
    "ifvg_bull_candidate",
    "ifvg_bear_candidate",
    "ifvg_bull_retest",
    "ifvg_bear_retest",
    "ifvg_state_bull",
    "ifvg_state_bear",
    "ifvg_origin_idx_bull",
    "ifvg_origin_idx_bear",
}


def _build_fvg_events_touching_window(
    research_table: pd.DataFrame,
    *,
    start_idx: int,
    end_idx: int,
) -> pd.DataFrame:
    if research_table.empty:
        return research_table.copy()

    terminal_idx = pd.to_numeric(
        research_table["fvg_terminal_idx"], errors="coerce"
    ).fillna(float(end_idx))
    mask = (
        pd.to_numeric(research_table["fvg_detect_idx"], errors="coerce") <= end_idx
    ) & (terminal_idx >= start_idx)
    return (
        research_table.loc[mask].copy().sort_values(["fvg_detect_idx", "fvg_event_id"])
    )


def _add_rep_timestamps(
    rep_table: pd.DataFrame,
    *,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    if rep_table.empty:
        return rep_table.copy()
    out = rep_table.copy()
    out["start_ts"] = pd.to_datetime(
        full_df.loc[out["start_idx"].astype(int), "timestamp"].to_numpy(),
        utc=True,
    )
    out["detect_ts"] = pd.to_datetime(
        full_df.loc[out["detect_idx"].astype(int), "timestamp"].to_numpy(),
        utc=True,
    )
    out["end_ts"] = pd.to_datetime(
        full_df.loc[out["end_idx"].astype(int), "timestamp"].to_numpy(),
        utc=True,
    )
    return out


def main() -> None:
    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = add_atr(df)
    debug_tables = collect_fvg_debug_tables(df)
    df = debug_tables["frame"]
    df = add_fvg_fill(df, debug_tables=debug_tables)
    df = add_ifvg(df, debug_tables=debug_tables)
    research_table = build_fvg_research_table(df, debug_tables=debug_tables)

    window_df = df[df["timestamp"] >= FORENSIC_WINDOW_START].copy()
    if window_df.empty:
        raise ValueError(f"No rows found from {FORENSIC_WINDOW_START} onward.")

    window_df.insert(0, "row_idx", window_df.index.astype(int))
    window_start_idx = int(window_df.index.min())
    window_end_idx = int(window_df.index.max())

    plot_surface_columns = _build_plot_surface_columns(window_df)
    plot_surface_df = window_df[plot_surface_columns].copy()

    active_rep_table = debug_tables["active_rep_table"].copy()
    active_rep_window = active_rep_table[
        (
            pd.to_numeric(active_rep_table["end_idx"], errors="coerce")
            >= window_start_idx
        )
        & (
            pd.to_numeric(active_rep_table["start_idx"], errors="coerce")
            <= window_end_idx
        )
    ].copy()
    active_rep_window = _add_rep_timestamps(active_rep_window, full_df=df)
    if not active_rep_window.empty:
        active_rep_window["carried_in"] = (
            pd.to_numeric(active_rep_window["start_idx"], errors="coerce")
            < window_start_idx
        )

    active_member_window = debug_tables["active_member_table"][
        debug_tables["active_member_table"]["row_idx"].between(
            window_start_idx, window_end_idx
        )
    ].copy()
    if not active_member_window.empty:
        active_member_window["in_window"] = 1
    fill_count_timeseries = _build_fill_count_timeseries(
        active_members=active_member_window,
        window_df=window_df,
    )

    fvg_detected_in_window = research_table[
        pd.to_datetime(research_table["fvg_detect_ts"], utc=True)
        >= FORENSIC_WINDOW_START
    ].copy()
    fvg_detected_in_window = fvg_detected_in_window.sort_values(
        ["fvg_detect_idx", "fvg_event_id"]
    )
    fvg_touching_window = _build_fvg_events_touching_window(
        research_table,
        start_idx=window_start_idx,
        end_idx=window_end_idx,
    )

    ifvg_detected_in_window = _build_ifvg_detect_events(window_df)

    fill_selection_reconciliation = _summarize_fill_selection_reconciliation(
        df, active_members=debug_tables["active_member_table"]
    )
    bull_row_audit = pd.DataFrame(
        (fill_selection_reconciliation or {})
        .get("bull", {})
        .get("row_level_mismatch_audit", [])
    )
    bear_row_audit = pd.DataFrame(
        (fill_selection_reconciliation or {})
        .get("bear", {})
        .get("row_level_mismatch_audit", [])
    )
    if not bull_row_audit.empty:
        bull_row_audit = bull_row_audit[
            bull_row_audit["row_idx"].between(window_start_idx, window_end_idx)
        ].copy()
    if not bear_row_audit.empty:
        bear_row_audit = bear_row_audit[
            bear_row_audit["row_idx"].between(window_start_idx, window_end_idx)
        ].copy()

    counts = {
        "window": {
            "start": str(pd.to_datetime(window_df["timestamp"], utc=True).min()),
            "end": str(pd.to_datetime(window_df["timestamp"], utc=True).max()),
            "start_idx": window_start_idx,
            "end_idx": window_end_idx,
            "row_count": int(len(window_df)),
        },
        "fvg_counts": {
            "detected_total": int(len(fvg_detected_in_window)),
            "detected_bull": int((fvg_detected_in_window["fvg_side"] == "bull").sum()),
            "detected_bear": int((fvg_detected_in_window["fvg_side"] == "bear").sum()),
            "touching_window_total": int(len(fvg_touching_window)),
            "touching_window_bull": int(
                (fvg_touching_window["fvg_side"] == "bull").sum()
            ),
            "touching_window_bear": int(
                (fvg_touching_window["fvg_side"] == "bear").sum()
            ),
            "active_reps_intersecting_window_total": int(len(active_rep_window)),
            "active_reps_intersecting_window_bull": int(
                (active_rep_window["side"] == "bull").sum()
            ),
            "active_reps_intersecting_window_bear": int(
                (active_rep_window["side"] == "bear").sum()
            ),
            "carried_in_active_reps_total": int(
                active_rep_window.get("carried_in", pd.Series(dtype=bool))
                .fillna(False)
                .sum()
            ),
            "carried_in_active_reps_bull": int(
                (
                    (
                        active_rep_window.get(
                            "carried_in", pd.Series(dtype=bool)
                        ).fillna(False)
                    )
                    & (active_rep_window.get("side", pd.Series(dtype=object)) == "bull")
                ).sum()
            ),
            "carried_in_active_reps_bear": int(
                (
                    (
                        active_rep_window.get(
                            "carried_in", pd.Series(dtype=bool)
                        ).fillna(False)
                    )
                    & (active_rep_window.get("side", pd.Series(dtype=object)) == "bear")
                ).sum()
            ),
        },
        "ifvg_counts": {
            "detected_total": int(len(ifvg_detected_in_window)),
            "detected_bull": int(
                (
                    ifvg_detected_in_window.get("ifvg_side", pd.Series(dtype=object))
                    == "bull"
                ).sum()
            ),
            "detected_bear": int(
                (
                    ifvg_detected_in_window.get("ifvg_side", pd.Series(dtype=object))
                    == "bear"
                ).sum()
            ),
            "active_rows_bull": int(
                pd.to_numeric(window_df["ifvg_bull_active"], errors="coerce")
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "active_rows_bear": int(
                pd.to_numeric(window_df["ifvg_bear_active"], errors="coerce")
                .fillna(0)
                .eq(1)
                .sum()
            ),
        },
        "fill_selection_counts": {
            "bull_rows_where_fill_rep_differs_from_structural_selected": int(
                len(bull_row_audit)
            ),
            "bear_rows_where_fill_rep_differs_from_structural_selected": int(
                len(bear_row_audit)
            ),
            "bull_rows_where_visual_confusion_pattern_occurs": int(len(bull_row_audit)),
            "bear_rows_where_visual_confusion_pattern_occurs": int(len(bear_row_audit)),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "window_indicator_frame": _write_table(
            OUT_DIR,
            "window_indicator_frame_2026_01_11_plus",
            window_df,
        ),
        "window_plot_surface": _write_table(
            OUT_DIR,
            "window_plot_surface_2026_01_11_plus",
            plot_surface_df,
        ),
        "fvg_active_reps_intersecting_window": _write_table(
            OUT_DIR,
            "fvg_active_reps_intersecting_window_2026_01_11_plus",
            active_rep_window,
        ),
        "fvg_active_members_window": _write_table(
            OUT_DIR,
            "fvg_active_members_window_2026_01_11_plus",
            active_member_window,
        ),
        "fvg_fill_count_timeseries_window": _write_table(
            OUT_DIR,
            "fvg_fill_count_timeseries_window_2026_01_11_plus",
            fill_count_timeseries,
        ),
        "fvg_events_detected_in_window": _write_table(
            OUT_DIR,
            "fvg_events_detected_in_window_2026_01_11_plus",
            fvg_detected_in_window,
        ),
        "fvg_events_touching_window": _write_table(
            OUT_DIR,
            "fvg_events_touching_window_2026_01_11_plus",
            fvg_touching_window,
        ),
        "ifvg_events_detected_in_window": _write_table(
            OUT_DIR,
            "ifvg_events_detected_in_window_2026_01_11_plus",
            ifvg_detected_in_window,
        ),
        "fill_selection_row_audit_bull": _write_table(
            OUT_DIR,
            "fill_selection_row_audit_bull_2026_01_11_plus",
            bull_row_audit,
        ),
        "fill_selection_row_audit_bear": _write_table(
            OUT_DIR,
            "fill_selection_row_audit_bear_2026_01_11_plus",
            bear_row_audit,
        ),
    }

    manifest = {
        "window_start": str(FORENSIC_WINDOW_START),
        "counts": counts,
        "files": outputs,
        "column_families": {
            "window_indicator_frame": {
                "description": "Full per-bar indicator frame from Jan 11, 2026 onward, including all FVG core, FVG fill, and IFVG columns.",
                "fvg_core_column_count": len(ALL_FVG_CORE_COLUMNS),
                "fvg_fill_column_count": len(FVG_FILL_COLUMNS),
                "ifvg_column_count": len(ALL_IFVG_COLUMNS),
            },
            "window_plot_surface": {
                "description": "Per-bar subset of the exact rowwise fields used by the validation chart layers and hover payloads.",
                "columns": plot_surface_columns,
            },
            "fvg_active_reps_intersecting_window": {
                "description": "Underlying FVG rep rectangles used for the light structural boxes, including detect/start/end/terminal metadata.",
            },
            "fvg_active_members_window": {
                "description": "Per-row core active-member truth for every active FVG member in the window, including live fill/touch fields, historical fill/touch fields, ownership, and selected flags.",
            },
            "fvg_fill_count_timeseries_window": {
                "description": "Per-bar counts of active FVG members by fill status in the window for both live and historical semantics: untouched, wick-touched-only, and partially filled, for bull, bear, and totals.",
            },
            "fvg_events_detected_in_window": {
                "description": "FVG detect/research events whose detect timestamp is on or after Jan 11, 2026.",
            },
            "fvg_events_touching_window": {
                "description": "FVG events whose lifecycle intersects the forensic window, including carried-in events detected earlier.",
            },
            "ifvg_events_detected_in_window": {
                "description": "Canonical IFVG detect events in the window, with detect, geometry, inversion-quality, and lifecycle columns.",
            },
            "fill_selection_row_audit_bull": {
                "description": "Rows where bull structural selected and fill rep differ, including both structural and fill remaining geometry.",
            },
            "fill_selection_row_audit_bear": {
                "description": "Rows where bear structural selected and fill rep differ, including both structural and fill remaining geometry.",
            },
        },
    }

    manifest_path = OUT_DIR / "forensics_manifest_2026_01_11_plus.json"
    counts_path = OUT_DIR / "forensics_counts_2026_01_11_plus.json"
    counts_csv_path = OUT_DIR / "forensics_counts_2026_01_11_plus.csv"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2))
    counts_path.write_text(json.dumps(_json_safe(counts), indent=2))
    counts_rows = []
    for section, payload in counts.items():
        for key, value in payload.items():
            counts_rows.append({"section": section, "metric": key, "value": value})
    pd.DataFrame(counts_rows).to_csv(counts_csv_path, index=False)

    print("\n=== FORENSIC EXPORT COMPLETE ===")
    print(f"window_start: {FORENSIC_WINDOW_START}")
    print(f"window_rows: {len(window_df)}")
    print(
        "fvg_detected_in_window: "
        f"{counts['fvg_counts']['detected_total']} "
        f"(bull={counts['fvg_counts']['detected_bull']}, "
        f"bear={counts['fvg_counts']['detected_bear']})"
    )
    print(
        "ifvg_detected_in_window: "
        f"{counts['ifvg_counts']['detected_total']} "
        f"(bull={counts['ifvg_counts']['detected_bull']}, "
        f"bear={counts['ifvg_counts']['detected_bear']})"
    )
    print(
        "fill_rep_mismatch_rows: "
        f"bull={counts['fill_selection_counts']['bull_rows_where_fill_rep_differs_from_structural_selected']}, "
        f"bear={counts['fill_selection_counts']['bear_rows_where_fill_rep_differs_from_structural_selected']}"
    )
    print(f"manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
