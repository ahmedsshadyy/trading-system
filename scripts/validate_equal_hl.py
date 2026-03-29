from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.research.equal_hl_research import build_equal_hl_research_table
from src.validation.indicators.equal_hl import validate_equal_hl

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/smc")
HTML_FILE = OUT_DIR / "equal_hl_validation_XAU_USD_H4.html"
STRONGEST_CSV = OUT_DIR / "equal_hl_strongest_clusters_XAU_USD_H4.csv"
STRUCTURAL_CSV = OUT_DIR / "equal_hl_structural_clusters_XAU_USD_H4.csv"
FRESH_TRADEABLE_CSV = OUT_DIR / "equal_hl_fresh_tradeable_XAU_USD_H4.csv"
BORDERLINE_CSV = OUT_DIR / "equal_hl_borderline_wide_XAU_USD_H4.csv"
STALE_CSV = OUT_DIR / "equal_hl_old_stale_XAU_USD_H4.csv"
CALIBRATION_EVENTS_CSV = OUT_DIR / "equal_hl_calibration_events_XAU_USD_H4.csv"
ACTIVE_SNAPSHOTS_CSV = OUT_DIR / "equal_hl_active_snapshots_XAU_USD_H4.csv"
COMPONENT_AUDIT_CSV = OUT_DIR / "equal_hl_component_direction_audit_XAU_USD_H4.csv"
COARSE_CANDIDATES_CSV = OUT_DIR / "equal_hl_coarse_candidates_XAU_USD_H4.csv"
REFINED_CANDIDATES_CSV = OUT_DIR / "equal_hl_refined_candidates_XAU_USD_H4.csv"
RESEARCH_STRONGEST_CSV = OUT_DIR / "equal_hl_research_strongest_XAU_USD_H4.csv"
RESEARCH_FAILURE_CSV = OUT_DIR / "equal_hl_research_failures_XAU_USD_H4.csv"
PLOT_START = pd.Timestamp("2026-01-11", tz="UTC")


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
    if isinstance(value, list):
        for item in value:
            print(f"{prefix}- {item}")
        return
    print(f"{prefix}{value}")


def main() -> None:
    raw_df = pd.read_parquet(DATA_FILE)
    df = build_research_indicators(
        raw_df,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
    )
    research_table = build_equal_hl_research_table(df)

    plot_df = df[df["timestamp"] >= PLOT_START].copy()
    if plot_df.empty:
        raise ValueError(
            f"No XAU_USD H4 rows found for validation window starting {PLOT_START}."
        )

    result = validate_equal_hl(
        plot_df,
        full_df=df,
        outpath=HTML_FILE,
        title=(
            "Equal H/L Validation — XAU_USD H4 "
            f"({PLOT_START.date()} to {plot_df['timestamp'].max().date()}, chart window only)"
        ),
        research_table=research_table,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result["tables"]["strongest_clusters"].to_csv(STRONGEST_CSV, index=False)
    result["tables"]["structurally_strongest_clusters"].to_csv(
        STRUCTURAL_CSV, index=False
    )
    result["tables"]["fresh_tradeable_clusters"].to_csv(
        FRESH_TRADEABLE_CSV, index=False
    )
    result["tables"]["borderline_wide_clusters"].to_csv(BORDERLINE_CSV, index=False)
    result["tables"]["old_stale_clusters"].to_csv(STALE_CSV, index=False)
    result["tables"]["calibration_events"].to_csv(CALIBRATION_EVENTS_CSV, index=False)
    result["tables"]["active_snapshots"].to_csv(ACTIVE_SNAPSHOTS_CSV, index=False)
    result["tables"]["component_direction_audit"].to_csv(
        COMPONENT_AUDIT_CSV, index=False
    )
    result["tables"]["coarse_score_candidates"].to_csv(
        COARSE_CANDIDATES_CSV, index=False
    )
    result["tables"]["refined_score_candidates"].to_csv(
        REFINED_CANDIDATES_CSV, index=False
    )
    result["tables"]["strongest_research"].to_csv(RESEARCH_STRONGEST_CSV, index=False)
    result["tables"]["strongest_failures"].to_csv(RESEARCH_FAILURE_CSV, index=False)

    print("\n=== EQUAL H/L SUMMARY ===")
    _print_summary(result["summary"])
    print(f"\nWrote chart to: {result['html_path']}")
    print(f"Wrote strongest clusters to: {STRONGEST_CSV}")
    print(f"Wrote structural clusters to: {STRUCTURAL_CSV}")
    print(f"Wrote fresh tradeable clusters to: {FRESH_TRADEABLE_CSV}")
    print(f"Wrote borderline wide clusters to: {BORDERLINE_CSV}")
    print(f"Wrote old/stale clusters to: {STALE_CSV}")
    print(f"Wrote calibration events to: {CALIBRATION_EVENTS_CSV}")
    print(f"Wrote active snapshots to: {ACTIVE_SNAPSHOTS_CSV}")
    print(f"Wrote component direction audit to: {COMPONENT_AUDIT_CSV}")
    print(f"Wrote coarse candidates to: {COARSE_CANDIDATES_CSV}")
    print(f"Wrote refined candidates to: {REFINED_CANDIDATES_CSV}")
    print(f"Wrote strongest research table to: {RESEARCH_STRONGEST_CSV}")
    print(f"Wrote research failures table to: {RESEARCH_FAILURE_CSV}")


if __name__ == "__main__":
    main()
