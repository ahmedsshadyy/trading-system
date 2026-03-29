from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.research.displacement_research import (
    build_displacement_research_table,
)
from src.validation.indicators.displacement import (
    DISPLACEMENT_DEFAULT_THRESHOLDS,
    build_displacement_analysis_base_frame,
    build_displacement_analysis_frame,
    extract_displacement_overlap_tables,
    summarize_displacement,
)

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/smc")
SUMMARY_JSON = OUT_DIR / "displacement_overlap_summary_XAU_USD_H4.json"
BOS_EVENTS_CSV = OUT_DIR / "displacement_overlap_bos_events_XAU_USD_H4.csv"
CHOCH_EVENTS_CSV = OUT_DIR / "displacement_overlap_choch_events_XAU_USD_H4.csv"
TOP_QUARTILE_CSV = OUT_DIR / "displacement_overlap_top_quartile_XAU_USD_H4.csv"


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


def main() -> None:
    raw_df = pd.read_parquet(DATA_FILE)
    base_df = build_displacement_analysis_base_frame(raw_df, instrument="XAU_USD")
    df = build_displacement_analysis_frame(base_df, **DISPLACEMENT_DEFAULT_THRESHOLDS)
    research = build_displacement_research_table(
        df,
        atr_length=DISPLACEMENT_DEFAULT_THRESHOLDS["atr_length"],
    )

    summary = summarize_displacement(
        df,
        research_table=research,
        **DISPLACEMENT_DEFAULT_THRESHOLDS,
    )
    overlap_tables = extract_displacement_overlap_tables(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BOS_EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    overlap_tables["bos_events"].to_csv(BOS_EVENTS_CSV, index=False)
    overlap_tables["choch_events"].to_csv(CHOCH_EVENTS_CSV, index=False)
    overlap_tables["top_quartile_summary"].to_csv(TOP_QUARTILE_CSV, index=False)

    export_summary = {
        "dataset": str(DATA_FILE),
        "thresholds": DISPLACEMENT_DEFAULT_THRESHOLDS,
        "candidate_comparison": summary.get("candidate_comparison", {}),
        "overlap_summary": summary.get("overlap_summary", {}),
        "recommended_excursion_ratio_variant": summary.get("research_summary", {}).get(
            "recommended_excursion_ratio_variant",
            "capped",
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(export_summary, indent=2, default=str))

    print("\n=== DISPLACEMENT OVERLAP SUMMARY ===")
    _print_summary(export_summary)
    print(f"\nWrote summary JSON to: {SUMMARY_JSON}")
    print(f"Wrote BOS overlap events to: {BOS_EVENTS_CSV}")
    print(f"Wrote CHoCH overlap events to: {CHOCH_EVENTS_CSV}")
    print(f"Wrote top-quartile overlap summary to: {TOP_QUARTILE_CSV}")


if __name__ == "__main__":
    main()
