"""Validation runner for the unified liquidity source framework (Step 10).

Usage::

    poetry run python scripts/validate_unified_sources.py
    poetry run python scripts/validate_unified_sources.py --instrument USOIL --timeframe H4

Outputs:
* stdout summary (sweep prerequisites, MTF policy stamp, family/age/strength
  distributions, dropped-by-crowding/dominance counters, deprecated-family
  presence check)
* CSV: ``notebooks/sweeps_v2/unified_sources_audit_{instrument}_{timeframe}.csv``
* CSV: ``notebooks/sweeps_v2/unified_sources_top_clusters_{instrument}_{timeframe}.csv``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.smc.sweeps.unified_sources import (
    build_unified_liquidity_clusters_audit,
)
from src.validation.indicators.unified_sources import (
    print_unified_sources_summary,
    summarize_unified_sources,
)

DEFAULT_OUT_DIR = Path("notebooks/sweeps_v2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--timeframe", default="H4")
    p.add_argument(
        "--data-file",
        default=None,
        help="Override the parquet path. Defaults to data/raw/{instrument}_{timeframe}.parquet.",
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for fast local runs.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    instrument = args.instrument
    timeframe = args.timeframe
    data_file = (
        Path(args.data_file)
        if args.data_file
        else Path(f"data/raw/{instrument}_{timeframe}.parquet")
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        print(f"ERROR: input data not found: {data_file}")
        return 2

    raw = pd.read_parquet(data_file)
    if args.max_rows is not None:
        raw = raw.tail(int(args.max_rows)).reset_index(drop=True)

    df = build_research_indicators(
        raw,
        instrument=instrument,
        include_vp=False,
        include_avwap=False,
        timeframe=timeframe,
    )

    summary = summarize_unified_sources(df, scan_timeframe=timeframe)
    print(
        f"\n=== unified liquidity sources — {instrument} {timeframe} "
        f"(rows={len(df)}) ===\n"
    )
    print_unified_sources_summary(summary)

    audit = build_unified_liquidity_clusters_audit(df)
    audit_csv = out_dir / f"unified_sources_audit_{instrument}_{timeframe}.csv"
    audit.to_csv(audit_csv, index=False)
    print(f"\nWrote audit table → {audit_csv} ({len(audit)} rows)")

    # Top clusters by strength (per-bar slot 1, both sides), useful for chart
    # overlays.
    top_rows = (
        audit[audit["rank"] == 1].sort_values("strength", ascending=False).head(500)
    )
    top_csv = out_dir / f"unified_sources_top_clusters_{instrument}_{timeframe}.csv"
    top_rows.to_csv(top_csv, index=False)
    print(f"Wrote top-clusters table → {top_csv} ({len(top_rows)} rows)")

    return 0 if summary.get("causality_violations", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
