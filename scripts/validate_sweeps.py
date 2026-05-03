"""Canonical sweep validation runner.

Drives the canonical sweep stack
(``add_unified_liquidity_sources`` → ``add_final_sweeps``) on each
configured (instrument, timeframe) and emits the 18-point report
defined in ``docs/indicator_contracts/sweeps.md``.

Usage::

    poetry run python scripts/validate_sweeps.py
    poetry run python scripts/validate_sweeps.py --instrument USOIL --timeframe H4
    poetry run python scripts/validate_sweeps.py --runs XAU_USD:H4,USOIL:H4

Outputs per (instrument, timeframe), under ``--out-dir``:

* stdout summary (the 18-point report + headline acceptance gate)
* CSV: ``sweeps_canonical_events_{instrument}_{timeframe}.csv``
* JSON: ``sweeps_canonical_report_{instrument}_{timeframe}.json``

Exits with status 1 if any acceptance gate failed for any run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.smc.sweeps.final_sweeps import (
    FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS,
    FINAL_SWEEPS_COLUMNS,
)
from src.validation.indicators.canonical_sweeps import (
    build_canonical_sweeps_report,
    report_passed,
)

DEFAULT_OUT_DIR = Path("notebooks/smc")
DEFAULT_RUNS = (
    ("XAU_USD", "H4"),
    ("XAU_USD", "H1"),
    ("USOIL", "H4"),
    ("EUR_USD", "H4"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instrument", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument(
        "--runs",
        default=None,
        help=(
            "Comma-separated INSTRUMENT:TIMEFRAME tuples, e.g. "
            "XAU_USD:H4,USOIL:H4. Overrides --instrument/--timeframe."
        ),
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for fast local runs.",
    )
    return p.parse_args()


def _resolve_runs(args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    if args.runs:
        out: list[tuple[str, str]] = []
        for chunk in args.runs.split(","):
            parts = chunk.strip().split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"--runs entry must be INSTRUMENT:TIMEFRAME, got {chunk!r}"
                )
            out.append((parts[0], parts[1]))
        return out
    if args.instrument and args.timeframe:
        return [(args.instrument, args.timeframe)]
    return list(DEFAULT_RUNS)


def _print_summary_block(value: object, *, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                print(f"{prefix}{key}:")
                _print_summary_block(child, indent=indent + 2)
            elif isinstance(child, list):
                if not child:
                    print(f"{prefix}{key}: <empty>")
                else:
                    print(f"{prefix}{key}: {child}")
            else:
                if isinstance(child, float):
                    formatted = f"{child:.4f}" if child == child else "nan"
                else:
                    formatted = str(child)
                print(f"{prefix}{key}: {formatted}")
        return
    print(f"{prefix}{value}")


def _print_18_point_report(summary: dict[str, object]) -> None:
    print("\n=== CANONICAL SWEEPS — 18-POINT VALIDATION ===")
    print(f"  1. total sweep count: {summary['total_sweep_count']}")
    print(f"  2. bullish sweep count: {summary['bullish_sweep_count']}")
    print(f"  3. bearish sweep count: {summary['bearish_sweep_count']}")
    print("  4. counts by source family:")
    _print_summary_block(summary.get("by_source_family", {}), indent=6)
    print("  5. counts by selectivity class:")
    _print_summary_block(summary.get("by_selectivity_class", {}), indent=6)
    print("  6. counts by session phase:")
    _print_summary_block(summary.get("by_session_phase", {}), indent=6)
    print("  7. counts by regime label:")
    _print_summary_block(summary.get("by_regime_label", {}), indent=6)
    print("  8. counts by volume_confirmed flag:")
    _print_summary_block(summary.get("by_volume_confirmed", {}), indent=6)
    print("  9. counts by displacement_confirmed flag:")
    _print_summary_block(summary.get("by_displacement_confirmed", {}), indent=6)
    print(" 10. sweep_breach_atr distribution:")
    _print_summary_block(summary.get("breach_atr_distribution", {}), indent=6)
    print(" 11. sweep_close_reclaim_atr distribution:")
    _print_summary_block(summary.get("close_reclaim_atr_distribution", {}), indent=6)
    print(" 12. sweep_distance_at_start_atr distribution:")
    _print_summary_block(
        summary.get("distance_at_start_atr_distribution", {}), indent=6
    )
    print(
        f" 13. sweeps with valid source metadata: "
        f"{summary['valid_source_metadata_pct']:.2f}%"
    )
    print(" 14. share by source family (%):")
    _print_summary_block(summary.get("source_family_share_pct", {}), indent=6)
    print(" 15-18. acceptance gates:")
    _print_summary_block(summary.get("schema_invariants", {}), indent=6)
    print("       causality_violations:")
    _print_summary_block(summary.get("causality_violations", {}), indent=8)
    print(
        f"       future_columns_required: "
        f"{summary.get('future_columns_required', [])}"
    )
    print(
        f"       upstream_origin_idx_invalid_count: "
        f"{summary.get('upstream_origin_idx_invalid_count', 0)} "
        f"(diagnostic; sanitized in canonical alias — does not gate)"
    )


def _write_outputs(
    df: pd.DataFrame,
    summary: dict[str, object],
    *,
    out_dir: Path,
    instrument: str,
    timeframe: str,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_mask = pd.to_numeric(df["sweep_flag"], errors="coerce").fillna(0) > 0

    keep_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "atr_14",
    ]
    keep_cols += [
        c
        for c in (
            *FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS,
            *(
                c
                for c in FINAL_SWEEPS_COLUMNS
                if c not in FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS
            ),
        )
        if c in df.columns
    ]
    csv_path = out_dir / f"sweeps_canonical_events_{instrument}_{timeframe}.csv"
    df.loc[sweep_mask, keep_cols].to_csv(csv_path, index=False)

    json_path = out_dir / f"sweeps_canonical_report_{instrument}_{timeframe}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    return {"events": csv_path, "report": json_path}


def _run_single(
    instrument: str,
    timeframe: str,
    *,
    out_dir: Path,
    max_rows: int | None,
) -> int:
    data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
    if not data_file.exists():
        print(f"  SKIP: input data not found: {data_file}")
        return 0

    raw = pd.read_parquet(data_file)
    if max_rows is not None:
        raw = raw.tail(int(max_rows)).reset_index(drop=True)
    df = build_research_indicators(
        raw,
        instrument=instrument,
        include_vp=False,
        include_avwap=False,
        timeframe=timeframe,
    )

    print(
        f"\n=== canonical sweeps validation — {instrument} {timeframe} "
        f"(rows={len(df)}) ==="
    )
    summary = build_canonical_sweeps_report(df)
    _print_18_point_report(summary)
    paths = _write_outputs(
        df, summary, out_dir=out_dir, instrument=instrument, timeframe=timeframe
    )
    print(f"\nWrote canonical events table → {paths['events']}")
    print(f"Wrote canonical report (json) → {paths['report']}")

    if not report_passed(summary):
        print("\nVALIDATION FAILED: acceptance-gate violations present.")
        return 1
    print("\nVALIDATION OK: all canonical acceptance gates passed.")
    return 0


def main() -> int:
    args = parse_args()
    runs = _resolve_runs(args)
    out_dir = Path(args.out_dir)
    failures = 0
    for instrument, timeframe in runs:
        failures += _run_single(
            instrument,
            timeframe,
            out_dir=out_dir,
            max_rows=args.max_rows,
        )
    if failures > 0:
        print(f"\n{failures} run(s) failed acceptance gates.")
        return 1
    print("\nAll runs passed canonical sweep acceptance gates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
