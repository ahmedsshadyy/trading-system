from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.features.cross_asset import GLOBAL_CONTEXT_SYMBOL
from src.indicators.pipelines.build_research import materialize_research_features
from src.indicators.research.cross_asset_research import (
    build_cross_asset_correlation_audit,
)
from src.pipeline_runtime import load_partitioned_dataset
from src.validation.indicators.cross_asset import validate_cross_asset

OUT_DIR = Path("notebooks/cross_asset")
FEATURES_ROOT = Path("data/validation_cache/features")
RUNS = (
    ("XAU_USD", "H1"),
    ("XAU_USD", "H4"),
    ("USOIL", "H1"),
    ("USOIL", "H4"),
    ("DXY", "H1"),
    ("DXY", "H4"),
)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate research-grade cross-asset matrices and lead-lag outputs."
    )
    parser.add_argument(
        "--instrument",
        choices=("all", "XAU_USD", "USOIL", "DXY"),
        default="all",
    )
    parser.add_argument(
        "--timeframe",
        choices=("all", "H1", "H4"),
        default="all",
    )
    parser.add_argument(
        "--input-rows",
        type=int,
        default=2500,
        help="Recent raw bars to load before running the pipeline. Use 0 for full file.",
    )
    parser.add_argument(
        "--features-root",
        default=str(FEATURES_ROOT),
        help="Feature/state root used for incremental validation materialization.",
    )
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Skip HTML generation and print summaries only.",
    )
    parser.add_argument(
        "--force-graph-recompute",
        type=_parse_bool,
        default=False,
        help="When true, bypass DAG cache and force a fresh graph recompute.",
    )
    return parser.parse_args()


def _selected_runs(args: argparse.Namespace) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for instrument, timeframe in RUNS:
        if args.instrument != "all" and instrument != args.instrument:
            continue
        if args.timeframe != "all" and timeframe != args.timeframe:
            continue
        selected.append((instrument, timeframe))
    return selected


def main() -> None:
    args = _parse_args()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features_root = Path(args.features_root)
    features_root.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in _selected_runs(args):
        raw = pd.read_parquet(Path(f"data/raw/{instrument}_{timeframe}.parquet"))
        if args.input_rows > 0:
            raw = raw.tail(args.input_rows).reset_index(drop=True)
        materialized = materialize_research_features(
            raw,
            instrument=instrument,
            timeframe=timeframe,
            include_vp=False,
            include_avwap=False,
            include_cross_asset=True,
            features_root=str(features_root),
            raw_data_root="data/raw",
            force_graph_recompute=args.force_graph_recompute,
        )
        market_context = load_partitioned_dataset(
            features_root,
            dataset="market_context_research",
            symbol=GLOBAL_CONTEXT_SYMBOL,
            timeframe=timeframe,
        )
        audit_tables = build_cross_asset_correlation_audit(
            materialized.frame,
            market_context,
            instrument=instrument,
            timeframe=timeframe,
        )
        outpath = OUT_DIR / f"cross_asset_validation_{instrument}_{timeframe}.html"
        result = validate_cross_asset(
            materialized.frame,
            market_context=market_context,
            instrument=instrument,
            timeframe=timeframe,
            audit_tables=audit_tables,
            outpath=None if args.numeric_only else outpath,
            title=f"Cross-Asset Validation — {instrument} {timeframe}",
        )

        print(f"\n=== CROSS-ASSET SUMMARY: {instrument} {timeframe} ===")
        _print_summary(result["summary"])
        if result["html_path"] is not None:
            print(f"\nWrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
