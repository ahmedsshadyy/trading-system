from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.features.cross_asset import (
    GLOBAL_CONTEXT_SYMBOL,
)
from src.indicators.pipelines.build_research import materialize_research_features
from src.indicators.research import (
    build_cross_asset_correlation_audit,
    summarize_cross_asset_correlation_audit,
)
from src.indicators.research.smt_research import build_smt_research_table
from src.pipeline_runtime import (
    load_partitioned_dataset,
    write_json_atomic,
    write_parquet_atomic,
)
from src.validation.indicators.smt import summarize_smt, validate_smt

OUT_DIR = Path("notebooks/cross_asset")
FEATURES_ROOT = Path("data/validation_cache/features")
PLOT_ROWS = 300
INPUT_ROWS = 2500
RUNS = (
    ("XAU_USD", "H1"),
    ("XAU_USD", "H4"),
    ("USOIL", "H1"),
    ("USOIL", "H4"),
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
    parser = argparse.ArgumentParser(description="Validate SMT cross-asset output.")
    parser.add_argument(
        "--instrument",
        choices=("all", "XAU_USD", "USOIL"),
        default="all",
        help="Primary instrument to validate.",
    )
    parser.add_argument(
        "--timeframe",
        choices=("all", "H1", "H4"),
        default="all",
        help="Timeframe to validate.",
    )
    parser.add_argument(
        "--plot-rows",
        type=int,
        default=PLOT_ROWS,
        help="Rows to keep in the HTML chart window.",
    )
    parser.add_argument(
        "--input-rows",
        type=int,
        default=INPUT_ROWS,
        help=(
            "Recent raw bars to load before running the pipeline. "
            "Use 0 to keep the full file."
        ),
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=5,
        help="Number of sample bullish/bearish SMT windows to print.",
    )
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Run numeric validation only and skip HTML chart generation.",
    )
    parser.add_argument(
        "--features-root",
        default=str(FEATURES_ROOT),
        help="Feature/state root used for incremental validation materialization.",
    )
    parser.add_argument(
        "--force-graph-recompute",
        type=_parse_bool,
        default=False,
        help=(
            "When true, bypass DAG cache and force a fresh graph recompute. "
            "When false, reuse DAG/materialization cache when possible."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes. 0 (default) auto-selects "
            "min(cpu_count, n_runs). 1 disables multiprocessing — useful "
            "for debugging."
        ),
    )
    parser.add_argument(
        "--rebuild-audit",
        action="store_true",
        help=(
            "Rebuild the expensive cross-asset audit if cached summary/tables "
            "are missing or stale. Default is fast-path reuse/skip."
        ),
    )
    parser.add_argument(
        "--force-rebuild-audit",
        action="store_true",
        help=(
            "Always rebuild the cross-asset audit and overwrite cached "
            "summary/tables for this run."
        ),
    )
    return parser.parse_args()


def _stage_seconds(started_at: float) -> float:
    return time.perf_counter() - started_at


def _load_cached_audit_summary(
    *,
    audit_dir: Path,
    numeric_only: bool,
    force_rebuild_audit: bool = False,
) -> tuple[dict[str, object] | None, dict[str, pd.DataFrame] | None]:
    if force_rebuild_audit or not audit_dir.exists():
        return None, None

    summary_path = audit_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8")), None

    if numeric_only:
        return None, None

    audit_tables: dict[str, pd.DataFrame] = {}
    for parquet_file in audit_dir.glob("*.parquet"):
        audit_tables[parquet_file.stem] = pd.read_parquet(parquet_file)
    if not audit_tables:
        return None, None
    return summarize_cross_asset_correlation_audit(audit_tables), audit_tables


def _persist_rebuilt_audit(
    *,
    audit_dir: Path,
    audit_tables: dict[str, pd.DataFrame],
    audit_summary: dict[str, object],
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    for table_name, table in audit_tables.items():
        if table.empty:
            continue
        write_parquet_atomic(table, audit_dir / f"{table_name}.parquet")
    write_json_atomic(audit_summary, audit_dir / "summary.json")


def _selected_runs(args: argparse.Namespace) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for instrument, timeframe in RUNS:
        if args.instrument != "all" and instrument != args.instrument:
            continue
        if args.timeframe != "all" and timeframe != args.timeframe:
            continue
        selected.append((instrument, timeframe))
    return selected


# Worker config is a plain dict (picklable) so multiprocessing can pass it
# across process boundaries without dragging in argparse.Namespace.
WorkerConfig = dict


def _run_one(config: WorkerConfig) -> dict:
    """Run one (instrument, timeframe) validation in isolation.

    Designed to be called either inline (sequential mode) or in a
    subprocess via multiprocessing.Pool. Returns a dict with everything
    needed for the main process to print results — no I/O ordering
    relies on this function.
    """
    instrument: str = config["instrument"]
    timeframe: str = config["timeframe"]
    features_root = Path(config["features_root"])
    timings: dict[str, float] = {}

    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    raw = pd.read_parquet(Path(f"data/raw/{instrument}_{timeframe}.parquet"))
    if config["input_rows"] > 0:
        raw = raw.tail(config["input_rows"]).reset_index(drop=True)

    started_at = time.perf_counter()
    materialized = materialize_research_features(
        raw,
        instrument=instrument,
        timeframe=timeframe,
        include_vp=False,
        include_avwap=False,
        include_cross_asset=True,
        features_root=str(features_root),
        raw_data_root="data/raw",
        force_graph_recompute=config["force_graph_recompute"],
        build_cross_asset_audit=False,
    )
    timings["materialize_research_features_seconds"] = _stage_seconds(started_at)
    full_df = materialized.frame.copy()
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)
    market_context = load_partitioned_dataset(
        features_root,
        dataset="market_context_research",
        symbol=GLOBAL_CONTEXT_SYMBOL,
        timeframe=timeframe,
    )
    started_at = time.perf_counter()
    research_table = build_smt_research_table(full_df)
    timings["build_smt_research_table_seconds"] = _stage_seconds(started_at)

    audit_dir = features_root / "research_cross_asset_audit" / instrument / timeframe
    started_at = time.perf_counter()
    cached_audit_summary, cached_audit = _load_cached_audit_summary(
        audit_dir=audit_dir,
        numeric_only=config["numeric_only"],
        force_rebuild_audit=config["force_rebuild_audit"],
    )
    timings["cached_audit_load_seconds"] = _stage_seconds(started_at)

    audit_cache_status = "hit-summary" if cached_audit_summary is not None else "miss"
    if cached_audit is not None:
        audit_cache_status = "hit-parquet"

    if cached_audit_summary is None and (
        config["rebuild_audit"] or config["force_rebuild_audit"]
    ):
        started_at = time.perf_counter()
        cached_audit = build_cross_asset_correlation_audit(
            full_df,
            market_context,
            instrument=instrument,
            timeframe=timeframe,
        )
        cached_audit_summary = summarize_cross_asset_correlation_audit(cached_audit)
        _persist_rebuilt_audit(
            audit_dir=audit_dir,
            audit_tables=cached_audit,
            audit_summary=cached_audit_summary,
        )
        timings["rebuild_audit_seconds"] = _stage_seconds(started_at)
        audit_cache_status = (
            "force-rebuilt" if config["force_rebuild_audit"] else "rebuilt"
        )

    plot_df = full_df.tail(config["plot_rows"]).copy()
    title = f"SMT Validation — {instrument} {timeframe}"
    outpath = Path(config["out_dir"]) / f"smt_validation_{instrument}_{timeframe}.html"

    if config["numeric_only"]:
        started_at = time.perf_counter()
        summary = summarize_smt(
            full_df,
            market_context=market_context,
            research_table=research_table,
            correlation_audit=cached_audit,
            correlation_audit_summary=cached_audit_summary,
            instrument=instrument,
            timeframe=timeframe,
        )
        timings["summarize_smt_seconds"] = _stage_seconds(started_at)
        return {
            "instrument": instrument,
            "timeframe": timeframe,
            "summary": summary,
            "bull_windows": [],
            "bear_windows": [],
            "html_path": None,
            "timings": timings,
            "audit_cache_status": audit_cache_status,
        }

    started_at = time.perf_counter()
    result = validate_smt(
        plot_df,
        full_df=full_df,
        market_context=market_context,
        research_table=research_table,
        correlation_audit=cached_audit,
        correlation_audit_summary=cached_audit_summary,
        outpath=outpath,
        title=title,
        n_windows=config["n_windows"],
        instrument=instrument,
        timeframe=timeframe,
    )
    timings["validate_smt_seconds"] = _stage_seconds(started_at)
    return {
        "instrument": instrument,
        "timeframe": timeframe,
        "timings": timings,
        "audit_cache_status": audit_cache_status,
        **result,
    }


def _print_result(result: dict) -> None:
    instrument = result["instrument"]
    timeframe = result["timeframe"]
    print(f"\n=== SMT SUMMARY: {instrument} {timeframe} ===")
    _print_summary(result["summary"])

    if result.get("bull_windows"):
        print("\n=== SAMPLE BULL SMT WINDOWS ===")
        for idx, window in enumerate(result["bull_windows"], start=1):
            print(f"\n--- bull window {idx} ---")
            print(window.to_string(index=True))

    if result.get("bear_windows"):
        print("\n=== SAMPLE BEAR SMT WINDOWS ===")
        for idx, window in enumerate(result["bear_windows"], start=1):
            print(f"\n--- bear window {idx} ---")
            print(window.to_string(index=True))

    if result.get("html_path") is not None:
        print(f"\nWrote chart to: {result['html_path']}")
    if result.get("audit_cache_status") is not None:
        print(f"\nAudit cache: {result['audit_cache_status']}")
    if result.get("timings"):
        print("\n=== STAGE TIMINGS ===")
        for key, value in result["timings"].items():
            print(f"{key}: {value:.3f}s")


def main() -> None:
    args = _parse_args()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features_root = Path(args.features_root)
    features_root.mkdir(parents=True, exist_ok=True)

    selected = _selected_runs(args)
    if not selected:
        print("No runs selected.")
        return

    base_config: WorkerConfig = {
        "features_root": str(features_root),
        "input_rows": args.input_rows,
        "plot_rows": args.plot_rows,
        "n_windows": args.n_windows,
        "numeric_only": args.numeric_only,
        "force_graph_recompute": args.force_graph_recompute,
        "rebuild_audit": args.rebuild_audit,
        "force_rebuild_audit": args.force_rebuild_audit,
        "out_dir": str(OUT_DIR),
    }
    configs = [
        {**base_config, "instrument": inst, "timeframe": tf} for inst, tf in selected
    ]

    if args.workers == 0:
        n_workers = min(os.cpu_count() or 1, len(configs))
    else:
        n_workers = max(1, args.workers)

    if n_workers <= 1 or len(configs) == 1:
        # Sequential mode — useful for debugging.
        for config in configs:
            _print_result(_run_one(config))
        return

    # Parallel mode. Use the "spawn" start method on macOS to avoid
    # accidentally inheriting Pandas / numpy state across forks (some
    # BLAS libraries deadlock when forking).
    print(
        f"Running {len(configs)} validations across {n_workers} worker processes...",
        flush=True,
    )
    ctx = get_context("spawn")
    results_by_pair: dict[tuple[str, str], dict] = {}
    with ctx.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_run_one, configs):
            key = (result["instrument"], result["timeframe"])
            results_by_pair[key] = result
            print(
                f"  finished {result['instrument']} {result['timeframe']}",
                flush=True,
            )

    # Print in the deterministic order the user supplied.
    for instrument, timeframe in selected:
        result = results_by_pair.get((instrument, timeframe))
        if result is None:
            print(f"\n!!! Missing result for {instrument} {timeframe}")
            continue
        _print_result(result)


if __name__ == "__main__":
    main()
