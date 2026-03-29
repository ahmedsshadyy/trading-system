from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.pipelines import (
    materialize_live_features,
    materialize_research_features,
    run_live_pipeline,
    run_research_pipeline,
)
from src.pipeline_runtime import PipelineRunProfiler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline benchmark for live/research pipeline execution."
    )
    parser.add_argument("--pipeline", choices=["live", "research"], default="live")
    parser.add_argument("--instrument", default="XAU_USD")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load the full parquet instead of tail rows.",
    )
    parser.add_argument("--tail-rows", type=int, default=1500)
    parser.add_argument("--include-vp", action="store_true")
    parser.add_argument("--include-avwap", action="store_true")
    parser.add_argument(
        "--persist", action="store_true", help="Persist canonical feature partitions."
    )
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()

    data_path = ROOT / "data" / "raw" / f"{args.instrument}_{args.timeframe}.parquet"
    raw = pd.read_parquet(data_path)
    if not args.full:
        raw = raw.tail(args.tail_rows).copy()

    profiler = PipelineRunProfiler(
        pipeline=f"benchmark_{args.pipeline}",
        symbol=args.instrument,
        timeframe=args.timeframe,
    )
    started_at = time.perf_counter()
    profiler.record_stage(
        "load_raw", started_at=started_at, input_frame=raw, output_frame=raw
    )

    if args.pipeline == "live":
        if args.persist:
            result = materialize_live_features(
                raw,
                instrument=args.instrument,
                timeframe=args.timeframe,
                include_vp=args.include_vp,
                features_root=args.features_root,
                profiler=profiler,
            )
        else:
            result = run_live_pipeline(
                raw,
                instrument=args.instrument,
                timeframe=args.timeframe,
                include_vp=args.include_vp,
                profiler=profiler,
            )
    else:
        if args.persist:
            result = materialize_research_features(
                raw,
                instrument=args.instrument,
                timeframe=args.timeframe,
                include_vp=args.include_vp,
                include_avwap=args.include_avwap,
                features_root=args.features_root,
                profiler=profiler,
            )
        else:
            result = run_research_pipeline(
                raw,
                instrument=args.instrument,
                timeframe=args.timeframe,
                include_vp=args.include_vp,
                include_avwap=args.include_avwap,
                profiler=profiler,
            )

    summary = result.profiler.summary()
    if args.write_summary:
        out_path = (
            ROOT
            / "artifacts"
            / "benchmarks"
            / f"{args.pipeline}_{args.instrument}_{args.timeframe}.json"
        )
        result.profiler.write_json(out_path)
        print(f"wrote summary -> {out_path}")
    print(summary)


if __name__ == "__main__":
    main()
