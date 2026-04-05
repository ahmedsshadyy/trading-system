from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.dag_runtime import GraphRunContext, execute_graph, explain_graph_run
from src.dag_runtime.builtin_graphs import get_builtin_graph
from src.dag_runtime.cache_store import invalidate_node_cache


def _default_raw_path(graph: str, symbol: str, timeframe: str) -> Path:
    return Path(f"data/raw/{symbol}_{timeframe}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or explain built-in DAG graphs.")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--symbol", default="XAU_USD")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--raw-path", default=None)
    parser.add_argument("--cache-root", default="data/dag_cache")
    parser.add_argument("--features-root", default="data/features")
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--swing-window", type=int, default=6)
    parser.add_argument("--include-vp", action="store_true")
    parser.add_argument("--include-avwap", action="store_true")
    parser.add_argument("--plot-rows", type=int, default=300)
    parser.add_argument("--date-from", default="2026-01-01")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--invalidate-cache", action="store_true")
    parser.add_argument("--invalidate-node", action="append", default=[])
    args = parser.parse_args()

    raw_path = (
        Path(args.raw_path)
        if args.raw_path
        else _default_raw_path(args.graph, args.symbol, args.timeframe)
    )
    raw_df = pd.read_parquet(raw_path)
    graph = get_builtin_graph(
        args.graph,
        instrument=args.symbol,
        timeframe=args.timeframe,
        swing_window=args.swing_window,
        include_vp=args.include_vp,
        include_avwap=args.include_avwap,
    )
    context = GraphRunContext(
        graph_name=graph.graph_name,
        symbol=args.symbol,
        timeframe=args.timeframe,
        inputs={"raw_input": raw_df},
        config={
            "raw_path": str(raw_path),
            "plot_rows": args.plot_rows,
            "date_from": args.date_from,
            "include_vp": args.include_vp,
            "include_avwap": args.include_avwap,
            "out_dir": (
                "notebooks/foundation"
                if "range" in args.graph or "regime" in args.graph or "sr" in args.graph
                else "notebooks/structure"
            ),
        },
        cache_root=args.cache_root,
        state_root=args.state_root,
        features_root=args.features_root,
        force=args.force,
        invalidate_cache=args.invalidate_cache,
    )
    if args.invalidate_node:
        for node_name in args.invalidate_node:
            removed = invalidate_node_cache(
                args.cache_root,
                graph_name=graph.graph_name,
                symbol=args.symbol,
                timeframe=args.timeframe,
                node_name=node_name,
            )
            print(f"invalidated {node_name}: {len(removed)} files")

    if args.explain:
        print(
            json.dumps(
                explain_graph_run(graph, target=args.target, context=context),
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = execute_graph(
        graph,
        target=args.target,
        context=context,
        invalidate_nodes=set(args.invalidate_node),
    )
    profiler_path = (
        Path(args.cache_root)
        / graph.graph_name
        / args.symbol
        / args.timeframe
        / "run-summary.json"
    )
    result.profiler.write_json(profiler_path)
    primary = result.primary_frame()
    print(
        json.dumps(
            {
                "graph_name": graph.graph_name,
                "target": result.target,
                "rows": int(len(primary)) if primary is not None else None,
                "profiler_path": str(profiler_path),
                "node_count": len(result.node_results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
