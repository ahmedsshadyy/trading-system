from __future__ import annotations

import pandas as pd

from src.dag_runtime import (
    CachePolicy,
    GraphManifest,
    GraphRunContext,
    NodeManifest,
    NodeOutput,
    execute_graph,
    explain_graph_run,
)
from src.dag_runtime.builtin_graphs import build_live_stage_graph
from src.indicators.pipelines import build_live as live_pipeline


def _sample_ohlcv(rows: int = 120) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="4h", tz="UTC")
    close = pd.Series(range(rows), dtype=float).mul(0.6).add(2000.0)
    open_ = close.shift(1).fillna(close.iloc[0] - 0.2)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.3
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.3
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close.to_numpy(),
            "volume": [1000 + i for i in range(rows)],
        }
    )


def test_executor_reuses_cached_node_outputs(tmp_path):
    graph = GraphManifest(
        graph_name="unit_graph",
        nodes=(
            NodeManifest(
                graph_name="unit_graph",
                node_name="source",
                node_kind="source",
                semantic_class="A",
                inputs=("raw",),
                upstream_nodes=(),
                output_artifacts=("frame",),
                cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
                compute_fn=lambda context, deps: NodeOutput(
                    frames={"frame": context.inputs["raw"].copy()}
                ),
            ),
            NodeManifest(
                graph_name="unit_graph",
                node_name="doubled",
                node_kind="compute",
                semantic_class="A",
                inputs=(),
                upstream_nodes=("source",),
                output_artifacts=("frame",),
                compute_fn=lambda context, deps: NodeOutput(
                    frames={
                        "frame": deps["source"]
                        .primary_frame()
                        .assign(close=lambda df: df["close"] * 2)
                    }
                ),
            ),
        ),
        default_target="doubled",
    )
    context = GraphRunContext(
        graph_name="unit_graph",
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw": _sample_ohlcv(24)},
        cache_root=tmp_path,
    )
    first = execute_graph(graph, context=context)
    second = execute_graph(graph, context=context)
    assert first.node_results["doubled"].cache_hit is False
    assert second.node_results["doubled"].cache_hit is True


def test_explain_graph_reports_node_execution_status(tmp_path):
    graph = GraphManifest(
        graph_name="explain_graph",
        nodes=(
            NodeManifest(
                graph_name="explain_graph",
                node_name="source",
                node_kind="source",
                semantic_class="A",
                inputs=("raw",),
                upstream_nodes=(),
                output_artifacts=("frame",),
                cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
                compute_fn=lambda context, deps: NodeOutput(
                    frames={"frame": context.inputs["raw"].copy()}
                ),
            ),
        ),
        default_target="source",
    )
    context = GraphRunContext(
        graph_name="explain_graph",
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw": _sample_ohlcv(10)},
        cache_root=tmp_path,
    )
    explanation = explain_graph_run(graph, context=context)
    assert explanation["graph_name"] == "explain_graph"
    assert explanation["nodes"][0]["node_name"] == "source"
    assert explanation["nodes"][0]["would_execute"] is True


def test_live_stage_graph_matches_manual_stage_chain():
    raw = _sample_ohlcv(180)
    graph = build_live_stage_graph(
        instrument="XAU_USD", swing_window=6, include_vp=False
    )
    result = execute_graph(
        graph,
        context=GraphRunContext(
            graph_name=graph.graph_name,
            symbol="XAU_USD",
            timeframe="H4",
            inputs={"raw_input": raw},
            force=True,
            invalidate_cache=True,
        ),
    )
    manual = raw.copy()
    for stage in live_pipeline._live_stages(
        instrument="XAU_USD", swing_window=6, include_vp=False
    ):
        manual = stage.fn(manual)
    pd.testing.assert_frame_equal(
        result.primary_frame().reset_index(drop=True),
        manual.reset_index(drop=True),
        check_dtype=False,
    )
