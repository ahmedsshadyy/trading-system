"""Tests for DAG node-level caching on the live/research pipelines.

Covers Phase A.1 (source-hash invalidation) and Phase A.2 (materialize=True
on Class A nodes + swings).
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.dag_runtime import GraphRunContext, execute_graph
from src.dag_runtime import builtin_graphs as graphs_module
from src.dag_runtime.builtin_graphs import (
    _PIPELINE_MATERIALIZE_NODES,
    build_live_stage_graph,
    build_range_boundaries_graph,
    build_research_stage_graph,
    get_builtin_graph,
)
from src.dag_runtime.fingerprints import (
    compute_multi_source_hash,
    compute_source_hash,
)
from scripts import validate_range_boundaries as vrb


def _sample_ohlcv(rows: int = 300) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    close = pd.Series(range(rows), dtype=float).mul(0.6).add(2000.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": (close - 0.2).to_numpy(),
            "high": (close + 0.5).to_numpy(),
            "low": (close - 0.5).to_numpy(),
            "close": close.to_numpy(),
            "volume": [1000.0 + i for i in range(rows)],
        }
    )


# --- compute_source_hash unit tests ---


def test_source_hash_is_stable():
    def fn():
        return 42

    assert compute_source_hash(fn) == compute_source_hash(fn)


def test_source_hash_changes_with_body():
    def fn_a():
        return 1

    def fn_b():
        return 2

    assert compute_source_hash(fn_a) != compute_source_hash(fn_b)


def test_multi_source_hash_combines():
    def fn1():
        return 1

    def fn2():
        return 2

    h12 = compute_multi_source_hash(fn1, fn2)
    h21 = compute_multi_source_hash(fn2, fn1)
    h11 = compute_multi_source_hash(fn1, fn1)
    # Order-sensitive (we want this — same set of fns in different stages
    # produce different hashes).
    assert h12 != h21
    assert h12 != h11


# --- Phase A.2: materialized nodes hit cache on re-run ---


@pytest.mark.parametrize(
    "graph_builder,builder_kwargs",
    [
        (
            build_live_stage_graph,
            {"instrument": "XAU_USD", "swing_window": 6, "include_vp": False},
        ),
        (
            build_research_stage_graph,
            {
                "instrument": "XAU_USD",
                "swing_window": 6,
                "include_vp": False,
                "include_avwap": False,
            },
        ),
    ],
)
def test_materialized_nodes_hit_cache_on_rerun(tmp_path, graph_builder, builder_kwargs):
    raw = _sample_ohlcv(300)
    graph = graph_builder(**builder_kwargs)

    ctx = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H1",
        inputs={"raw_input": raw},
        cache_root=tmp_path,
    )

    first = execute_graph(graph, context=ctx)
    second = execute_graph(graph, context=ctx)

    # First run: nothing cached.
    for name, result in first.node_results.items():
        if result.manifest.node_kind == "source":
            continue
        assert result.cache_hit is False, f"{name} should not hit cache on first run"

    # Second run: every materialized node should be a hit.
    second_hits = {
        name for name, result in second.node_results.items() if result.cache_hit
    }
    expected = {n for n in _PIPELINE_MATERIALIZE_NODES if n in second.node_results}
    missing = expected - second_hits
    assert not missing, f"materialized nodes that did not hit cache: {missing}"

    # Class B nodes (not in MATERIALIZE_NODES) should still recompute.
    for name, result in second.node_results.items():
        if name in _PIPELINE_MATERIALIZE_NODES or result.manifest.node_kind == "source":
            continue
        assert (
            result.cache_hit is False
        ), f"{name} unexpectedly hit cache (not in MATERIALIZE_NODES)"


def test_cache_hit_preserves_output_frame(tmp_path):
    raw = _sample_ohlcv(300)
    graph = build_live_stage_graph(
        instrument="XAU_USD", swing_window=6, include_vp=False
    )

    ctx = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H1",
        inputs={"raw_input": raw},
        cache_root=tmp_path,
    )

    first = execute_graph(graph, context=ctx)
    second = execute_graph(graph, context=ctx)

    # Output of materialized nodes must be bit-identical when served from cache.
    for name in _PIPELINE_MATERIALIZE_NODES:
        if name not in first.node_results:
            continue
        a = first.node_results[name].primary_frame()
        b = second.node_results[name].primary_frame()
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True),
            b.reset_index(drop=True),
            check_dtype=False,
        )


# --- Phase A.1: source-hash invalidation propagates downstream ---


def test_source_hash_change_invalidates_downstream(tmp_path):
    """When atr's source hash changes, atr and every downstream node
    (ema, adx, ..., swings, ...) must miss cache. Upstream node
    (normalize_candles) must still hit cache.
    """
    raw = _sample_ohlcv(300)
    graph = build_live_stage_graph(
        instrument="XAU_USD", swing_window=6, include_vp=False
    )

    ctx = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H1",
        inputs={"raw_input": raw},
        cache_root=tmp_path,
    )

    # Warm the cache.
    execute_graph(graph, context=ctx)

    # Sanity: second run hits cache for all materialized nodes.
    second = execute_graph(graph, context=ctx)
    for name in _PIPELINE_MATERIALIZE_NODES:
        if name in second.node_results:
            assert second.node_results[name].cache_hit, name

    # Simulate an atr source-code change by perturbing its hash on next
    # graph build only. Other nodes' hashes are preserved.
    real_hash = compute_multi_source_hash

    def perturbed_hash(*fns):
        import inspect

        try:
            srcs = [inspect.getsource(f) for f in fns]
        except Exception:
            srcs = [repr(f) for f in fns]
        if any("def add_atr" in s for s in srcs):
            return "PERTURBED_FOR_TEST"
        return real_hash(*fns)

    with patch.object(
        graphs_module, "compute_multi_source_hash", side_effect=perturbed_hash
    ):
        rebuilt = build_live_stage_graph(
            instrument="XAU_USD", swing_window=6, include_vp=False
        )
        third = execute_graph(rebuilt, context=ctx)

    # Upstream of atr: still hits cache.
    assert third.node_results["normalize_candles"].cache_hit is True

    # atr itself: invalidated.
    assert third.node_results["atr"].cache_hit is False

    # Downstream materialized nodes: invalidated (transitively, via
    # upstream_fingerprints in the fingerprint payload).
    downstream_class_a = ["ema", "adx", "rsi", "macd", "bb_width", "body_ratio"]
    for name in downstream_class_a:
        if name in third.node_results:
            assert (
                third.node_results[name].cache_hit is False
            ), f"{name} should be invalidated when upstream atr changes"


def test_node_config_includes_source_hash():
    """Materialized stages must embed source_hash in their NodeManifest.config
    so the fingerprint changes when underlying logic changes.
    """
    graph = build_live_stage_graph(
        instrument="XAU_USD", swing_window=6, include_vp=False
    )
    for node in graph.nodes:
        if node.node_name in _PIPELINE_MATERIALIZE_NODES:
            assert (
                "source_hash" in node.config
            ), f"{node.node_name} missing source_hash in config"
            assert isinstance(node.config["source_hash"], str)
            assert len(node.config["source_hash"]) > 0


def test_structural_carried_state_nodes_remain_unmaterialized_on_rerun(tmp_path):
    raw = _sample_ohlcv(300)
    graph = build_live_stage_graph(
        instrument="XAU_USD", swing_window=6, include_vp=False
    )

    ctx = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H1",
        inputs={"raw_input": raw},
        cache_root=tmp_path,
    )

    execute_graph(graph, context=ctx)
    warm = execute_graph(graph, context=ctx)
    assert warm.node_results["swings"].cache_hit is True
    assert warm.node_results["trend_state"].cache_hit is False
    assert warm.node_results["bos"].cache_hit is False
    assert warm.node_results["choch"].cache_hit is False


def test_smc_carried_state_nodes_remain_unmaterialized_on_rerun(tmp_path):
    raw = _sample_ohlcv(300)
    graph = build_live_stage_graph(
        instrument="XAU_USD", swing_window=6, include_vp=False
    )

    ctx = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H1",
        inputs={"raw_input": raw},
        cache_root=tmp_path,
    )

    execute_graph(graph, context=ctx)
    warm = execute_graph(graph, context=ctx)
    assert warm.node_results["volume_features"].cache_hit is True
    assert warm.node_results["fvg_stack"].cache_hit is False
    assert warm.node_results["displacement"].cache_hit is False
    assert warm.node_results["order_blocks"].cache_hit is False
    assert warm.node_results["ob_mitigation"].cache_hit is False


def test_research_carried_state_tail_nodes_remain_unmaterialized_on_rerun(tmp_path):
    raw = _sample_ohlcv(300)
    graph = build_research_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        include_avwap=True,
    )

    ctx = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H1",
        inputs={"raw_input": raw},
        cache_root=tmp_path,
    )

    execute_graph(graph, context=ctx)
    warm = execute_graph(graph, context=ctx)
    assert warm.node_results["volume_features"].cache_hit is True
    assert warm.node_results["amd_engine"].cache_hit is False
    assert warm.node_results["anchored_vwap"].cache_hit is False
    assert warm.node_results["regime"].cache_hit is False


def test_ob_validation_graph_hits_cache_on_rerun(tmp_path):
    raw = _sample_ohlcv(300)
    graph = get_builtin_graph("validate_ob", instrument="XAU_USD", timeframe="H4")
    ctx = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={
            "html": False,
            "out_dir": str(tmp_path),
            "plot_start": "2026-01-01 00:00:00+00:00",
        },
        cache_root=tmp_path,
    )

    execute_graph(graph, context=ctx, target="ob_validation_bundle")
    warm = execute_graph(graph, context=ctx, target="ob_validation_bundle")

    assert warm.node_results["ob_context"].cache_hit is True
    assert warm.node_results["ob_summary_bundle"].cache_hit is True
    assert warm.node_results["ob_artifact_bundle"].cache_hit is True
    assert warm.node_results["ob_main_chart"].cache_hit is True
    assert warm.node_results["ob_validation_bundle"].cache_hit is False


def _range_context(raw: pd.DataFrame, cache_root) -> GraphRunContext:
    return GraphRunContext(
        graph_name="validate_range_boundaries",
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={
            "date_from": "2026-01-01",
            "plot_rows": 120,
            "full": False,
            "html": False,
            "write_csv": False,
            "out_dir": "notebooks/foundation",
        },
        cache_root=cache_root,
    )


def test_range_boundaries_materialized_nodes_include_source_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vrb, "_load_canonical_live_context", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        vrb,
        "_build_recovery_ladder",
        lambda: [("mid_a", {}), ("mid_b", {})],
    )

    graph = build_range_boundaries_graph(instrument="XAU_USD", timeframe="H4")

    materialized = {
        node.node_name: node
        for node in graph.nodes
        if node.cache_policy.materialize and node.node_kind != "source"
    }
    assert materialized
    for name, node in materialized.items():
        assert "source_hash" in node.config, f"{name} missing source_hash"
        assert isinstance(node.config["source_hash"], str)
        assert node.config["source_hash"]


def test_range_boundaries_materialized_geometry_closure_hits_cache_on_rerun(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vrb, "_load_canonical_live_context", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        vrb,
        "_build_recovery_ladder",
        lambda: [("mid_a", {}), ("mid_b", {})],
    )

    raw = _sample_ohlcv(120)
    graph = build_range_boundaries_graph(instrument="XAU_USD", timeframe="H4")
    ctx = _range_context(raw, tmp_path)

    first = execute_graph(graph, context=ctx, target="range_geometry_audit")
    second = execute_graph(graph, context=ctx, target="range_geometry_audit")

    for name, result in first.node_results.items():
        if result.manifest.node_kind == "source":
            continue
        assert result.cache_hit is False, f"{name} should compute on first run"

    assert set(second.executed_nodes).issubset({"raw_input"})
    for name, result in second.node_results.items():
        if result.manifest.node_kind == "source":
            continue
        assert result.cache_hit is True, f"{name} should hit cache on warm rerun"


def test_range_boundaries_source_hash_change_invalidates_downstream_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vrb, "_load_canonical_live_context", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        vrb,
        "_build_recovery_ladder",
        lambda: [("mid_a", {}), ("mid_b", {})],
    )

    raw = _sample_ohlcv(120)
    graph = build_range_boundaries_graph(instrument="XAU_USD", timeframe="H4")
    ctx = _range_context(raw, tmp_path)

    execute_graph(graph, context=ctx, target="range_geometry_audit")
    warm = execute_graph(graph, context=ctx, target="range_geometry_audit")
    for name, result in warm.node_results.items():
        if result.manifest.node_kind != "source":
            assert result.cache_hit is True, name

    real_hash = compute_multi_source_hash

    def perturbed_hash(*fns):
        if any(getattr(fn, "__name__", "") == "_run_debug_with_params" for fn in fns):
            return "PERTURBED_RANGE_DEBUG_HASH"
        return real_hash(*fns)

    with patch.object(
        graphs_module, "compute_multi_source_hash", side_effect=perturbed_hash
    ):
        rebuilt = build_range_boundaries_graph(instrument="XAU_USD", timeframe="H4")
        third = execute_graph(rebuilt, context=ctx, target="range_geometry_audit")

    assert third.node_results["range_context"].cache_hit is True
    assert third.node_results["range_rung_debug__step8e_a__mid_a"].cache_hit is False
    assert third.node_results["range_rung_debug__step8e_a__mid_b"].cache_hit is False
    assert third.node_results["range_selected_rung"].cache_hit is False
    assert third.node_results["range_selected_debug"].cache_hit is False
    assert third.node_results["range_geometry_audit"].cache_hit is False
