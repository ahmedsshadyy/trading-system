from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.dag_runtime import GraphRunContext, execute_graph
from src.dag_runtime import builtin_graphs as graphs_module
from src.dag_runtime.builtin_graphs import (
    build_live_stage_graph,
    build_research_stage_graph,
)
from src.indicators.features.cross_asset import GLOBAL_CONTEXT_SYMBOL
from src.indicators.features.cross_asset import (
    SMT_PARTNERS,
    attach_cross_asset_context,
    load_raw_context_frames,
    market_context_cache_is_current,
    persist_market_context,
    read_market_context_summary,
    relevant_correlation_pairs,
    resolve_cross_asset_inputs,
)
from src.indicators.pipelines.build_research import build_smt_partner_indicators
from src.indicators.pipelines.build_live import (
    build_live_indicators,
    materialize_live_features,
)
from src.indicators.pipelines.build_research import (
    build_research_indicators,
    materialize_research_features,
)
from src.pipeline_runtime import load_partitioned_dataset


def _raw_frame(
    timestamps: pd.DatetimeIndex,
    close: np.ndarray,
) -> pd.DataFrame:
    close_series = pd.Series(close, dtype=float)
    open_ = close_series.shift(1).fillna(close_series.iloc[0] - 0.1)
    high = pd.concat([open_, close_series], axis=1).max(axis=1) + 0.2
    low = pd.concat([open_, close_series], axis=1).min(axis=1) - 0.2
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close_series.to_numpy(),
            "volume": np.arange(len(timestamps), dtype=float) + 1000.0,
        }
    )


def _cross_asset_universe(rows: int = 60) -> dict[str, pd.DataFrame]:
    ts = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    base = np.sin(np.linspace(0.0, 6.0, rows))
    return {
        "XAU_USD": _raw_frame(ts, 2000.0 + np.cumsum(base + 0.4)),
        "DXY": _raw_frame(ts, 100.0 + np.cumsum(-0.3 * base + 0.05)),
        "USD_JPY": _raw_frame(ts, 145.0 + np.cumsum(0.2 * base + 0.03)),
        "USOIL": _raw_frame(ts, 70.0 + np.cumsum(0.5 * base + 0.06)),
        "USD_CAD": _raw_frame(ts, 1.30 + np.cumsum(-0.02 * base + 0.002)),
        "EUR_USD": _raw_frame(ts, 1.10 + np.cumsum(0.01 * base + 0.001)),
    }


def _live_graph_context(
    *,
    raw: pd.DataFrame,
    peers: dict[str, pd.DataFrame],
    cache_root: Path,
    features_root: Path,
) -> GraphRunContext:
    return GraphRunContext(
        graph_name="live_pipeline",
        symbol="XAU_USD",
        timeframe="H1",
        inputs={"raw_input": raw, "peer_raw_frames": peers},
        config={
            "instrument": "XAU_USD",
            "timeframe": "H1",
            "swing_window": 6,
            "include_vp": False,
            "include_cross_asset": True,
            "raw_data_root": None,
        },
        cache_root=cache_root,
        features_root=features_root,
    )


def _research_graph_context(
    *,
    raw: pd.DataFrame,
    peers: dict[str, pd.DataFrame],
    cache_root: Path,
    features_root: Path,
) -> GraphRunContext:
    return GraphRunContext(
        graph_name="research_pipeline",
        symbol="XAU_USD",
        timeframe="H1",
        inputs={"raw_input": raw, "peer_raw_frames": peers},
        config={
            "instrument": "XAU_USD",
            "timeframe": "H1",
            "swing_window": 6,
            "include_vp": False,
            "include_avwap": False,
            "include_cross_asset": True,
            "raw_data_root": None,
        },
        cache_root=cache_root,
        features_root=features_root,
    )


def test_live_stage_graph_includes_cross_asset_nodes_when_enabled() -> None:
    graph = build_live_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        timeframe="H1",
        include_cross_asset=True,
    )

    node_names = {node.node_name for node in graph.nodes}
    expected_partners = {
        f"live_partner_{symbol}" for symbol, _relation in SMT_PARTNERS["XAU_USD"]
    }

    assert graph.default_target == "live_feature_bundle"
    assert "live_peer_context_source" in node_names
    assert "live_market_context_source" in node_names
    assert "live_cross_asset_attach" in node_names
    assert "live_feature_bundle" in node_names
    assert expected_partners.issubset(node_names)


def test_live_stage_graph_omits_cross_asset_nodes_when_disabled() -> None:
    graph = build_live_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        timeframe="H1",
        include_cross_asset=False,
    )

    node_names = {node.node_name for node in graph.nodes}
    assert "live_peer_context_source" not in node_names
    assert "live_market_context_source" not in node_names
    assert "live_cross_asset_attach" not in node_names
    assert "live_feature_bundle" not in node_names
    assert not any(name.startswith("live_partner_") for name in node_names)


def test_live_and_research_cross_asset_columns_match() -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}

    live = build_live_indicators(
        primary,
        instrument="XAU_USD",
        include_vp=False,
        timeframe="H1",
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
    )
    research = build_research_indicators(
        primary,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
        timeframe="H1",
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
    )

    shared_xasset_columns = sorted(
        column
        for column in live.columns
        if column.startswith("xasset_") and column in research.columns
    )
    assert shared_xasset_columns
    pd.testing.assert_frame_equal(
        live[shared_xasset_columns].reset_index(drop=True),
        research[shared_xasset_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_research_stage_graph_includes_cross_asset_nodes_when_enabled() -> None:
    graph = build_research_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        include_avwap=False,
        timeframe="H1",
        include_cross_asset=True,
    )

    node_names = {node.node_name for node in graph.nodes}
    expected_partners = {
        f"research_partner_{symbol}" for symbol, _relation in SMT_PARTNERS["XAU_USD"]
    }

    assert graph.default_target == "research_feature_bundle"
    assert "research_peer_context_source" in node_names
    assert "research_market_context_source" in node_names
    assert "research_cross_asset_attach" in node_names
    assert "research_smt_research_table" in node_names
    assert "research_cross_asset_audit" in node_names
    assert "research_feature_bundle" in node_names
    assert "research_full_bundle" in node_names
    assert expected_partners.issubset(node_names)


def test_research_stage_graph_omits_cross_asset_nodes_when_disabled() -> None:
    graph = build_research_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        include_avwap=False,
        timeframe="H1",
        include_cross_asset=False,
    )

    node_names = {node.node_name for node in graph.nodes}
    assert "research_peer_context_source" not in node_names
    assert "research_market_context_source" not in node_names
    assert "research_cross_asset_attach" not in node_names
    assert "research_smt_research_table" not in node_names
    assert "research_cross_asset_audit" not in node_names
    assert "research_feature_bundle" not in node_names
    assert "research_full_bundle" not in node_names
    assert not any(name.startswith("research_partner_") for name in node_names)


def test_research_cross_asset_graph_matches_legacy_attach_path() -> None:
    universe = _cross_asset_universe(rows=96)
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}

    graph_research = build_research_indicators(
        primary,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
        timeframe="H1",
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
    )
    primary_only = build_research_indicators(
        primary,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
        timeframe="H1",
        include_cross_asset=False,
        raw_data_root=None,
    )
    legacy_market_context, legacy_processed = resolve_cross_asset_inputs(
        primary_only,
        instrument="XAU_USD",
        timeframe="H1",
        market_context=None,
        processed_frames=None,
        peer_raw_frames=peers,
        raw_data_root=None,
        partner_builder=lambda frame, _symbol: build_smt_partner_indicators(
            frame, swing_window=6
        ),
        full_pair_matrix=True,
    )
    legacy_research = attach_cross_asset_context(
        primary_only.copy(),
        instrument="XAU_USD",
        timeframe="H1",
        market_context=legacy_market_context,
        processed_frames=legacy_processed,
    )

    shared_columns = sorted(
        set(graph_research.columns)
        & set(legacy_research.columns)
        & {"timestamp", *[c for c in graph_research.columns if c.startswith("xasset_")]}
    )
    assert shared_columns
    pd.testing.assert_frame_equal(
        graph_research[shared_columns].reset_index(drop=True),
        legacy_research[shared_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_research_cross_asset_graph_hits_market_context_partner_and_smt_cache_on_rerun(
    tmp_path: Path,
) -> None:
    universe = _cross_asset_universe(rows=96)
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}
    graph = build_research_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        include_avwap=False,
        timeframe="H1",
        include_cross_asset=True,
    )
    context = _research_graph_context(
        raw=primary,
        peers=peers,
        cache_root=tmp_path / "dag_cache",
        features_root=tmp_path / "features",
    )

    execute_graph(graph, context=context, target="research_feature_bundle")
    second = execute_graph(graph, context=context, target="research_feature_bundle")

    assert second.node_results["research_market_context_source"].cache_hit is True
    assert second.node_results["research_smt_research_table"].cache_hit is True
    for symbol, _relation in SMT_PARTNERS["XAU_USD"]:
        assert second.node_results[f"research_partner_{symbol}"].cache_hit is True
    assert second.node_results["research_cross_asset_attach"].cache_hit is False
    assert second.node_results["research_feature_bundle"].cache_hit is False


def test_research_cross_asset_partner_source_change_invalidates_only_partner_closure(
    tmp_path: Path,
) -> None:
    universe = _cross_asset_universe(rows=96)
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}
    graph = build_research_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        include_avwap=False,
        timeframe="H1",
        include_cross_asset=True,
    )
    context = _research_graph_context(
        raw=primary,
        peers=peers,
        cache_root=tmp_path / "dag_cache",
        features_root=tmp_path / "features",
    )

    execute_graph(graph, context=context, target="research_full_bundle")
    warm = execute_graph(graph, context=context, target="research_full_bundle")
    assert warm.node_results["research_market_context_source"].cache_hit is True
    assert warm.node_results["research_partner_DXY"].cache_hit is True
    assert warm.node_results["research_partner_USD_JPY"].cache_hit is True
    assert warm.node_results["research_smt_research_table"].cache_hit is True

    real_partner_hash = graphs_module._research_partner_source_hash

    def perturbed_partner_hash(symbol: str) -> str:
        if symbol == "DXY":
            return "PERTURBED_RESEARCH_DXY_PARTNER_HASH"
        return real_partner_hash(symbol)

    with patch.object(
        graphs_module,
        "_research_partner_source_hash",
        side_effect=perturbed_partner_hash,
    ):
        rebuilt = build_research_stage_graph(
            instrument="XAU_USD",
            swing_window=6,
            include_vp=False,
            include_avwap=False,
            timeframe="H1",
            include_cross_asset=True,
        )
        third = execute_graph(rebuilt, context=context, target="research_full_bundle")

    assert third.node_results["research_market_context_source"].cache_hit is True
    assert third.node_results["research_partner_DXY"].cache_hit is False
    assert third.node_results["research_partner_USD_JPY"].cache_hit is True
    assert third.node_results["research_cross_asset_attach"].cache_hit is False
    assert third.node_results["research_smt_research_table"].cache_hit is False
    assert third.node_results["research_cross_asset_audit"].cache_hit is False
    assert third.node_results["research_full_bundle"].cache_hit is False


def test_research_feature_bundle_excludes_audit_node(tmp_path: Path) -> None:
    graph = build_research_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        include_avwap=False,
        timeframe="H1",
        include_cross_asset=True,
    )
    universe = _cross_asset_universe(rows=96)
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}
    context = _research_graph_context(
        raw=primary,
        peers=peers,
        cache_root=tmp_path / "dag_cache",
        features_root=tmp_path / "features",
    )
    feature_result = execute_graph(
        graph, context=context, target="research_feature_bundle"
    )
    full_result = execute_graph(graph, context=context, target="research_full_bundle")

    assert "research_cross_asset_audit" not in feature_result.node_results
    assert "research_cross_asset_audit" in full_result.node_results


def test_live_cross_asset_graph_matches_legacy_attach_path() -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}

    graph_live = build_live_indicators(
        primary,
        instrument="XAU_USD",
        include_vp=False,
        timeframe="H1",
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
    )
    primary_only = build_live_indicators(
        primary,
        instrument="XAU_USD",
        include_vp=False,
        timeframe="H1",
        include_cross_asset=False,
        raw_data_root=None,
    )
    legacy_market_context, legacy_processed = resolve_cross_asset_inputs(
        primary_only,
        instrument="XAU_USD",
        timeframe="H1",
        market_context=None,
        processed_frames=None,
        peer_raw_frames=peers,
        raw_data_root=None,
        partner_builder=lambda frame, _symbol: build_smt_partner_indicators(
            frame, swing_window=6
        ),
    )
    legacy_live = attach_cross_asset_context(
        primary_only.copy(),
        instrument="XAU_USD",
        timeframe="H1",
        market_context=legacy_market_context,
        processed_frames=legacy_processed,
    )

    shared_columns = sorted(
        set(graph_live.columns)
        & set(legacy_live.columns)
        & {"timestamp", *[c for c in graph_live.columns if c.startswith("xasset_")]}
    )
    assert shared_columns
    pd.testing.assert_frame_equal(
        graph_live[shared_columns].reset_index(drop=True),
        legacy_live[shared_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_live_cross_asset_graph_hits_market_context_and_partner_caches_on_rerun(
    tmp_path: Path,
) -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}
    graph = build_live_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        timeframe="H1",
        include_cross_asset=True,
    )
    context = _live_graph_context(
        raw=primary,
        peers=peers,
        cache_root=tmp_path / "dag_cache",
        features_root=tmp_path / "features",
    )

    execute_graph(graph, context=context)
    second = execute_graph(graph, context=context)

    assert second.node_results["live_market_context_source"].cache_hit is True
    for symbol, _relation in SMT_PARTNERS["XAU_USD"]:
        assert second.node_results[f"live_partner_{symbol}"].cache_hit is True
    assert second.node_results["live_cross_asset_attach"].cache_hit is False
    assert second.node_results["live_feature_bundle"].cache_hit is False


def test_live_cross_asset_partner_source_change_invalidates_only_partner_closure(
    tmp_path: Path,
) -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}
    graph = build_live_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        timeframe="H1",
        include_cross_asset=True,
    )
    context = _live_graph_context(
        raw=primary,
        peers=peers,
        cache_root=tmp_path / "dag_cache",
        features_root=tmp_path / "features",
    )

    execute_graph(graph, context=context)
    warm = execute_graph(graph, context=context)
    assert warm.node_results["live_market_context_source"].cache_hit is True
    assert warm.node_results["live_partner_DXY"].cache_hit is True
    assert warm.node_results["live_partner_USD_JPY"].cache_hit is True

    real_partner_hash = graphs_module._live_partner_source_hash

    def perturbed_partner_hash(symbol: str) -> str:
        if symbol == "DXY":
            return "PERTURBED_DXY_PARTNER_HASH"
        return real_partner_hash(symbol)

    with patch.object(
        graphs_module,
        "_live_partner_source_hash",
        side_effect=perturbed_partner_hash,
    ):
        rebuilt = build_live_stage_graph(
            instrument="XAU_USD",
            swing_window=6,
            include_vp=False,
            timeframe="H1",
            include_cross_asset=True,
        )
        third = execute_graph(rebuilt, context=context)

    assert third.node_results["live_market_context_source"].cache_hit is True
    assert third.node_results["live_partner_DXY"].cache_hit is False
    assert third.node_results["live_partner_USD_JPY"].cache_hit is True
    assert third.node_results["live_cross_asset_attach"].cache_hit is False
    assert third.node_results["live_feature_bundle"].cache_hit is False


def test_live_cross_asset_peer_input_change_rebuilds_only_affected_partner_node(
    tmp_path: Path,
) -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}
    graph = build_live_stage_graph(
        instrument="XAU_USD",
        swing_window=6,
        include_vp=False,
        timeframe="H1",
        include_cross_asset=True,
    )
    base_context = _live_graph_context(
        raw=primary,
        peers=peers,
        cache_root=tmp_path / "dag_cache",
        features_root=tmp_path / "features",
    )

    execute_graph(graph, context=base_context)

    changed_peers = dict(peers)
    changed_dxy = changed_peers["DXY"].copy()
    changed_dxy.loc[changed_dxy.index[-1], "close"] += 1.25
    changed_dxy.loc[changed_dxy.index[-1], "high"] += 1.25
    changed_peers["DXY"] = changed_dxy
    changed_context = _live_graph_context(
        raw=primary,
        peers=changed_peers,
        cache_root=tmp_path / "dag_cache",
        features_root=tmp_path / "features",
    )

    second = execute_graph(graph, context=changed_context)

    assert second.node_results["live_partner_DXY"].cache_hit is False
    assert second.node_results["live_partner_USD_JPY"].cache_hit is True
    assert second.node_results["live_cross_asset_attach"].cache_hit is False
    assert second.node_results["live_feature_bundle"].cache_hit is False


def test_live_materialization_persists_market_context_dataset(tmp_path: Path) -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}

    result = materialize_live_features(
        primary,
        instrument="XAU_USD",
        timeframe="H1",
        include_vp=False,
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
        features_root=str(tmp_path),
    )
    market_context = load_partitioned_dataset(
        tmp_path,
        dataset="market_context_live",
        symbol=GLOBAL_CONTEXT_SYMBOL,
        timeframe="H1",
    )

    assert result.metadata is not None
    assert result.metadata.extra["include_cross_asset"] is True
    assert not market_context.empty


def test_research_materialization_emits_cross_asset_audit_summary(
    tmp_path: Path,
) -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}

    materialize_research_features(
        primary,
        instrument="XAU_USD",
        timeframe="H1",
        include_vp=False,
        include_avwap=False,
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
        features_root=str(tmp_path),
    )

    summary_path = (
        tmp_path / "research_cross_asset_audit" / "XAU_USD" / "H1" / "summary.json"
    )
    assert summary_path.exists()


def test_research_materialization_can_skip_cross_asset_audit(
    tmp_path: Path,
) -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}

    materialize_research_features(
        primary,
        instrument="XAU_USD",
        timeframe="H1",
        include_vp=False,
        include_avwap=False,
        include_cross_asset=True,
        features_root=str(tmp_path),
        raw_data_root=None,
        build_cross_asset_audit=False,
        peer_raw_frames=peers,
    )

    summary_path = (
        tmp_path / "research_cross_asset_audit" / "XAU_USD" / "H1" / "summary.json"
    )
    assert not summary_path.exists()


def test_persist_market_context_writes_summary_and_validates_config_hash(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    market_context = pd.DataFrame(
        {
            "timestamp": timestamps,
            "timeframe": ["H1"] * len(timestamps),
            "corr_XAU_USD__DXY__w24": np.linspace(-0.2, 0.2, len(timestamps)),
        }
    )
    relevant = relevant_correlation_pairs("XAU_USD")

    artifacts = persist_market_context(
        market_context,
        features_root=tmp_path,
        timeframe="H1",
        variant="live",
        frontier_from_ts=None,
        full_rebuild=True,
        relevant_pairs=relevant,
    )

    summary = read_market_context_summary(
        features_root=tmp_path,
        timeframe="H1",
        variant="live",
    )
    assert summary is not None
    assert summary["variant"] == "live"
    assert any(Path(artifact.path).name == "summary.json" for artifact in artifacts)
    assert market_context_cache_is_current(
        features_root=tmp_path,
        timeframe="H1",
        variant="live",
        relevant_pairs=relevant,
    )
    assert not market_context_cache_is_current(
        features_root=tmp_path,
        timeframe="H1",
        variant="live",
        relevant_pairs=None,
    )


def test_load_raw_context_frames_uses_run_scoped_cache(tmp_path: Path) -> None:
    universe = _cross_asset_universe(rows=24)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    for symbol in ("DXY", "USD_JPY"):
        universe[symbol].to_parquet(raw_root / f"{symbol}_H1.parquet", index=False)

    frame_cache: dict[tuple[str, int, int], object] = {}
    first_details: dict[str, object] = {}
    second_details: dict[str, object] = {}

    first = load_raw_context_frames(
        raw_data_root=raw_root,
        timeframe="H1",
        instruments=("DXY", "USD_JPY"),
        frame_cache=frame_cache,
        runtime_details=first_details,
    )
    second = load_raw_context_frames(
        raw_data_root=raw_root,
        timeframe="H1",
        instruments=("DXY", "USD_JPY"),
        frame_cache=frame_cache,
        runtime_details=second_details,
    )

    assert sorted(first) == ["DXY", "USD_JPY"]
    assert sorted(second) == ["DXY", "USD_JPY"]
    assert first_details["raw_frame_disk_reads"] == 2
    assert first_details["raw_frame_cache_hits"] == 0
    assert second_details["raw_frame_disk_reads"] == 0
    assert second_details["raw_frame_cache_hits"] == 2
    assert second_details["raw_frame_read_bytes"] == 0
