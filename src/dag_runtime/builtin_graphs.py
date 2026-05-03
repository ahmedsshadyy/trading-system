from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.dag_runtime.artifacts import write_csv_atomic, write_text_atomic
from src.dag_runtime.contracts import (
    CachePolicy,
    MutableScope,
    ReplayPolicyContract,
    ValidationPolicy,
    WindowPolicy,
)
from src.dag_runtime.fingerprints import (
    compute_multi_source_hash,
    default_node_fingerprint_payload,
)
from src.dag_runtime.graph import GraphManifest
from src.dag_runtime.node import (
    GraphRunContext,
    NodeExecutionResult,
    NodeManifest,
    NodeOutput,
)
from src.validation.common.chart_core import save_figure_html
from src.validation.indicators.range_boundaries import validate_range_boundaries
from src.validation.indicators.regime import validate_regime
from src.validation.indicators.trend_state import validate_trend_state
from src.validation.indicators.sr_levels import summarize_sr_levels


def _runtime_config_fingerprint(*keys: str):
    def _fingerprint(
        manifest: NodeManifest,
        context: GraphRunContext,
        dependency_results: dict[str, NodeExecutionResult],
    ) -> dict[str, Any]:
        payload = default_node_fingerprint_payload(
            manifest, context, dependency_results
        )
        payload["runtime_config"] = {
            key: payload["runtime_config"][key]
            for key in keys
            if key in payload["runtime_config"]
        }
        return payload

    return _fingerprint


def _node_cache_policy(*, materialize: bool, artifact_kind: str) -> CachePolicy:
    return CachePolicy(materialize=materialize, artifact_kind=artifact_kind)


def _source_hash_config(*source_funcs: Any, materialize: bool = True) -> dict[str, Any]:
    if not materialize or not source_funcs:
        return {}
    return {"source_hash": compute_multi_source_hash(*source_funcs)}


def _source_node(graph_name: str, name: str, input_key: str) -> NodeManifest:
    def _fingerprint(
        manifest: NodeManifest,
        context: GraphRunContext,
        dependency_results: dict[str, NodeExecutionResult],
    ) -> dict[str, Any]:
        payload = default_node_fingerprint_payload(
            manifest, context, dependency_results
        )
        payload["runtime_config"] = {}
        return payload

    return NodeManifest(
        graph_name=graph_name,
        node_name=name,
        node_kind="source",
        semantic_class="A",
        inputs=(input_key,),
        upstream_nodes=(),
        output_artifacts=("frame",),
        fingerprint_fn=_fingerprint,
        cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
        validation_policy=ValidationPolicy(level="unit"),
        mutable_scope=MutableScope(scope="immutable"),
        compute_fn=lambda context, _deps, input_key=input_key: NodeOutput(
            frames={"frame": context.inputs[input_key].copy()}
        ),
    )


# --- Pipeline stage caching configuration ---
#
# Stages whose output is cached at the DAG node level.
#
# Carried-state Class B nodes are intentionally excluded until replay-context
# parity is proven for persistent node-cache reuse. The current safe promoted
# set is Class A nodes, `swings`, and non-carried `rsi_divergence`.
_PIPELINE_MATERIALIZE_NODES: frozenset[str] = frozenset(
    {
        "normalize_candles",
        "atr",
        "ema",
        "adx",
        "rsi",
        "macd",
        "bb_width",
        "body_ratio",
        "swings",
        "rsi_divergence",
        "rolling_atr_ratio",
        "volume_features",
        "prev_day_hl",
        "prev_week_hl",
        "round_number_flag",
        "intraday_context",
        # Sweeps v2 (Steps 9-11). All three are promoted because they
        # dominate cold-run time (per-bar Python loops). The research
        # pipeline always processes full frames, so the fingerprint-based
        # cache lookup is sound; the live pipeline is intentionally not
        # wired to these stages, so partial-replay parity for the
        # carried-state ones is not exercised here.
        "sr_levels",
        "unified_liquidity_sources",
        "final_sweeps",
    }
)


def _pipeline_stage_source_funcs(stage_name: str) -> tuple[Any, ...]:
    """Return the underlying functions whose source determines a stage's output.

    The DAG node's compute_fn is a small lambda wrapper. Hashing only the
    wrapper would miss logic changes inside the wrapped indicator function
    (e.g. editing ``add_atr``'s body wouldn't change the lambda's source).
    This map names the actual functions the wrapper delegates to so we hash
    them all and embed the result in the node fingerprint.
    """
    # Import lazily to avoid pulling indicator modules at import time
    # (build_live / build_research already import them for stage construction).
    from src.indicators._helpers.schema import normalize_candle_schema
    from src.indicators.foundation.adx import add_adx
    from src.indicators.foundation.ema import add_emas
    from src.indicators.foundation.momentum import (
        add_macd,
        add_rsi,
        add_rsi_divergence,
    )
    from src.indicators.foundation.regime import add_regime
    from src.indicators.foundation.session import add_session_features
    from src.indicators.foundation.value import (
        add_anchored_vwap,
        add_asian_session_hl,
        add_prev_day_hl,
        add_prev_week_hl,
        add_round_number_flag,
    )
    from src.indicators.foundation.volatility import (
        add_atr,
        add_bb_width,
        add_body_ratio,
        add_rolling_atr_ratio,
    )
    from src.indicators.foundation.volume import add_volume_features
    from src.indicators.smc.amd import add_amd_engine
    from src.indicators.smc.displacement import add_displacement_candle
    from src.indicators.smc.equal_hl import add_equal_hl
    from src.indicators.smc.fvg import collect_fvg_debug_tables
    from src.indicators.smc.fvg_fill import add_fvg_fill
    from src.indicators.smc.ifvg import add_ifvg
    from src.indicators.smc.ob import add_ob
    from src.indicators.smc.ob_mitigation import add_ob_mitigation
    from src.indicators.structure.bos import add_bos
    from src.indicators.structure.choch import add_choch
    from src.indicators.structure.trend_state import add_trend_state
    from src.indicators.structure.swings import add_swings

    # Sweeps v2 (Steps 9-11)
    from src.indicators.foundation.sr_levels import (
        add_sr_levels,
        build_sr_level_registry,
        project_sr_context,
        update_sr_lifecycle,
    )
    from src.indicators.smc.sweeps.unified_sources import (
        add_unified_liquidity_sources,
    )
    from src.indicators.smc.sweeps.final_sweeps import add_final_sweeps

    table: dict[str, tuple[Any, ...]] = {
        "normalize_candles": (normalize_candle_schema,),
        "atr": (add_atr,),
        "ema": (add_emas,),
        "adx": (add_adx,),
        "rsi": (add_rsi,),
        "macd": (add_macd,),
        "bb_width": (add_bb_width,),
        "body_ratio": (add_body_ratio,),
        "swings": (add_swings,),
        "trend_state": (add_trend_state,),
        "bos": (add_bos,),
        "choch": (add_choch,),
        "rsi_divergence": (add_rsi_divergence,),
        "rolling_atr_ratio": (add_rolling_atr_ratio,),
        "volume_features": (add_volume_features,),
        "fvg_stack": (collect_fvg_debug_tables, add_fvg_fill, add_ifvg),
        "displacement": (add_displacement_candle,),
        "order_blocks": (add_ob,),
        "ob_mitigation": (add_ob_mitigation,),
        "equal_hl": (add_equal_hl,),
        "amd_engine": (add_amd_engine,),
        "prev_day_hl": (add_prev_day_hl,),
        "prev_week_hl": (add_prev_week_hl,),
        "round_number_flag": (add_round_number_flag,),
        "anchored_vwap": (add_anchored_vwap,),
        "regime": (add_regime,),
        # intraday_context conditionally calls both — hash both so either's
        # source change invalidates.
        "intraday_context": (add_session_features, add_asian_session_hl),
        # Sweeps v2 (Steps 9-11). Hash all underlying functions so a logic
        # change in any of them invalidates the node + downstream cache.
        "sr_levels": (
            add_sr_levels,
            build_sr_level_registry,
            project_sr_context,
            update_sr_lifecycle,
        ),
        "unified_liquidity_sources": (add_unified_liquidity_sources,),
        "final_sweeps": (add_final_sweeps,),
    }
    return table.get(stage_name, ())


def _pipeline_stage_config(stage: Any) -> dict[str, Any]:
    """Build the per-node ``config`` used in the fingerprint.

    For materialized stages we embed a hash of the underlying indicator
    function's source so any logic change invalidates the cache. For
    non-materialized stages we return an empty config (fingerprint
    unaffected — saves a bit of hashing work).
    """
    if stage.name not in _PIPELINE_MATERIALIZE_NODES:
        return {}
    source_funcs = _pipeline_stage_source_funcs(stage.name)
    return _source_hash_config(*source_funcs)


def _pipeline_cache_policy(stage: Any) -> CachePolicy:
    """CachePolicy for a pipeline stage based on the materialize whitelist."""
    if stage.name in _PIPELINE_MATERIALIZE_NODES:
        return _node_cache_policy(materialize=True, artifact_kind="frame")
    return _node_cache_policy(materialize=False, artifact_kind="ephemeral")


def _live_peer_symbols(instrument: str) -> tuple[str, ...]:
    from src.indicators.features.cross_asset import CONTEXT_SYMBOLS

    return tuple(symbol for symbol in CONTEXT_SYMBOLS if symbol != instrument)


def _live_partner_node_name(symbol: str) -> str:
    return f"live_partner_{symbol}"


def _live_market_context_symbols(instrument: str) -> tuple[str, ...]:
    from src.indicators.features.cross_asset import relevant_correlation_pairs

    relevant_pairs = relevant_correlation_pairs(instrument)
    return tuple(
        sorted(
            {value for pair in relevant_pairs for value in pair if value != instrument}
        )
    )


def _live_peer_context_source_hash() -> str:
    from src.indicators._helpers.schema import normalize_candle_schema
    from src.indicators.features.cross_asset import load_raw_context_frames

    return compute_multi_source_hash(normalize_candle_schema, load_raw_context_frames)


def _live_market_context_source_hash() -> str:
    from src.indicators.features.cross_asset import (
        build_global_market_context,
        build_global_market_context_incremental,
        market_context_cache_is_current,
        relevant_correlation_pairs,
    )

    return compute_multi_source_hash(
        build_global_market_context,
        build_global_market_context_incremental,
        market_context_cache_is_current,
        relevant_correlation_pairs,
    )


def _live_partner_source_hash(symbol: str) -> str:
    from src.indicators._helpers.schema import normalize_candle_schema
    from src.indicators.foundation.adx import add_adx
    from src.indicators.foundation.ema import add_emas
    from src.indicators.foundation.momentum import add_macd, add_rsi
    from src.indicators.foundation.volatility import (
        add_atr,
        add_bb_width,
        add_body_ratio,
    )
    from src.indicators.pipelines.build_research import build_smt_partner_indicators
    from src.indicators.structure.swings import add_swings

    return compute_multi_source_hash(
        build_smt_partner_indicators,
        normalize_candle_schema,
        add_atr,
        add_emas,
        add_adx,
        add_rsi,
        add_macd,
        add_bb_width,
        add_body_ratio,
        add_swings,
        lambda value=symbol: value,
    )


def _live_cross_asset_attach_hash() -> str:
    from src.indicators.features.cross_asset import attach_cross_asset_context

    return compute_multi_source_hash(attach_cross_asset_context)


def _live_peer_context_fingerprint(
    manifest: NodeManifest,
    context: GraphRunContext,
    dependency_results: dict[str, NodeExecutionResult],
) -> dict[str, Any]:
    from src.pipeline_runtime import dataframe_fingerprint

    payload = default_node_fingerprint_payload(manifest, context, dependency_results)
    instrument = str(context.config.get("instrument", context.symbol))
    timeframe = str(context.config.get("timeframe", context.timeframe))
    raw_data_root = context.config.get("raw_data_root")
    provided = context.inputs.get("peer_raw_frames") or {}
    peer_inputs: dict[str, Any] = {}
    for symbol in _live_peer_symbols(instrument):
        frame = provided.get(symbol) if isinstance(provided, dict) else None
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            peer_inputs[symbol] = {
                "source": "input",
                "fingerprint": dataframe_fingerprint(frame, strategy="content"),
            }
            continue
        if raw_data_root is None:
            peer_inputs[symbol] = {"source": "missing"}
            continue
        path = Path(raw_data_root) / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            peer_inputs[symbol] = {"source": "missing"}
            continue
        stat = path.stat()
        peer_inputs[symbol] = {
            "source": "file",
            "path": str(path.resolve()),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    payload["runtime_config"] = {
        key: payload["runtime_config"][key]
        for key in ("timeframe", "raw_data_root")
        if key in payload["runtime_config"]
    }
    payload["peer_inputs"] = peer_inputs
    payload["input_fingerprints"] = {}
    return payload


def _live_market_context_fingerprint(
    manifest: NodeManifest,
    context: GraphRunContext,
    dependency_results: dict[str, NodeExecutionResult],
) -> dict[str, Any]:
    from src.indicators.features.cross_asset import (
        cross_asset_runtime_config_hash,
        relevant_correlation_pairs,
    )
    from src.pipeline_runtime import dataframe_fingerprint

    instrument = str(context.config.get("instrument", context.symbol))
    timeframe = str(context.config.get("timeframe", context.timeframe))
    relevant_pairs = relevant_correlation_pairs(instrument)
    peer_frames = dependency_results["live_peer_context_source"].output.frames
    relevant_symbols = _live_market_context_symbols(instrument)
    return {
        "graph_name": manifest.graph_name,
        "node_name": manifest.node_name,
        "config": dict(manifest.config),
        "runtime_config": {"timeframe": timeframe},
        "primary_input_fingerprint": dataframe_fingerprint(
            dependency_results["raw_input"].primary_frame(), strategy="content"
        ),
        "relevant_peer_fingerprints": {
            symbol: (
                dataframe_fingerprint(peer_frames[symbol], strategy="content")
                if symbol in peer_frames and peer_frames[symbol] is not None
                else None
            )
            for symbol in relevant_symbols
        },
        "cross_asset_config_hash": cross_asset_runtime_config_hash(
            timeframe=timeframe,
            relevant_pairs=relevant_pairs,
        ),
        "relevant_pairs": sorted(relevant_pairs),
    }


def _live_partner_fingerprint(symbol: str):
    def _fingerprint(
        manifest: NodeManifest,
        context: GraphRunContext,
        dependency_results: dict[str, NodeExecutionResult],
    ) -> dict[str, Any]:
        from src.indicators._helpers.schema import normalize_candle_schema
        from src.indicators.features.cross_asset import (
            _trim_frame_to_range,
            _warmup_buffer,
        )
        from src.pipeline_runtime import dataframe_fingerprint

        timeframe = str(context.config.get("timeframe", context.timeframe))
        primary_raw = normalize_candle_schema(
            dependency_results["raw_input"].primary_frame().copy(),
            require_volume=True,
        )
        primary_ts = pd.to_datetime(
            primary_raw["timestamp"], utc=True, errors="coerce"
        ).dropna()
        primary_max = primary_ts.max()
        primary_min = primary_ts.min() - _warmup_buffer(timeframe)
        peer_frame = dependency_results["live_peer_context_source"].output.frames.get(
            symbol
        )
        trimmed = (
            _trim_frame_to_range(peer_frame, min_ts=primary_min, max_ts=primary_max)
            if peer_frame is not None and not peer_frame.empty
            else pd.DataFrame()
        )
        return {
            "graph_name": manifest.graph_name,
            "node_name": manifest.node_name,
            "config": dict(manifest.config),
            "runtime_config": {"timeframe": timeframe, "symbol": symbol},
            "trimmed_partner_fingerprint": dataframe_fingerprint(
                trimmed, strategy="content"
            ),
        }

    return _fingerprint


def _live_cross_asset_attach_fingerprint(
    partner_node_names: tuple[str, ...],
):
    def _fingerprint(
        manifest: NodeManifest,
        context: GraphRunContext,
        dependency_results: dict[str, NodeExecutionResult],
    ) -> dict[str, Any]:
        from src.pipeline_runtime import dataframe_fingerprint

        primary_node = manifest.upstream_nodes[0]
        return {
            "graph_name": manifest.graph_name,
            "node_name": manifest.node_name,
            "config": dict(manifest.config),
            "runtime_config": {
                key: context.config.get(key) for key in ("instrument", "timeframe")
            },
            "primary_frame_fingerprint": dataframe_fingerprint(
                dependency_results[primary_node].primary_frame(), strategy="content"
            ),
            "market_context_fingerprint": dependency_results[
                "live_market_context_source"
            ].fingerprint,
            "partner_fingerprints": {
                name: dependency_results[name].fingerprint
                for name in partner_node_names
            },
        }

    return _fingerprint


def _research_partner_node_name(symbol: str) -> str:
    return f"research_partner_{symbol}"


def _research_peer_context_source_hash() -> str:
    return _live_peer_context_source_hash()


def _research_market_context_source_hash() -> str:
    from src.indicators.features.cross_asset import (
        build_global_market_context,
        build_global_market_context_incremental,
        market_context_cache_is_current,
    )

    return compute_multi_source_hash(
        build_global_market_context,
        build_global_market_context_incremental,
        market_context_cache_is_current,
    )


def _research_partner_source_hash(symbol: str) -> str:
    return _live_partner_source_hash(symbol)


def _research_cross_asset_attach_hash() -> str:
    return _live_cross_asset_attach_hash()


def _research_smt_research_hash() -> str:
    from src.indicators.research.smt_research import (
        build_smt_research_table,
        summarize_smt_research,
    )

    return compute_multi_source_hash(build_smt_research_table, summarize_smt_research)


def _research_cross_asset_audit_hash() -> str:
    from src.indicators.research.cross_asset_research import (
        build_cross_asset_correlation_audit,
        summarize_cross_asset_correlation_audit,
    )

    return compute_multi_source_hash(
        build_cross_asset_correlation_audit,
        summarize_cross_asset_correlation_audit,
    )


def _research_market_context_fingerprint(
    manifest: NodeManifest,
    context: GraphRunContext,
    dependency_results: dict[str, NodeExecutionResult],
) -> dict[str, Any]:
    from src.indicators.features.cross_asset import cross_asset_runtime_config_hash
    from src.pipeline_runtime import dataframe_fingerprint

    timeframe = str(context.config.get("timeframe", context.timeframe))
    peer_frames = dependency_results["research_peer_context_source"].output.frames
    return {
        "graph_name": manifest.graph_name,
        "node_name": manifest.node_name,
        "config": dict(manifest.config),
        "runtime_config": {"timeframe": timeframe, "full_pair_matrix": True},
        "primary_input_fingerprint": dataframe_fingerprint(
            dependency_results["raw_input"].primary_frame(), strategy="content"
        ),
        "peer_frame_fingerprints": {
            symbol: dataframe_fingerprint(frame, strategy="content")
            for symbol, frame in sorted(peer_frames.items())
            if frame is not None and not frame.empty
        },
        "cross_asset_config_hash": cross_asset_runtime_config_hash(
            timeframe=timeframe,
            relevant_pairs=None,
        ),
    }


def _research_partner_fingerprint(symbol: str):
    def _fingerprint(
        manifest: NodeManifest,
        context: GraphRunContext,
        dependency_results: dict[str, NodeExecutionResult],
    ) -> dict[str, Any]:
        from src.indicators._helpers.schema import normalize_candle_schema
        from src.indicators.features.cross_asset import (
            _trim_frame_to_range,
            _warmup_buffer,
        )
        from src.pipeline_runtime import dataframe_fingerprint

        timeframe = str(context.config.get("timeframe", context.timeframe))
        primary_raw = normalize_candle_schema(
            dependency_results["raw_input"].primary_frame().copy(),
            require_volume=True,
        )
        primary_ts = pd.to_datetime(
            primary_raw["timestamp"], utc=True, errors="coerce"
        ).dropna()
        primary_max = primary_ts.max()
        primary_min = primary_ts.min() - _warmup_buffer(timeframe)
        peer_frame = dependency_results[
            "research_peer_context_source"
        ].output.frames.get(symbol)
        trimmed = (
            _trim_frame_to_range(peer_frame, min_ts=primary_min, max_ts=primary_max)
            if peer_frame is not None and not peer_frame.empty
            else pd.DataFrame()
        )
        return {
            "graph_name": manifest.graph_name,
            "node_name": manifest.node_name,
            "config": dict(manifest.config),
            "runtime_config": {"timeframe": timeframe, "symbol": symbol},
            "trimmed_partner_fingerprint": dataframe_fingerprint(
                trimmed, strategy="content"
            ),
        }

    return _fingerprint


def _research_cross_asset_attach_fingerprint(
    partner_node_names: tuple[str, ...],
):
    def _fingerprint(
        manifest: NodeManifest,
        context: GraphRunContext,
        dependency_results: dict[str, NodeExecutionResult],
    ) -> dict[str, Any]:
        from src.pipeline_runtime import dataframe_fingerprint

        primary_node = manifest.upstream_nodes[0]
        return {
            "graph_name": manifest.graph_name,
            "node_name": manifest.node_name,
            "config": dict(manifest.config),
            "runtime_config": {
                key: context.config.get(key) for key in ("instrument", "timeframe")
            },
            "primary_frame_fingerprint": dataframe_fingerprint(
                dependency_results[primary_node].primary_frame(), strategy="content"
            ),
            "market_context_fingerprint": dependency_results[
                "research_market_context_source"
            ].fingerprint,
            "partner_fingerprints": {
                name: dependency_results[name].fingerprint
                for name in partner_node_names
            },
        }

    return _fingerprint


def _research_smt_research_fingerprint(
    manifest: NodeManifest,
    context: GraphRunContext,
    dependency_results: dict[str, NodeExecutionResult],
) -> dict[str, Any]:
    from src.pipeline_runtime import dataframe_fingerprint

    return {
        "graph_name": manifest.graph_name,
        "node_name": manifest.node_name,
        "config": dict(manifest.config),
        "runtime_config": {
            key: context.config.get(key) for key in ("instrument", "timeframe")
        },
        "attach_node_fingerprint": dependency_results[
            "research_cross_asset_attach"
        ].fingerprint,
        "attached_frame_fingerprint": dataframe_fingerprint(
            dependency_results["research_cross_asset_attach"].primary_frame(),
            strategy="content",
        ),
    }


def _research_cross_asset_audit_fingerprint(
    manifest: NodeManifest,
    context: GraphRunContext,
    dependency_results: dict[str, NodeExecutionResult],
) -> dict[str, Any]:
    from src.pipeline_runtime import dataframe_fingerprint

    return {
        "graph_name": manifest.graph_name,
        "node_name": manifest.node_name,
        "config": dict(manifest.config),
        "runtime_config": {
            key: context.config.get(key) for key in ("instrument", "timeframe")
        },
        "attach_node_fingerprint": dependency_results[
            "research_cross_asset_attach"
        ].fingerprint,
        "attached_frame_fingerprint": dataframe_fingerprint(
            dependency_results["research_cross_asset_attach"].primary_frame(),
            strategy="content",
        ),
        "market_context_fingerprint": dependency_results[
            "research_market_context_source"
        ].fingerprint,
    }


def build_live_stage_graph(
    *,
    instrument: str,
    swing_window: int,
    include_vp: bool,
    timeframe: str = "H4",
    include_cross_asset: bool = False,
) -> GraphManifest:
    from src.indicators.pipelines import build_live as live
    from src.indicators._helpers.schema import normalize_candle_schema
    from src.indicators.features.cross_asset import (
        CONTEXT_SYMBOLS,
        SMT_PARTNERS,
        SUPPORTED_CROSS_ASSET_TIMEFRAMES,
        _trim_frame_to_range,
        _warmup_buffer,
        attach_cross_asset_context,
        build_global_market_context,
        build_global_market_context_incremental,
        load_raw_context_frames,
        market_context_cache_is_current,
        relevant_correlation_pairs,
    )
    from src.indicators.pipelines.build_research import build_smt_partner_indicators
    from src.pipeline_runtime import load_partitioned_dataset

    graph_name = "live_pipeline"
    nodes: list[NodeManifest] = [_source_node(graph_name, "raw_input", "raw_input")]
    upstream = "raw_input"
    for stage in live._live_stages(
        instrument=instrument,
        swing_window=swing_window,
        include_vp=include_vp,
        timeframe=timeframe,
    ):
        prev = upstream
        nodes.append(
            NodeManifest(
                graph_name=graph_name,
                node_name=stage.name,
                node_kind="compute",
                semantic_class=stage.policy.classification,
                inputs=(),
                upstream_nodes=(prev,),
                output_artifacts=("frame",),
                cache_policy=_pipeline_cache_policy(stage),
                config=_pipeline_stage_config(stage),
                replay_policy=ReplayPolicyContract(
                    mode=(
                        "carried_state"
                        if stage.policy.carried_state
                        else (
                            "bounded_replay"
                            if stage.policy.replay_bars > 0
                            else "append_only_safe"
                        )
                    ),
                    replay_bars=stage.policy.replay_bars,
                    notes=stage.policy.notes,
                ),
                window_policy=WindowPolicy(mode="full"),
                validation_policy=ValidationPolicy(level="node_parity"),
                mutable_scope=MutableScope(scope="frontier_only"),
                compute_fn=lambda context, deps, stage=stage, prev=prev: NodeOutput(
                    frames={"frame": stage.fn(deps[prev].primary_frame().copy())}
                ),
            )
        )
        upstream = stage.name
    if not include_cross_asset:
        return GraphManifest(
            graph_name=graph_name, nodes=tuple(nodes), default_target=upstream
        )

    if timeframe not in SUPPORTED_CROSS_ASSET_TIMEFRAMES:
        raise ValueError(
            f"Cross-asset context only supports {sorted(SUPPORTED_CROSS_ASSET_TIMEFRAMES)}"
        )

    peer_context_node = "live_peer_context_source"
    materialized_bundle_policy = _node_cache_policy(
        materialize=True, artifact_kind="bundle"
    )
    materialized_frame_policy = _node_cache_policy(
        materialize=True, artifact_kind="frame"
    )
    ephemeral_policy = _node_cache_policy(materialize=False, artifact_kind="ephemeral")
    partner_symbols = tuple(
        partner for partner, _relation in SMT_PARTNERS.get(instrument, ())
    )

    def _compute_peer_context(
        context: GraphRunContext, _deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        provided = context.inputs.get("peer_raw_frames") or {}
        symbols = _live_peer_symbols(instrument)
        frames: dict[str, pd.DataFrame] = {}
        loaded_from_input: list[str] = []
        for symbol in symbols:
            frame = provided.get(symbol) if isinstance(provided, dict) else None
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames[symbol] = normalize_candle_schema(
                    frame.copy(), require_volume=True
                ).reset_index(drop=True)
                loaded_from_input.append(symbol)
        missing_symbols = tuple(symbol for symbol in symbols if symbol not in frames)
        raw_data_root = context.config.get("raw_data_root")
        if raw_data_root is not None and missing_symbols:
            loaded = load_raw_context_frames(
                raw_data_root=raw_data_root,
                timeframe=timeframe,
                instruments=missing_symbols,
            )
            for symbol, frame in loaded.items():
                frames.setdefault(symbol, frame.reset_index(drop=True))
        return NodeOutput(
            frames=frames,
            payload={"symbols": sorted(frames)},
            profile_details={
                "loaded_from_input": loaded_from_input,
                "loaded_from_raw": sorted(
                    symbol for symbol in frames if symbol not in loaded_from_input
                ),
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=peer_context_node,
            node_kind="source",
            semantic_class="A",
            inputs=(),
            upstream_nodes=(),
            output_artifacts=("payload",),
            fingerprint_fn=_live_peer_context_fingerprint,
            cache_policy=materialized_bundle_policy,
            config={"source_hash": _live_peer_context_source_hash()},
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=_compute_peer_context,
        )
    )

    def _compute_live_market_context(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        primary_raw = normalize_candle_schema(
            deps["raw_input"].primary_frame().copy(),
            require_volume=True,
        )
        relevant_pairs = relevant_correlation_pairs(instrument)
        cache_current = market_context_cache_is_current(
            features_root=context.features_root,
            timeframe=timeframe,
            variant="live",
            relevant_pairs=relevant_pairs,
        )
        cached_mc = load_partitioned_dataset(
            context.features_root,
            dataset="market_context_live",
            symbol="GLOBAL",
            timeframe=timeframe,
        )
        primary_ts = pd.to_datetime(
            primary_raw["timestamp"], utc=True, errors="coerce"
        ).dropna()
        if cache_current and not cached_mc.empty:
            mc_ts = pd.to_datetime(
                cached_mc["timestamp"], utc=True, errors="coerce"
            ).dropna()
            if mc_ts.min() <= primary_ts.min() and mc_ts.max() >= primary_ts.max():
                return NodeOutput(
                    frames={"frame": cached_mc.reset_index(drop=True)},
                    profile_details={"market_context_source": "persisted-full"},
                )

        peer_frames = {
            symbol: frame.reset_index(drop=True)
            for symbol, frame in deps[peer_context_node].output.frames.items()
            if frame is not None and not frame.empty
        }
        raw_context_frames: dict[str, pd.DataFrame] = {}
        required_symbols = sorted(
            {value for pair in relevant_pairs for value in pair if value != instrument}
        )
        for symbol in required_symbols:
            frame = peer_frames.get(symbol)
            if frame is not None and not frame.empty:
                raw_context_frames[symbol] = frame
        if instrument in CONTEXT_SYMBOLS:
            raw_context_frames[instrument] = primary_raw

        if cache_current and not cached_mc.empty:
            mc_ts = pd.to_datetime(
                cached_mc["timestamp"], utc=True, errors="coerce"
            ).dropna()
            if (
                mc_ts.min() <= primary_ts.min()
                and mc_ts.max() < primary_ts.max()
                and mc_ts.max() >= primary_ts.min()
            ):
                frontier_from_ts = mc_ts.max() - _warmup_buffer(timeframe)
                rebuilt = build_global_market_context_incremental(
                    raw_context_frames,
                    timeframe=timeframe,
                    prior_context=cached_mc,
                    frontier_from_ts=frontier_from_ts,
                    relevant_pairs=relevant_pairs,
                )
                return NodeOutput(
                    frames={"frame": rebuilt.reset_index(drop=True)},
                    profile_details={"market_context_source": "persisted-incremental"},
                )

        built = build_global_market_context(
            raw_context_frames,
            timeframe=timeframe,
            relevant_pairs=relevant_pairs,
        )
        return NodeOutput(
            frames={"frame": built.reset_index(drop=True)},
            profile_details={"market_context_source": "build"},
        )

    market_context_node = "live_market_context_source"
    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=market_context_node,
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input", peer_context_node),
            output_artifacts=("frame",),
            fingerprint_fn=_live_market_context_fingerprint,
            cache_policy=materialized_frame_policy,
            config={"source_hash": _live_market_context_source_hash()},
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=_compute_live_market_context,
        )
    )

    partner_node_names: list[str] = []
    for partner_symbol in partner_symbols:
        node_name = _live_partner_node_name(partner_symbol)
        partner_node_names.append(node_name)

        def _compute_partner(
            context: GraphRunContext,
            deps: dict[str, NodeExecutionResult],
            partner_symbol: str = partner_symbol,
        ) -> NodeOutput:
            primary_raw = normalize_candle_schema(
                deps["raw_input"].primary_frame().copy(),
                require_volume=True,
            )
            primary_ts = pd.to_datetime(
                primary_raw["timestamp"], utc=True, errors="coerce"
            ).dropna()
            primary_max = primary_ts.max()
            primary_min = primary_ts.min() - _warmup_buffer(timeframe)
            raw_partner = deps[peer_context_node].output.frames.get(partner_symbol)
            if raw_partner is None or raw_partner.empty:
                return NodeOutput(frames={"frame": pd.DataFrame()})
            trimmed = _trim_frame_to_range(
                raw_partner,
                min_ts=primary_min,
                max_ts=primary_max,
            )
            built = build_smt_partner_indicators(
                trimmed.copy(),
                swing_window=swing_window,
            )
            return NodeOutput(frames={"frame": built.reset_index(drop=True)})

        nodes.append(
            NodeManifest(
                graph_name=graph_name,
                node_name=node_name,
                node_kind="compute",
                semantic_class="B",
                inputs=(),
                upstream_nodes=("raw_input", peer_context_node),
                output_artifacts=("frame",),
                fingerprint_fn=_live_partner_fingerprint(partner_symbol),
                cache_policy=materialized_frame_policy,
                config={
                    "partner_symbol": partner_symbol,
                    "source_hash": _live_partner_source_hash(partner_symbol),
                },
                validation_policy=ValidationPolicy(level="node_parity"),
                mutable_scope=MutableScope(scope="frontier_only"),
                compute_fn=_compute_partner,
            )
        )

    attach_node = "live_cross_asset_attach"

    def _compute_attach(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        processed_frames = {
            symbol: deps[_live_partner_node_name(symbol)].primary_frame()
            for symbol in partner_symbols
            if deps[_live_partner_node_name(symbol)].primary_frame() is not None
            and not deps[_live_partner_node_name(symbol)].primary_frame().empty
        }
        attached = attach_cross_asset_context(
            deps[upstream].primary_frame().copy(),
            instrument=instrument,
            timeframe=timeframe,
            market_context=deps[market_context_node].primary_frame(),
            processed_frames=processed_frames,
        )
        return NodeOutput(frames={"frame": attached.reset_index(drop=True)})

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=attach_node,
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=(upstream, market_context_node, *tuple(partner_node_names)),
            output_artifacts=("frame",),
            fingerprint_fn=_live_cross_asset_attach_fingerprint(
                tuple(partner_node_names)
            ),
            cache_policy=ephemeral_policy,
            config={"source_hash": _live_cross_asset_attach_hash()},
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=_compute_attach,
        )
    )

    bundle_node = "live_feature_bundle"
    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=bundle_node,
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(attach_node,),
            output_artifacts=("frame",),
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={"frame": deps[attach_node].primary_frame().copy()}
            ),
        )
    )
    return GraphManifest(
        graph_name=graph_name, nodes=tuple(nodes), default_target=bundle_node
    )


def build_research_stage_graph(
    *,
    instrument: str,
    swing_window: int,
    include_vp: bool,
    include_avwap: bool,
    timeframe: str = "H4",
    include_cross_asset: bool = False,
) -> GraphManifest:
    from src.indicators.pipelines import build_research as research
    from src.indicators._helpers.schema import normalize_candle_schema
    from src.indicators.features.cross_asset import (
        CONTEXT_SYMBOLS,
        SMT_PARTNERS,
        SUPPORTED_CROSS_ASSET_TIMEFRAMES,
        _trim_frame_to_range,
        _warmup_buffer,
        attach_cross_asset_context,
        build_global_market_context,
        build_global_market_context_incremental,
        load_raw_context_frames,
        market_context_cache_is_current,
    )
    from src.indicators.research import (
        build_cross_asset_correlation_audit,
        build_smt_research_table,
        summarize_cross_asset_correlation_audit,
        summarize_smt_research,
    )
    from src.pipeline_runtime import load_partitioned_dataset

    graph_name = "research_pipeline"
    nodes: list[NodeManifest] = [_source_node(graph_name, "raw_input", "raw_input")]
    upstream = "raw_input"
    for stage in research._research_stages(
        instrument=instrument,
        swing_window=swing_window,
        include_vp=include_vp,
        include_avwap=include_avwap,
        timeframe=timeframe,
    ):
        prev = upstream
        nodes.append(
            NodeManifest(
                graph_name=graph_name,
                node_name=stage.name,
                node_kind="compute",
                semantic_class=stage.policy.classification,
                inputs=(),
                upstream_nodes=(prev,),
                output_artifacts=("frame",),
                cache_policy=_pipeline_cache_policy(stage),
                config=_pipeline_stage_config(stage),
                replay_policy=ReplayPolicyContract(
                    mode=(
                        "carried_state"
                        if stage.policy.carried_state
                        else (
                            "bounded_replay"
                            if stage.policy.replay_bars > 0
                            else "append_only_safe"
                        )
                    ),
                    replay_bars=stage.policy.replay_bars,
                    notes=stage.policy.notes,
                ),
                validation_policy=ValidationPolicy(level="node_parity"),
                mutable_scope=MutableScope(
                    scope=(
                        "explicit_rebuild_only"
                        if stage.policy.classification == "C"
                        else "frontier_only"
                    )
                ),
                compute_fn=lambda context, deps, stage=stage, prev=prev: NodeOutput(
                    frames={"frame": stage.fn(deps[prev].primary_frame().copy())}
                ),
            )
        )
        upstream = stage.name
    if not include_cross_asset:
        return GraphManifest(
            graph_name=graph_name, nodes=tuple(nodes), default_target=upstream
        )

    if timeframe not in SUPPORTED_CROSS_ASSET_TIMEFRAMES:
        raise ValueError(
            f"Cross-asset context only supports {sorted(SUPPORTED_CROSS_ASSET_TIMEFRAMES)}"
        )

    peer_context_node = "research_peer_context_source"
    market_context_node = "research_market_context_source"
    attach_node = "research_cross_asset_attach"
    smt_node = "research_smt_research_table"
    audit_node = "research_cross_asset_audit"
    bundle_node = "research_feature_bundle"
    full_bundle_node = "research_full_bundle"
    materialized_bundle_policy = _node_cache_policy(
        materialize=True, artifact_kind="bundle"
    )
    materialized_frame_policy = _node_cache_policy(
        materialize=True, artifact_kind="frame"
    )
    ephemeral_policy = _node_cache_policy(materialize=False, artifact_kind="ephemeral")
    partner_symbols = tuple(
        partner for partner, _relation in SMT_PARTNERS.get(instrument, ())
    )

    def _compute_peer_context(
        context: GraphRunContext, _deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        provided = context.inputs.get("peer_raw_frames") or {}
        symbols = _live_peer_symbols(instrument)
        frames: dict[str, pd.DataFrame] = {}
        loaded_from_input: list[str] = []
        for symbol in symbols:
            frame = provided.get(symbol) if isinstance(provided, dict) else None
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames[symbol] = normalize_candle_schema(
                    frame.copy(), require_volume=True
                ).reset_index(drop=True)
                loaded_from_input.append(symbol)
        missing_symbols = tuple(symbol for symbol in symbols if symbol not in frames)
        raw_data_root = context.config.get("raw_data_root")
        if raw_data_root is not None and missing_symbols:
            loaded = load_raw_context_frames(
                raw_data_root=raw_data_root,
                timeframe=timeframe,
                instruments=missing_symbols,
            )
            for symbol, frame in loaded.items():
                frames.setdefault(symbol, frame.reset_index(drop=True))
        return NodeOutput(
            frames=frames,
            payload={"symbols": sorted(frames)},
            profile_details={
                "loaded_from_input": loaded_from_input,
                "loaded_from_raw": sorted(
                    symbol for symbol in frames if symbol not in loaded_from_input
                ),
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=peer_context_node,
            node_kind="source",
            semantic_class="A",
            inputs=(),
            upstream_nodes=(),
            output_artifacts=("payload",),
            fingerprint_fn=_live_peer_context_fingerprint,
            cache_policy=materialized_bundle_policy,
            config={"source_hash": _research_peer_context_source_hash()},
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=_compute_peer_context,
        )
    )

    def _compute_research_market_context(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        primary_raw = normalize_candle_schema(
            deps["raw_input"].primary_frame().copy(),
            require_volume=True,
        )
        cache_current = market_context_cache_is_current(
            features_root=context.features_root,
            timeframe=timeframe,
            variant="research",
            relevant_pairs=None,
        )
        cached_mc = load_partitioned_dataset(
            context.features_root,
            dataset="market_context_research",
            symbol="GLOBAL",
            timeframe=timeframe,
        )
        primary_ts = pd.to_datetime(
            primary_raw["timestamp"], utc=True, errors="coerce"
        ).dropna()
        if cache_current and not cached_mc.empty:
            mc_ts = pd.to_datetime(
                cached_mc["timestamp"], utc=True, errors="coerce"
            ).dropna()
            if mc_ts.min() <= primary_ts.min() and mc_ts.max() >= primary_ts.max():
                return NodeOutput(
                    frames={"frame": cached_mc.reset_index(drop=True)},
                    profile_details={"market_context_source": "persisted-full"},
                )

        peer_frames = {
            symbol: frame.reset_index(drop=True)
            for symbol, frame in deps[peer_context_node].output.frames.items()
            if frame is not None and not frame.empty
        }
        raw_context_frames = dict(peer_frames)
        if instrument in CONTEXT_SYMBOLS:
            raw_context_frames[instrument] = primary_raw

        if cache_current and not cached_mc.empty:
            mc_ts = pd.to_datetime(
                cached_mc["timestamp"], utc=True, errors="coerce"
            ).dropna()
            if (
                mc_ts.min() <= primary_ts.min()
                and mc_ts.max() < primary_ts.max()
                and mc_ts.max() >= primary_ts.min()
            ):
                frontier_from_ts = mc_ts.max() - _warmup_buffer(timeframe)
                rebuilt = build_global_market_context_incremental(
                    raw_context_frames,
                    timeframe=timeframe,
                    prior_context=cached_mc,
                    frontier_from_ts=frontier_from_ts,
                    relevant_pairs=None,
                )
                return NodeOutput(
                    frames={"frame": rebuilt.reset_index(drop=True)},
                    profile_details={"market_context_source": "persisted-incremental"},
                )

        built = build_global_market_context(
            raw_context_frames,
            timeframe=timeframe,
            relevant_pairs=None,
        )
        return NodeOutput(
            frames={"frame": built.reset_index(drop=True)},
            profile_details={"market_context_source": "build"},
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=market_context_node,
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input", peer_context_node),
            output_artifacts=("frame",),
            fingerprint_fn=_research_market_context_fingerprint,
            cache_policy=materialized_frame_policy,
            config={"source_hash": _research_market_context_source_hash()},
            replay_policy=ReplayPolicyContract(mode="bounded_replay", replay_bars=200),
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=_compute_research_market_context,
        )
    )

    partner_node_names: list[str] = []
    for partner_symbol in partner_symbols:
        node_name = _research_partner_node_name(partner_symbol)
        partner_node_names.append(node_name)

        def _compute_partner(
            context: GraphRunContext,
            deps: dict[str, NodeExecutionResult],
            partner_symbol: str = partner_symbol,
        ) -> NodeOutput:
            primary_raw = normalize_candle_schema(
                deps["raw_input"].primary_frame().copy(),
                require_volume=True,
            )
            primary_ts = pd.to_datetime(
                primary_raw["timestamp"], utc=True, errors="coerce"
            ).dropna()
            primary_max = primary_ts.max()
            primary_min = primary_ts.min() - _warmup_buffer(timeframe)
            raw_partner = deps[peer_context_node].output.frames.get(partner_symbol)
            if raw_partner is None or raw_partner.empty:
                return NodeOutput(frames={"frame": pd.DataFrame()})
            trimmed = _trim_frame_to_range(
                raw_partner,
                min_ts=primary_min,
                max_ts=primary_max,
            )
            built = research.build_smt_partner_indicators(
                trimmed.copy(),
                swing_window=swing_window,
            )
            return NodeOutput(frames={"frame": built.reset_index(drop=True)})

        nodes.append(
            NodeManifest(
                graph_name=graph_name,
                node_name=node_name,
                node_kind="compute",
                semantic_class="B",
                inputs=(),
                upstream_nodes=("raw_input", peer_context_node),
                output_artifacts=("frame",),
                fingerprint_fn=_research_partner_fingerprint(partner_symbol),
                cache_policy=materialized_frame_policy,
                config={
                    "partner_symbol": partner_symbol,
                    "source_hash": _research_partner_source_hash(partner_symbol),
                },
                replay_policy=ReplayPolicyContract(
                    mode="bounded_replay", replay_bars=200
                ),
                validation_policy=ValidationPolicy(level="node_parity"),
                mutable_scope=MutableScope(scope="frontier_only"),
                compute_fn=_compute_partner,
            )
        )

    def _compute_attach(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        processed_frames = {
            symbol: deps[_research_partner_node_name(symbol)].primary_frame()
            for symbol in partner_symbols
            if deps[_research_partner_node_name(symbol)].primary_frame() is not None
            and not deps[_research_partner_node_name(symbol)].primary_frame().empty
        }
        attached = attach_cross_asset_context(
            deps[upstream].primary_frame().copy(),
            instrument=instrument,
            timeframe=timeframe,
            market_context=deps[market_context_node].primary_frame(),
            processed_frames=processed_frames,
        )
        return NodeOutput(frames={"frame": attached.reset_index(drop=True)})

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=attach_node,
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=(upstream, market_context_node, *tuple(partner_node_names)),
            output_artifacts=("frame",),
            fingerprint_fn=_research_cross_asset_attach_fingerprint(
                tuple(partner_node_names)
            ),
            cache_policy=ephemeral_policy,
            config={"source_hash": _research_cross_asset_attach_hash()},
            replay_policy=ReplayPolicyContract(mode="bounded_replay", replay_bars=200),
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=_compute_attach,
        )
    )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=smt_node,
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=(attach_node,),
            output_artifacts=("frame", "payload"),
            fingerprint_fn=_research_smt_research_fingerprint,
            cache_policy=materialized_frame_policy,
            config={"source_hash": _research_smt_research_hash()},
            replay_policy=ReplayPolicyContract(mode="bounded_replay", replay_bars=200),
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": build_smt_research_table(
                        deps[attach_node].primary_frame()
                    ).reset_index(drop=True)
                },
                payload={
                    "summary": summarize_smt_research(
                        build_smt_research_table(deps[attach_node].primary_frame())
                    )
                },
            ),
        )
    )

    def _compute_audit(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        audit_tables = build_cross_asset_correlation_audit(
            deps[attach_node].primary_frame(),
            deps[market_context_node].primary_frame(),
            instrument=instrument,
            timeframe=timeframe,
        )
        return NodeOutput(
            frames={
                table_name: table.reset_index(drop=True)
                for table_name, table in audit_tables.items()
            },
            payload={"summary": summarize_cross_asset_correlation_audit(audit_tables)},
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=audit_node,
            node_kind="compute",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(attach_node, market_context_node),
            output_artifacts=("payload",),
            fingerprint_fn=_research_cross_asset_audit_fingerprint,
            cache_policy=ephemeral_policy,
            config={"source_hash": _research_cross_asset_audit_hash()},
            validation_policy=ValidationPolicy(level="graph_parity"),
            mutable_scope=MutableScope(scope="explicit_rebuild_only"),
            compute_fn=_compute_audit,
        )
    )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=bundle_node,
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(attach_node, smt_node),
            output_artifacts=("frame", "payload"),
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": deps[attach_node].primary_frame().copy(),
                    "smt_research": deps[smt_node].primary_frame().copy(),
                },
                payload={"smt_summary": deps[smt_node].output.payload.get("summary")},
            ),
        )
    )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name=full_bundle_node,
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(bundle_node, audit_node),
            output_artifacts=("frame", "payload"),
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": deps[bundle_node].output.frames["frame"].copy(),
                    "smt_research": deps[bundle_node]
                    .output.frames["smt_research"]
                    .copy(),
                },
                payload={
                    "smt_summary": deps[bundle_node].output.payload.get("smt_summary"),
                    "audit_summary": deps[audit_node].output.payload.get("summary"),
                },
            ),
        )
    )

    return GraphManifest(
        graph_name=graph_name, nodes=tuple(nodes), default_target=bundle_node
    )


def build_range_boundaries_graph(*, instrument: str, timeframe: str) -> GraphManifest:
    from scripts import validate_range_boundaries as vrb

    graph_name = "validate_range_boundaries"
    nodes: list[NodeManifest] = [_source_node(graph_name, "raw_input", "raw_input")]

    no_runtime_config_fingerprint = _runtime_config_fingerprint()
    chart_runtime_config_fingerprint = _runtime_config_fingerprint(
        "html", "date_from", "plot_rows", "full", "out_dir"
    )
    csv_runtime_config_fingerprint = _runtime_config_fingerprint("write_csv", "out_dir")
    materialized_bundle_policy = _node_cache_policy(
        materialize=True, artifact_kind="bundle"
    )
    materialized_frame_policy = _node_cache_policy(
        materialize=True, artifact_kind="frame"
    )
    materialized_report_policy = _node_cache_policy(
        materialize=True, artifact_kind="report"
    )
    ephemeral_policy = _node_cache_policy(materialize=False, artifact_kind="ephemeral")

    def context_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        raw = deps["raw_input"].primary_frame().copy()
        canonical = vrb._load_canonical_live_context(
            raw, instrument=instrument, timeframe=timeframe
        )
        frame = canonical.copy() if canonical is not None else vrb._build_context(raw)
        return NodeOutput(
            payload={
                "source": "canonical_live" if canonical is not None else "raw_rebuild"
            },
            frames={"frame": frame},
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(
                context_compute,
                vrb._load_canonical_live_context,
                vrb._build_context,
            ),
            replay_policy=ReplayPolicyContract(mode="bounded_replay", replay_bars=400),
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=context_compute,
        )
    )

    rung_names: list[str] = []
    step8e_a_names: list[str] = []
    step8e_b_names: list[str] = []
    for phase, extra in (("step8e_a", {}), ("step8e_b", vrb.STEP8E_B_RETUNE_PARAMS)):
        ladder = vrb._build_recovery_ladder()
        for label, overrides in ladder:
            node_name = f"range_rung_debug__{phase}__{label}"
            params = {**vrb.BASE_RECOVERY_PARAMS, **overrides, **extra}
            rung_names.append(node_name)
            if phase == "step8e_a":
                step8e_a_names.append(node_name)
            else:
                step8e_b_names.append(node_name)
            assessment_label = f"{phase}/{label}"

            def rung_compute(
                context: GraphRunContext,
                deps: dict[str, NodeExecutionResult],
                params: dict[str, Any] = params,
                assessment_label: str = assessment_label,
                phase: str = phase,
            ) -> NodeOutput:
                if (
                    phase == "step8e_b"
                    and not deps["range_retune_gate"].output.payload["retune_needed"]
                ):
                    return NodeOutput(
                        payload={
                            "params": params,
                            "summary": {},
                            "label": assessment_label,
                            "skipped": True,
                        },
                        frames={
                            "event_table": pd.DataFrame(),
                            "candidate_table": pd.DataFrame(),
                        },
                        profile_details={
                            "label": assessment_label,
                            "skipped": True,
                            "substage_seconds": {
                                "debug_collect": 0.0,
                                "pressure_imbalance_legacy": 0.0,
                                "pressure_imbalance_v2": 0.0,
                                "contract_scores": 0.0,
                                "summary_build": 0.0,
                            },
                        },
                    )
                result = vrb._run_debug_with_params(
                    deps["range_context"].primary_frame().copy(),
                    params,
                )
                return NodeOutput(
                    payload={
                        "params": params,
                        "summary": result["summary"],
                        "label": assessment_label,
                        "skipped": False,
                    },
                    frames={
                        "event_table": result["event_table"],
                        "candidate_table": result["candidate_table"],
                    },
                    profile_details={
                        "label": assessment_label,
                        "skipped": False,
                        **result.get("profile_details", {}),
                    },
                )

            nodes.append(
                NodeManifest(
                    graph_name=graph_name,
                    node_name=node_name,
                    node_kind="compute",
                    semantic_class="B",
                    inputs=(),
                    upstream_nodes=(
                        ("range_context",)
                        if phase == "step8e_a"
                        else ("range_context", "range_retune_gate")
                    ),
                    output_artifacts=("event_table", "candidate_table"),
                    fingerprint_fn=no_runtime_config_fingerprint,
                    cache_policy=materialized_bundle_policy,
                    config=_source_hash_config(
                        rung_compute, vrb._run_debug_with_params
                    ),
                    replay_policy=ReplayPolicyContract(
                        mode="bounded_replay", replay_bars=400
                    ),
                    validation_policy=ValidationPolicy(level="node_parity"),
                    mutable_scope=MutableScope(scope="frontier_only"),
                    compute_fn=rung_compute,
                )
            )

    def retune_gate_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        assessments: list[dict[str, Any]] = []
        params_by_label: dict[str, dict[str, Any]] = {}
        for name in step8e_a_names:
            dep = deps[name]
            label = dep.output.payload["label"]
            summary = dep.output.payload["summary"]
            event_table = dep.output.frames["event_table"]
            assessment = vrb._assess_rung(
                label, {"summary": summary, "event_table": event_table}
            )
            assessments.append(assessment)
            params_by_label[label] = dep.output.payload["params"]
        best, has_valid = vrb._select_best_assessment(assessments)
        return NodeOutput(
            payload={
                "step8e_a_assessments": assessments,
                "step8e_a_best": best,
                "step8e_a_has_valid_rung": has_valid,
                "retune_needed": not has_valid,
                "params_by_label": params_by_label,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_retune_gate",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=tuple(step8e_a_names),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config=_source_hash_config(
                retune_gate_compute,
                vrb._assess_rung,
                vrb._select_best_assessment,
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=retune_gate_compute,
        )
    )

    def select_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        gate = deps["range_retune_gate"].output.payload
        assessments: list[dict[str, Any]] = list(gate["step8e_a_assessments"])
        params_by_label: dict[str, dict[str, Any]] = dict(gate["params_by_label"])
        has_valid = bool(gate["step8e_a_has_valid_rung"])
        best = dict(gate["step8e_a_best"])
        used_retune = not has_valid
        if used_retune:
            retune_assessments: list[dict[str, Any]] = []
            for name in step8e_b_names:
                dep = deps[name]
                if dep.output.payload.get("skipped"):
                    continue
                label = dep.output.payload["label"]
                summary = dep.output.payload["summary"]
                event_table = dep.output.frames["event_table"]
                assessment = vrb._assess_rung(
                    label, {"summary": summary, "event_table": event_table}
                )
                retune_assessments.append(assessment)
                params_by_label[label] = dep.output.payload["params"]
            if retune_assessments:
                assessments.extend(retune_assessments)
                best, has_valid = vrb._select_best_assessment(assessments)
        selected_label = str(best["label"])
        return NodeOutput(
            payload={
                "rung_assessments": assessments,
                "selected_label": (
                    selected_label if has_valid else "no_valid_contract_rung"
                ),
                "reporting_label": selected_label,
                "selected_params": params_by_label[selected_label],
                "has_valid_rung": has_valid,
                "used_retune": used_retune,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_selected_rung",
            node_kind="selection",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_retune_gate",) + tuple(step8e_b_names),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config=_source_hash_config(
                select_compute,
                vrb._assess_rung,
                vrb._select_best_assessment,
            ),
            validation_policy=ValidationPolicy(level="graph_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=select_compute,
        )
    )

    def selection_bundle_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        return NodeOutput(
            payload={"selected_rung": deps["range_selected_rung"].output.payload}
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_selection_bundle",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_selected_rung",),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=selection_bundle_compute,
        )
    )

    def selected_debug_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        selected_params = deps["range_selected_rung"].output.payload["selected_params"]
        result = vrb._run_debug_with_params(
            deps["range_context"].primary_frame().copy(),
            selected_params,
        )
        return NodeOutput(
            payload={"params": selected_params, "summary": result["summary"]},
            frames={
                "frame": result["frame"],
                "event_table": result["event_table"],
                "candidate_table": result["candidate_table"],
            },
            profile_details={
                "label": "selected_debug",
                "skipped": False,
                **result.get("profile_details", {}),
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_selected_debug",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_context", "range_selected_rung"),
            output_artifacts=("frame", "event_table", "candidate_table"),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(
                selected_debug_compute, vrb._run_debug_with_params
            ),
            replay_policy=ReplayPolicyContract(mode="bounded_replay", replay_bars=400),
            validation_policy=ValidationPolicy(level="graph_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=selected_debug_compute,
        )
    )

    def forensics_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        forensics, short_high, long_medium = vrb._build_forensics_tables(event_table)
        scored = vrb._add_path_c2_candidate_scores(forensics)
        tagged = vrb._assign_contract_bucket_labels(scored)
        return NodeOutput(
            frames={
                "forensics": tagged,
                "short_high": short_high,
                "long_medium": long_medium,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_forensics",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_selected_debug",),
            output_artifacts=("forensics",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config=_source_hash_config(
                forensics_compute,
                vrb._build_forensics_tables,
                vrb._add_path_c2_candidate_scores,
                vrb._assign_contract_bucket_labels,
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=forensics_compute,
        )
    )

    def geometry_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        candidate_table = deps["range_selected_debug"].output.frames["candidate_table"]
        geometry = vrb._build_geometry_audit(full_df, event_table, candidate_table)
        geometry_candidate_comparison, geometry_candidate_summary = (
            vrb._build_geometry_candidate_comparison(
                full_df,
                geometry,
            )
        )
        return NodeOutput(
            frames={
                "geometry_audit": geometry,
                "geometry_candidate_comparison": geometry_candidate_comparison,
                "geometry_candidate_summary": geometry_candidate_summary,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_geometry_audit",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_selected_debug",),
            output_artifacts=(
                "geometry_audit",
                "geometry_candidate_comparison",
                "geometry_candidate_summary",
            ),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config=_source_hash_config(
                geometry_compute,
                vrb._build_geometry_audit,
                vrb._build_geometry_candidate_comparison,
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=geometry_compute,
        )
    )

    def active_truth_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        full_df = deps["range_selected_debug"].output.frames["frame"]
        geometry = deps["range_geometry_audit"].output.frames["geometry_audit"]
        active_truth, doctrine = vrb._build_active_truth_audit(full_df, geometry)
        geometry_candidate_truth = vrb._build_geometry_candidate_active_truth_summary(
            full_df,
            geometry,
            active_truth,
        )
        return NodeOutput(
            frames={
                "active_truth_audit": active_truth,
                "doctrine_report": doctrine,
                "geometry_candidate_truth_summary": geometry_candidate_truth,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_active_truth_audit",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_selected_debug", "range_geometry_audit"),
            output_artifacts=(
                "active_truth_audit",
                "doctrine_report",
                "geometry_candidate_truth_summary",
            ),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config=_source_hash_config(
                active_truth_compute,
                vrb._build_active_truth_audit,
                vrb._build_geometry_candidate_active_truth_summary,
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=active_truth_compute,
        )
    )

    def coverage_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        rung_results = []
        assessments = deps["range_selected_rung"].output.payload["rung_assessments"]
        assessment_map = {str(item["label"]): item for item in assessments}
        for name in rung_names:
            dep = deps[name]
            label = dep.output.payload["label"]
            rung_results.append(
                (
                    label,
                    {
                        "summary": dep.output.payload["summary"],
                        "event_table": dep.output.frames["event_table"],
                    },
                )
            )
        coverage = vrb._build_coverage_regime_report(
            rung_results, list(assessment_map.values())
        )
        return NodeOutput(frames={"coverage_regime_report": coverage})

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_coverage_regime_report",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=tuple(rung_names) + ("range_selected_rung",),
            output_artifacts=("coverage_regime_report",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config=_source_hash_config(
                coverage_compute, vrb._build_coverage_regime_report
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=coverage_compute,
        )
    )

    def ranking_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        forensics = deps["range_forensics"].output.frames["forensics"]
        repair_gates, recommendation = vrb._evaluate_path_c2_candidates(forensics)
        geometry_candidate_comparison = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_comparison"
        ]
        geometry_ranking_preservation = vrb._build_geometry_ranking_preservation_report(
            forensics,
            geometry_candidate_comparison,
        )
        return NodeOutput(
            payload={"ranking_repair_recommendation": recommendation},
            frames={
                "ranking_report": vrb._build_ranking_disagreement_report(forensics),
                "ranking_rebase_report": vrb._build_ranking_rebase_comparison_report(
                    forensics
                ),
                "ranking_repair_report": vrb._build_path_c2_candidate_report(forensics),
                "ranking_repair_gates": repair_gates,
                "agreement_report": vrb._build_agreement_matrix(forensics),
                "bucket_lift_report": vrb._build_bucket_lift_report(forensics),
                "family_report": vrb._build_family_comparison_report(forensics),
                "path_c2_archetype_report": vrb._build_path_c2_archetype_report(
                    forensics
                ),
                "geometry_ranking_preservation": geometry_ranking_preservation,
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_ranking_bundle",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_forensics", "range_geometry_audit"),
            output_artifacts=("ranking_bundle",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config=_source_hash_config(
                ranking_compute,
                vrb._evaluate_path_c2_candidates,
                vrb._build_geometry_ranking_preservation_report,
                vrb._build_ranking_disagreement_report,
                vrb._build_ranking_rebase_comparison_report,
                vrb._build_path_c2_candidate_report,
                vrb._build_agreement_matrix,
                vrb._build_bucket_lift_report,
                vrb._build_family_comparison_report,
                vrb._build_path_c2_archetype_report,
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=ranking_compute,
        )
    )

    def diagnostics_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        forensics = deps["range_forensics"].output.frames["forensics"]
        short_high = deps["range_forensics"].output.frames["short_high"]
        long_medium = deps["range_forensics"].output.frames["long_medium"]
        geometry_audit = deps["range_geometry_audit"].output.frames["geometry_audit"]
        geometry_candidate_comparison = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_comparison"
        ]
        geometry_candidate_summary = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_summary"
        ]
        active_truth_audit = deps["range_active_truth_audit"].output.frames[
            "active_truth_audit"
        ]
        doctrine_report = deps["range_active_truth_audit"].output.frames[
            "doctrine_report"
        ]
        geometry_candidate_truth_summary = deps[
            "range_active_truth_audit"
        ].output.frames["geometry_candidate_truth_summary"]
        ranking_report = deps["range_ranking_bundle"].output.frames["ranking_report"]
        geometry_ranking_preservation = deps["range_ranking_bundle"].output.frames[
            "geometry_ranking_preservation"
        ]
        downstream_summary = deps["range_downstream_usefulness"].output.payload[
            "summary"
        ]
        geometry_candidate_downstream_summary = deps[
            "range_downstream_usefulness"
        ].output.frames["geometry_candidate_downstream_summary"]
        path_summary = vrb._primary_path_from_reports(
            active_truth_audit,
            doctrine_report,
            ranking_report,
            downstream_summary,
            geometry_audit,
        )
        geometry_candidate_gate_report, geometry_candidate_recommendation = (
            vrb._build_geometry_candidate_gate_report(
                geometry_candidate_summary,
                geometry_candidate_truth_summary,
                geometry_candidate_downstream_summary,
                geometry_ranking_preservation,
            )
        )

        recommended_improvement_ids: list[Any] = []
        if (
            geometry_candidate_recommendation != "no_candidate_passed"
            and not geometry_candidate_comparison.empty
        ):
            recommended_rows = geometry_candidate_comparison[
                geometry_candidate_comparison["candidate_family"]
                == geometry_candidate_recommendation
            ][["range_id", "geometry_chart_fit_score"]].copy()
            legacy_rows = geometry_candidate_comparison[
                geometry_candidate_comparison["candidate_family"] == "g1_legacy"
            ][["range_id", "geometry_chart_fit_score"]].copy()
            legacy_rows.rename(
                columns={"geometry_chart_fit_score": "legacy_geometry_chart_fit_score"},
                inplace=True,
            )
            recommended_improvement_ids = (
                recommended_rows.merge(legacy_rows, on="range_id", how="left")
                .assign(
                    improvement_score=(
                        pd.to_numeric(
                            recommended_rows["geometry_chart_fit_score"],
                            errors="coerce",
                        )
                        - pd.to_numeric(
                            recommended_rows.merge(
                                legacy_rows, on="range_id", how="left"
                            )["legacy_geometry_chart_fit_score"],
                            errors="coerce",
                        ).fillna(0.0)
                    )
                )
                .sort_values("improvement_score", ascending=False)["range_id"]
                .head(4)
                .tolist()
            )
        geometry_samples = geometry_audit[
            pd.to_numeric(geometry_audit["range_id"], errors="coerce").isin(
                recommended_improvement_ids
            )
        ].copy()
        if geometry_samples.empty:
            geometry_samples = (
                pd.concat(
                    [
                        forensics.nlargest(3, "rb_plausibility_score"),
                        forensics[
                            forensics["contract_bucket"] == "strong_false_positive"
                        ].head(3),
                    ],
                    ignore_index=True,
                )
                .drop_duplicates(subset=["range_id"])
                .head(6)
            )
        refresh_ids: list[Any] = []
        if not active_truth_audit.empty:
            refresh_rank = active_truth_audit.assign(
                refresh_gain=(
                    pd.to_numeric(
                        active_truth_audit["bounded_refresh_visual_plausibility_score"],
                        errors="coerce",
                    )
                    - pd.to_numeric(
                        active_truth_audit["frozen_visual_plausibility_score"],
                        errors="coerce",
                    )
                )
            ).sort_values("refresh_gain", ascending=False)
            refresh_ids = refresh_rank["range_id"].head(6).tolist()
        refresh_samples = geometry_audit[
            pd.to_numeric(geometry_audit["range_id"], errors="coerce").isin(refresh_ids)
        ].copy()
        downstream_usefulness = deps["range_downstream_usefulness"].output.frames[
            "downstream_usefulness"
        ]
        helpful = downstream_usefulness[
            pd.to_numeric(
                downstream_usefulness.get("interpretive_value_flag"), errors="coerce"
            )
            .fillna(0)
            .eq(1)
        ].nlargest(2, "rb_plausibility_score")
        little = downstream_usefulness[
            pd.to_numeric(
                downstream_usefulness.get("interpretive_value_flag"), errors="coerce"
            )
            .fillna(0)
            .eq(0)
        ].head(2)
        misled = downstream_usefulness[
            downstream_usefulness.get("contract_bucket", pd.Series(dtype="object")).eq(
                "strong_false_positive"
            )
        ].head(2)
        downstream_samples = (
            pd.concat([helpful, little, misled], ignore_index=True)
            .drop_duplicates(subset=["range_id"])
            .head(6)
        )

        if not forensics.empty:
            shortest_df = forensics.nsmallest(20, "duration_bars")
            longest_df = forensics.nlargest(20, "duration_bars")
            strongest_df = forensics.nlargest(20, "strength")
            weakest_df = forensics.nsmallest(20, "strength")
            ranging_short_lived_df = (
                forensics[
                    pd.to_numeric(forensics["confirm_regime"], errors="coerce").eq(0)
                    & pd.to_numeric(forensics["duration_bars"], errors="coerce").le(2)
                ]
                .sort_values(["duration_bars", "strength"], ascending=[True, False])
                .head(20)
            )
            plausibility_df = forensics.nlargest(20, "rb_plausibility_score")
            monitor_df = forensics.nlargest(20, "rb_monitor_worthiness_score")
            micro_box_df = forensics.nlargest(20, "rb_micro_box_risk_score")
            late_fragility_df = forensics.nlargest(
                20, "rb_late_confirm_fragility_score"
            )
        else:
            shortest_df = pd.DataFrame()
            longest_df = pd.DataFrame()
            strongest_df = pd.DataFrame()
            weakest_df = pd.DataFrame()
            ranging_short_lived_df = pd.DataFrame()
            plausibility_df = pd.DataFrame()
            monitor_df = pd.DataFrame()
            micro_box_df = pd.DataFrame()
            late_fragility_df = pd.DataFrame()

        forensics_bundle = vrb._bundle_named_reports(
            {
                "shortest_lived": shortest_df,
                "longest_lived": longest_df,
                "strongest": strongest_df,
                "weakest": weakest_df,
                "ranging_short_lived": ranging_short_lived_df,
                "short_lived_high_strength": short_high,
                "long_lived_medium_strength": long_medium,
                "highest_plausibility": plausibility_df,
                "highest_monitor_worthiness": monitor_df,
                "highest_micro_box_risk": micro_box_df,
                "highest_late_confirm_fragility": late_fragility_df,
            }
        )
        ranking_bundle = vrb._bundle_named_reports(
            {
                "ranking_disagreement": deps["range_ranking_bundle"].output.frames[
                    "ranking_report"
                ],
                "ranking_rebase": deps["range_ranking_bundle"].output.frames[
                    "ranking_rebase_report"
                ],
                "ranking_repair_candidates": deps["range_ranking_bundle"].output.frames[
                    "ranking_repair_report"
                ],
                "ranking_repair_gates": deps["range_ranking_bundle"].output.frames[
                    "ranking_repair_gates"
                ],
                "ranking_agreement": deps["range_ranking_bundle"].output.frames[
                    "agreement_report"
                ],
                "bucket_lift": deps["range_ranking_bundle"].output.frames[
                    "bucket_lift_report"
                ],
                "family_comparison": deps["range_ranking_bundle"].output.frames[
                    "family_report"
                ],
                "path_c2_archetypes": deps["range_ranking_bundle"].output.frames[
                    "path_c2_archetype_report"
                ],
                "geometry_ranking_preservation": deps[
                    "range_ranking_bundle"
                ].output.frames["geometry_ranking_preservation"],
                "geometry_candidate_gate_report": geometry_candidate_gate_report,
                "geometry_candidate_summary": geometry_candidate_summary,
                "geometry_candidate_truth_summary": geometry_candidate_truth_summary,
                "geometry_candidate_downstream_summary": geometry_candidate_downstream_summary,
            }
        )
        return NodeOutput(
            payload={
                "interpretability_summary": vrb._build_interpretability_metrics_summary(
                    forensics
                ),
                "contract_bucket_summary": vrb._build_contract_bucket_summary(
                    forensics
                ),
                "archetype_summary": vrb._build_archetype_summary(
                    short_high, long_medium
                ),
                "alignment_audit": vrb._build_viability_alignment_audit(
                    short_high, long_medium
                ),
                "pressure_audit": vrb._build_pressure_alignment_audit(
                    short_high, long_medium
                ),
                "path_summary": path_summary,
                "geometry_candidate_recommendation": geometry_candidate_recommendation,
            },
            frames={
                "geometry_samples": geometry_samples,
                "refresh_samples": refresh_samples,
                "downstream_samples": downstream_samples,
                "forensics_bundle": forensics_bundle,
                "ranking_bundle": ranking_bundle,
                "geometry_candidate_comparison": geometry_candidate_comparison,
                "geometry_candidate_summary": geometry_candidate_summary,
                "geometry_candidate_gate_report": geometry_candidate_gate_report,
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_diagnostics_bundle",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=(
                "range_forensics",
                "range_geometry_audit",
                "range_active_truth_audit",
                "range_ranking_bundle",
                "range_downstream_usefulness",
            ),
            output_artifacts=(
                "payload",
                "geometry_samples",
                "refresh_samples",
                "downstream_samples",
                "forensics_bundle",
                "ranking_bundle",
                "geometry_candidate_comparison",
                "geometry_candidate_summary",
                "geometry_candidate_gate_report",
            ),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config=_source_hash_config(
                diagnostics_compute,
                vrb._primary_path_from_reports,
                vrb._build_geometry_candidate_gate_report,
                vrb._build_interpretability_metrics_summary,
                vrb._build_contract_bucket_summary,
                vrb._build_archetype_summary,
                vrb._build_viability_alignment_audit,
                vrb._build_pressure_alignment_audit,
                vrb._bundle_named_reports,
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=diagnostics_compute,
        )
    )

    def downstream_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        full_df = deps["range_selected_debug"].output.frames["frame"]
        forensics = deps["range_forensics"].output.frames["forensics"]
        downstream_usefulness, summary = vrb._build_downstream_usefulness_report(
            full_df, forensics
        )
        geometry_audit = deps["range_geometry_audit"].output.frames["geometry_audit"]
        geometry_candidate_downstream_summary = (
            vrb._build_geometry_candidate_downstream_summary(full_df, geometry_audit)
        )
        return NodeOutput(
            payload={"summary": summary},
            frames={
                "downstream_usefulness": downstream_usefulness,
                "geometry_candidate_downstream_summary": geometry_candidate_downstream_summary,
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_downstream_usefulness",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_forensics",
                "range_geometry_audit",
            ),
            output_artifacts=(
                "downstream_usefulness",
                "geometry_candidate_downstream_summary",
            ),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config=_source_hash_config(
                downstream_compute,
                vrb._build_downstream_usefulness_report,
                vrb._build_geometry_candidate_downstream_summary,
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=downstream_compute,
        )
    )

    def analysis_bundle_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        return NodeOutput(
            payload={
                "selected_summary": deps["range_selected_debug"].output.payload[
                    "summary"
                ],
                "downstream_summary": deps[
                    "range_downstream_usefulness"
                ].output.payload["summary"],
                "diagnostics": deps["range_diagnostics_bundle"].output.payload,
                "ranking_repair_recommendation": deps[
                    "range_ranking_bundle"
                ].output.payload["ranking_repair_recommendation"],
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_analysis_bundle",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_geometry_audit",
                "range_active_truth_audit",
                "range_coverage_regime_report",
                "range_ranking_bundle",
                "range_downstream_usefulness",
                "range_diagnostics_bundle",
            ),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=analysis_bundle_compute,
        )
    )

    def chart_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("html", False):
            return NodeOutput(
                payload={
                    "html_path": None,
                    "summary": deps["range_selected_debug"].output.payload["summary"],
                }
            )
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        html_path = (
            out_dir / f"range_boundaries_validation_{instrument}_{timeframe}.html"
        )
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        candidate_table = deps["range_selected_debug"].output.frames["candidate_table"]
        selected = deps["range_selected_rung"].output.payload["reporting_label"]
        plot_rows = int(context.config.get("plot_rows", 300))
        date_from = pd.Timestamp(
            context.config.get("date_from", "2026-01-01"), tz="UTC"
        )
        plot_df = full_df[full_df["timestamp"] >= date_from].copy()
        if plot_df.empty:
            plot_df = full_df.tail(plot_rows).copy()
        elif not context.config.get("full", False):
            plot_df = plot_df.tail(plot_rows).copy()
        result = validate_range_boundaries(
            plot_df,
            outpath=html_path,
            title=f"Range Boundary Validation — {instrument} {timeframe} [{selected}]",
            summary_df=full_df,
            event_table=event_table,
            candidate_table=candidate_table,
        )
        return NodeOutput(
            payload={
                "html_path": str(result["html_path"]),
                "summary": result["summary"],
            },
            artifacts={"html": result["html_path"]},
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_main_chart",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("range_selected_debug", "range_selected_rung"),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config=_source_hash_config(chart_compute, validate_range_boundaries),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=chart_compute,
        )
    )

    def geometry_chart_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("html", False):
            return NodeOutput(payload={"html_path": None})
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        html_path = (
            out_dir
            / f"range_boundaries_geometry_chart_pack_{instrument}_{timeframe}.html"
        )
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        sample_df = deps["range_diagnostics_bundle"].output.frames["geometry_samples"]
        written = vrb._plot_audit_chart_pack(
            full_df,
            sample_df,
            event_table=event_table,
            outpath=html_path,
            title=f"Range Boundary Geometry Audit — {instrument} {timeframe}",
            active_truth_audit=deps["range_active_truth_audit"].output.frames[
                "active_truth_audit"
            ],
            show_geometry_candidates=True,
            highlight_candidate_family=deps[
                "range_diagnostics_bundle"
            ].output.payload.get("geometry_candidate_recommendation"),
        )
        return NodeOutput(
            payload={"html_path": str(written)}, artifacts={"html": written}
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_geometry_chart_pack",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_diagnostics_bundle",
                "range_active_truth_audit",
            ),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config=_source_hash_config(
                geometry_chart_compute, vrb._plot_audit_chart_pack
            ),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=geometry_chart_compute,
        )
    )

    def refresh_chart_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("html", False):
            return NodeOutput(payload={"html_path": None})
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        html_path = (
            out_dir
            / f"range_boundaries_refresh_chart_pack_{instrument}_{timeframe}.html"
        )
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        geometry_samples = deps["range_diagnostics_bundle"].output.frames[
            "geometry_samples"
        ]
        refresh_samples = deps["range_diagnostics_bundle"].output.frames[
            "refresh_samples"
        ]
        sample_df = (
            refresh_samples if not refresh_samples.empty else geometry_samples.head(4)
        )
        written = vrb._plot_audit_chart_pack(
            full_df,
            sample_df,
            event_table=event_table,
            outpath=html_path,
            title=f"Range Boundary Frozen vs Refresh Audit — {instrument} {timeframe}",
            active_truth_audit=deps["range_active_truth_audit"].output.frames[
                "active_truth_audit"
            ],
        )
        return NodeOutput(
            payload={"html_path": str(written)}, artifacts={"html": written}
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_refresh_chart_pack",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_diagnostics_bundle",
                "range_active_truth_audit",
            ),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config=_source_hash_config(
                refresh_chart_compute, vrb._plot_audit_chart_pack
            ),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=refresh_chart_compute,
        )
    )

    def downstream_chart_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("html", False):
            return NodeOutput(payload={"html_path": None})
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        html_path = (
            out_dir
            / f"range_boundaries_downstream_chart_pack_{instrument}_{timeframe}.html"
        )
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        geometry_samples = deps["range_diagnostics_bundle"].output.frames[
            "geometry_samples"
        ]
        downstream_samples = deps["range_diagnostics_bundle"].output.frames[
            "downstream_samples"
        ]
        sample_df = (
            downstream_samples
            if not downstream_samples.empty
            else geometry_samples.head(4)
        )
        written = vrb._plot_audit_chart_pack(
            full_df,
            sample_df,
            event_table=event_table,
            outpath=html_path,
            title=f"Range Boundary Downstream Operator Pack — {instrument} {timeframe}",
            active_truth_audit=deps["range_active_truth_audit"].output.frames[
                "active_truth_audit"
            ],
        )
        return NodeOutput(
            payload={"html_path": str(written)}, artifacts={"html": written}
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_downstream_chart_pack",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_diagnostics_bundle",
                "range_active_truth_audit",
            ),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config=_source_hash_config(
                downstream_chart_compute, vrb._plot_audit_chart_pack
            ),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=downstream_chart_compute,
        )
    )

    def chart_bundle_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        artifacts: dict[str, Any] = {}
        for node_name in (
            "range_main_chart",
            "range_geometry_chart_pack",
            "range_refresh_chart_pack",
            "range_downstream_chart_pack",
        ):
            artifacts[node_name] = {
                **deps[node_name].output.payload,
                "cache_hit": deps[node_name].cache_hit,
            }
        return NodeOutput(payload={"artifacts": artifacts})

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_chart_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_main_chart",
                "range_geometry_chart_pack",
                "range_refresh_chart_pack",
                "range_downstream_chart_pack",
            ),
            output_artifacts=("payload",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="report"),
            compute_fn=chart_bundle_compute,
        )
    )

    def csv_bundle_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("write_csv", False):
            return NodeOutput(payload={"artifact_paths": {}})
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        candidate_table = deps["range_selected_debug"].output.frames["candidate_table"]
        geometry_audit = deps["range_geometry_audit"].output.frames["geometry_audit"]
        geometry_candidate_comparison = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_comparison"
        ]
        geometry_candidate_summary = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_summary"
        ]
        active_truth_audit = deps["range_active_truth_audit"].output.frames[
            "active_truth_audit"
        ]
        doctrine_report = deps["range_active_truth_audit"].output.frames[
            "doctrine_report"
        ]
        geometry_candidate_truth_summary = deps[
            "range_active_truth_audit"
        ].output.frames["geometry_candidate_truth_summary"]
        coverage_regime_report = deps["range_coverage_regime_report"].output.frames[
            "coverage_regime_report"
        ]
        forensics_bundle = deps["range_diagnostics_bundle"].output.frames[
            "forensics_bundle"
        ]
        ranking_bundle = deps["range_diagnostics_bundle"].output.frames[
            "ranking_bundle"
        ]
        geometry_candidate_gate_report = deps["range_diagnostics_bundle"].output.frames[
            "geometry_candidate_gate_report"
        ]
        downstream_usefulness = deps["range_downstream_usefulness"].output.frames[
            "downstream_usefulness"
        ]
        geometry_candidate_downstream_summary = deps[
            "range_downstream_usefulness"
        ].output.frames["geometry_candidate_downstream_summary"]
        reporting_label = deps["range_selected_rung"].output.payload["reporting_label"]
        selected_label = deps["range_selected_rung"].output.payload["selected_label"]
        best_assessment = next(
            item
            for item in deps["range_selected_rung"].output.payload["rung_assessments"]
            if str(item["label"]) == str(reporting_label)
        )
        diagnosis_text = vrb._build_diagnosis_memo_text(
            geometry_audit=geometry_audit,
            active_truth_audit=active_truth_audit,
            doctrine_report=doctrine_report,
            coverage_regime_report=coverage_regime_report,
            ranking_report=deps["range_ranking_bundle"].output.frames["ranking_report"],
            family_report=deps["range_ranking_bundle"].output.frames["family_report"],
            downstream_summary=deps["range_downstream_usefulness"].output.payload[
                "summary"
            ],
            path_summary=deps["range_diagnostics_bundle"].output.payload[
                "path_summary"
            ],
            selected_label=reporting_label,
        )
        ranking_text = "\n".join(
            [
                "# Step 8F Path C Ranking Rebase",
                "",
                "## Score Architecture",
                "- `range_strength_structure`: retained structure-side quality block.",
                "- `range_strength_monitorability`: monitor-worthiness proxy rebased toward interpretability truth.",
                "- `range_strength_semantic`: semantic keep-watching block favoring plausible low-risk ranges.",
                "- `range_strength_viability`: rebased visible viability score.",
                "- `range_strength`: rebased production ranking score.",
                "",
                "## Mapping Doctrine",
                "- Positive: plausibility-adjacent monitorability, boundary relevance, low pressure imbalance, stable two-sided structure.",
                "- Negative: micro-box risk proxy built from low confirm latency, minimal touches, narrow/weak width, and hyper-fresh tidy behavior.",
                "- Legacy formation/viability remain exported as `_legacy` comparison fields.",
                "",
                "## Acceptance Snapshot",
                f"- Selected/reporting rung: `{reporting_label}`",
                f"- Contract-selected rung: `{selected_label}`",
                f"- Long-lived vs short-lived final strength gate passed: `{bool(best_assessment.get('strength_not_badly_inverted'))}`",
                f"- Coverage stable: confirmed=`{best_assessment.get('confirmed_ranges')}`, active_rows=`{best_assessment.get('active_rows')}`",
            ]
        )
        ranking_repair_text = "\n".join(
            [
                "# Step 8F.2 Path C2 Ranking Repair",
                "",
                f"- Recommendation: `{deps['range_ranking_bundle'].output.payload['ranking_repair_recommendation']}`",
                "",
                "## Candidate Gate Results",
                (
                    deps["range_ranking_bundle"]
                    .output.frames["ranking_repair_gates"]
                    .to_string(index=False)
                    if not deps["range_ranking_bundle"]
                    .output.frames["ranking_repair_gates"]
                    .empty
                    else "No gate results."
                ),
                "",
                "## Candidate Top-Rank Report",
                (
                    deps["range_ranking_bundle"]
                    .output.frames["ranking_repair_report"]
                    .to_string(index=False)
                    if not deps["range_ranking_bundle"]
                    .output.frames["ranking_repair_report"]
                    .empty
                    else "No repair report."
                ),
                "",
                "## Candidate Archetype Report",
                (
                    deps["range_ranking_bundle"]
                    .output.frames["path_c2_archetype_report"]
                    .to_string(index=False)
                    if not deps["range_ranking_bundle"]
                    .output.frames["path_c2_archetype_report"]
                    .empty
                    else "No archetype report."
                ),
            ]
        )
        geometry_candidate_bundle = vrb._bundle_named_reports(
            {
                "geometry_candidate_comparison": geometry_candidate_comparison,
                "geometry_candidate_summary": geometry_candidate_summary,
                "geometry_candidate_truth_summary": geometry_candidate_truth_summary,
                "geometry_candidate_downstream_summary": geometry_candidate_downstream_summary,
                "geometry_ranking_preservation": deps[
                    "range_ranking_bundle"
                ].output.frames["geometry_ranking_preservation"],
                "geometry_candidate_gate_report": geometry_candidate_gate_report,
            },
            group_col="geometry_report_group",
        )
        artifacts = {
            "events": out_dir / f"range_boundaries_events_{instrument}_{timeframe}.csv",
            "candidates": out_dir
            / f"range_boundaries_candidates_{instrument}_{timeframe}.csv",
            "forensics_bundle": out_dir
            / f"range_boundaries_forensics_bundle_{instrument}_{timeframe}.csv",
            "geometry_audit": out_dir
            / f"range_boundaries_geometry_audit_{instrument}_{timeframe}.csv",
            "geometry_candidates": out_dir
            / f"range_boundaries_geometry_candidates_{instrument}_{timeframe}.csv",
            "active_truth": out_dir
            / f"range_boundaries_active_truth_audit_{instrument}_{timeframe}.csv",
            "doctrine": out_dir
            / f"range_boundaries_frozen_vs_refresh_{instrument}_{timeframe}.csv",
            "coverage": out_dir
            / f"range_boundaries_coverage_regimes_{instrument}_{timeframe}.csv",
            "ranking_bundle": out_dir
            / f"range_boundaries_ranking_bundle_{instrument}_{timeframe}.csv",
            "downstream": out_dir
            / f"range_boundaries_downstream_usefulness_{instrument}_{timeframe}.csv",
            "diagnosis_memo": out_dir
            / f"range_boundaries_step8x_diagnosis_{instrument}_{timeframe}.md",
            "ranking_memo": out_dir
            / f"range_boundaries_path_c_ranking_rebase_{instrument}_{timeframe}.md",
            "ranking_repair_memo": out_dir
            / f"range_boundaries_path_c2_ranking_repair_{instrument}_{timeframe}.md",
        }
        write_csv_atomic(event_table, artifacts["events"])
        write_csv_atomic(candidate_table, artifacts["candidates"])
        write_csv_atomic(forensics_bundle, artifacts["forensics_bundle"])
        write_csv_atomic(geometry_audit, artifacts["geometry_audit"])
        write_csv_atomic(geometry_candidate_bundle, artifacts["geometry_candidates"])
        write_csv_atomic(active_truth_audit, artifacts["active_truth"])
        write_csv_atomic(doctrine_report, artifacts["doctrine"])
        write_csv_atomic(coverage_regime_report, artifacts["coverage"])
        write_csv_atomic(ranking_bundle, artifacts["ranking_bundle"])
        write_csv_atomic(downstream_usefulness, artifacts["downstream"])
        write_text_atomic(diagnosis_text, artifacts["diagnosis_memo"])
        write_text_atomic(ranking_text, artifacts["ranking_memo"])
        write_text_atomic(ranking_repair_text, artifacts["ranking_repair_memo"])
        return NodeOutput(
            payload={
                "artifact_paths": {name: str(path) for name, path in artifacts.items()}
            },
            artifacts=artifacts,
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_csv_bundle",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_selected_rung",
                "range_geometry_audit",
                "range_active_truth_audit",
                "range_coverage_regime_report",
                "range_ranking_bundle",
                "range_downstream_usefulness",
                "range_diagnostics_bundle",
            ),
            output_artifacts=("artifacts",),
            fingerprint_fn=csv_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config=_source_hash_config(
                csv_bundle_compute,
                vrb._build_diagnosis_memo_text,
                vrb._bundle_named_reports,
                write_csv_atomic,
                write_text_atomic,
            ),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=csv_bundle_compute,
        )
    )

    def validation_bundle_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        return NodeOutput(
            payload={
                "selected_rung": deps["range_selection_bundle"].output.payload[
                    "selected_rung"
                ],
                "selected_summary": deps["range_analysis_bundle"].output.payload[
                    "selected_summary"
                ],
                "downstream_summary": deps["range_analysis_bundle"].output.payload[
                    "downstream_summary"
                ],
                "diagnostics": deps["range_analysis_bundle"].output.payload[
                    "diagnostics"
                ],
                "artifacts": {
                    **deps["range_chart_bundle"].output.payload["artifacts"],
                    "range_csv_bundle": {
                        **deps["range_csv_bundle"].output.payload,
                        "cache_hit": deps["range_csv_bundle"].cache_hit,
                    },
                },
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_validation_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selection_bundle",
                "range_analysis_bundle",
                "range_chart_bundle",
                "range_csv_bundle",
            ),
            output_artifacts=("payload",),
            fingerprint_fn=_runtime_config_fingerprint(
                "html", "write_csv", "date_from", "plot_rows", "full", "out_dir"
            ),
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=validation_bundle_compute,
        )
    )

    return GraphManifest(
        graph_name=graph_name,
        nodes=tuple(nodes),
        default_target="range_validation_bundle",
    )


def build_regime_validation_graph(*, instrument: str, timeframe: str) -> GraphManifest:
    from scripts import validate_regime as vr

    graph_name = "validate_regime"
    nodes: list[NodeManifest] = [_source_node(graph_name, "raw_input", "raw_input")]
    no_runtime_config_fingerprint = _runtime_config_fingerprint()
    summary_runtime_config_fingerprint = _runtime_config_fingerprint(
        "plot_rows", "full"
    )
    chart_runtime_config_fingerprint = _runtime_config_fingerprint(
        "html", "plot_rows", "full", "out_dir"
    )
    materialized_frame_policy = _node_cache_policy(
        materialize=True, artifact_kind="frame"
    )
    materialized_report_policy = _node_cache_policy(
        materialize=True, artifact_kind="report"
    )
    ephemeral_policy = _node_cache_policy(materialize=False, artifact_kind="ephemeral")

    def make_context(include_research_only: bool, node_name: str) -> NodeManifest:
        return NodeManifest(
            graph_name=graph_name,
            node_name=node_name,
            node_kind="compute",
            semantic_class="C" if include_research_only else "B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(vr._load_canonical_context, vr._build_context),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps, include_research_only=include_research_only: NodeOutput(
                frames={
                    "frame": (
                        canonical.copy()
                        if (
                            canonical := vr._load_canonical_context(
                                deps["raw_input"].primary_frame().copy(),
                                instrument=instrument,
                                timeframe=timeframe,
                                dataset="research" if include_research_only else "live",
                            )
                        )
                        is not None
                        else vr._build_context(
                            deps["raw_input"].primary_frame().copy(),
                            include_research_only=include_research_only,
                        )
                    )
                }
            ),
        )

    nodes.extend(
        [
            make_context(False, "regime_live_context"),
            make_context(True, "regime_research_context"),
            NodeManifest(
                graph_name=graph_name,
                node_name="regime_summary",
                node_kind="aggregate",
                semantic_class="C",
                inputs=(),
                upstream_nodes=("regime_live_context", "regime_research_context"),
                output_artifacts=("payload",),
                fingerprint_fn=summary_runtime_config_fingerprint,
                cache_policy=ephemeral_policy,
                config=_source_hash_config(
                    validate_regime,
                    vr._synthetic_fixture_summary,
                ),
                validation_policy=ValidationPolicy(level="graph_parity"),
                compute_fn=lambda context, deps: NodeOutput(
                    payload=validate_regime(
                        (
                            deps["regime_research_context"].primary_frame()
                            if context.config.get("full", False)
                            else deps["regime_research_context"]
                            .primary_frame()
                            .tail(int(context.config.get("plot_rows", vr.PLOT_ROWS)))
                        ),
                        summary_df=deps["regime_research_context"].primary_frame(),
                        live_df=deps["regime_live_context"].primary_frame(),
                        research_df=deps["regime_research_context"].primary_frame(),
                        outpath=None,
                        title=f"Regime Validation — {instrument} {timeframe}",
                        synthetic_summary=vr._synthetic_fixture_summary(),
                    )
                ),
            ),
            NodeManifest(
                graph_name=graph_name,
                node_name="regime_main_chart",
                node_kind="report",
                semantic_class="C",
                inputs=(),
                upstream_nodes=(
                    "regime_summary",
                    "regime_live_context",
                    "regime_research_context",
                ),
                output_artifacts=("html",),
                fingerprint_fn=chart_runtime_config_fingerprint,
                cache_policy=materialized_report_policy,
                config=_source_hash_config(validate_regime),
                validation_policy=ValidationPolicy(level="report"),
                mutable_scope=MutableScope(scope="immutable"),
                compute_fn=lambda context, deps: (
                    NodeOutput(payload={"html_path": None})
                    if not context.config.get("html", False)
                    else NodeOutput(
                        payload={
                            "html_path": str(
                                validate_regime(
                                    (
                                        deps["regime_research_context"].primary_frame()
                                        if context.config.get("full", False)
                                        else deps["regime_research_context"]
                                        .primary_frame()
                                        .tail(
                                            int(
                                                context.config.get(
                                                    "plot_rows", vr.PLOT_ROWS
                                                )
                                            )
                                        )
                                    ),
                                    summary_df=deps[
                                        "regime_research_context"
                                    ].primary_frame(),
                                    live_df=deps["regime_live_context"].primary_frame(),
                                    research_df=deps[
                                        "regime_research_context"
                                    ].primary_frame(),
                                    outpath=Path(
                                        context.config.get(
                                            "out_dir", "notebooks/foundation"
                                        )
                                    )
                                    / f"regime_validation_{instrument}_{timeframe}.html",
                                    title=f"Regime Validation — {instrument} {timeframe}",
                                    synthetic_summary=vr._synthetic_fixture_summary(),
                                )["html_path"]
                            )
                        },
                        artifacts={
                            "html": Path(
                                context.config.get("out_dir", "notebooks/foundation")
                            )
                            / f"regime_validation_{instrument}_{timeframe}.html"
                        },
                    )
                ),
            ),
            NodeManifest(
                graph_name=graph_name,
                node_name="regime_validation_bundle",
                node_kind="aggregate",
                semantic_class="C",
                inputs=(),
                upstream_nodes=("regime_summary", "regime_main_chart"),
                output_artifacts=("payload",),
                fingerprint_fn=chart_runtime_config_fingerprint,
                cache_policy=ephemeral_policy,
                validation_policy=ValidationPolicy(level="graph_parity"),
                compute_fn=lambda context, deps: NodeOutput(
                    payload={
                        "summary": deps["regime_summary"].output.payload["summary"],
                        "html_path": deps["regime_main_chart"].output.payload.get(
                            "html_path"
                        ),
                    }
                ),
            ),
        ]
    )
    return GraphManifest(
        graph_name=graph_name,
        nodes=tuple(nodes),
        default_target="regime_validation_bundle",
    )


def build_trend_state_validation_graph(
    *, instrument: str, timeframe: str
) -> GraphManifest:
    from scripts import validate_trend_state as vt

    graph_name = "validate_trend_state"
    no_runtime_config_fingerprint = _runtime_config_fingerprint()
    summary_runtime_config_fingerprint = _runtime_config_fingerprint(
        "plot_rows", "full"
    )
    chart_runtime_config_fingerprint = _runtime_config_fingerprint(
        "html", "plot_rows", "full", "out_dir"
    )
    materialized_frame_policy = _node_cache_policy(
        materialize=True, artifact_kind="frame"
    )
    materialized_report_policy = _node_cache_policy(
        materialize=True, artifact_kind="report"
    )
    ephemeral_policy = _node_cache_policy(materialize=False, artifact_kind="ephemeral")
    nodes = [
        _source_node(graph_name, "raw_input", "raw_input"),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_minimal_overlay_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(
                vt._load_canonical_live_context, vt._build_trend_state_context
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": (
                        canonical.copy()
                        if (
                            canonical := vt._load_canonical_live_context(
                                deps["raw_input"].primary_frame().copy(),
                                instrument=instrument,
                                timeframe=timeframe,
                            )
                        )
                        is not None
                        else vt._build_trend_state_context(
                            deps["raw_input"].primary_frame().copy()
                        )
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(
                vt._load_canonical_live_context, vt._build_context
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": (
                        canonical.copy()
                        if (
                            canonical := vt._load_canonical_live_context(
                                deps["raw_input"].primary_frame().copy(),
                                instrument=instrument,
                                timeframe=timeframe,
                            )
                        )
                        is not None
                        else vt._build_context(deps["raw_input"].primary_frame().copy())
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_summary",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("trend_state_context",),
            output_artifacts=("payload",),
            fingerprint_fn=summary_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            config=_source_hash_config(validate_trend_state),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload=validate_trend_state(
                    (
                        deps["trend_state_context"].primary_frame()
                        if context.config.get("full", False)
                        else deps["trend_state_context"]
                        .primary_frame()
                        .tail(int(context.config.get("plot_rows", vt.PLOT_ROWS)))
                    ),
                    summary_df=deps["trend_state_context"].primary_frame(),
                    outpath=None,
                    title=f"Trend State Validation — {instrument} {timeframe}",
                    n_windows=5,
                )
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_main_chart",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("trend_state_summary", "trend_state_context"),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config=_source_hash_config(validate_trend_state),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=lambda context, deps: (
                NodeOutput(payload={"html_path": None})
                if not context.config.get("html", False)
                else NodeOutput(
                    payload={
                        "html_path": str(
                            validate_trend_state(
                                (
                                    deps["trend_state_context"].primary_frame()
                                    if context.config.get("full", False)
                                    else deps["trend_state_context"]
                                    .primary_frame()
                                    .tail(
                                        int(
                                            context.config.get(
                                                "plot_rows", vt.PLOT_ROWS
                                            )
                                        )
                                    )
                                ),
                                summary_df=deps["trend_state_context"].primary_frame(),
                                outpath=Path(
                                    context.config.get("out_dir", "notebooks/structure")
                                )
                                / f"trend_state_validation_{instrument}_{timeframe}.html",
                                title=f"Trend State Validation — {instrument} {timeframe}",
                                n_windows=5,
                            )["html_path"]
                        )
                    },
                    artifacts={
                        "html": Path(
                            context.config.get("out_dir", "notebooks/structure")
                        )
                        / f"trend_state_validation_{instrument}_{timeframe}.html"
                    },
                )
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_validation_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("trend_state_summary", "trend_state_main_chart"),
            output_artifacts=("payload",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": deps["trend_state_summary"].output.payload["summary"],
                    "html_path": deps["trend_state_main_chart"].output.payload.get(
                        "html_path"
                    ),
                }
            ),
        ),
    ]
    return GraphManifest(
        graph_name=graph_name,
        nodes=tuple(nodes),
        default_target="trend_state_validation_bundle",
    )


def build_sr_levels_validation_graph(
    *, instrument: str, timeframe: str
) -> GraphManifest:
    from scripts import validate_sr_levels as vsr
    from src.indicators._helpers.schema import normalize_candle_schema
    from src.indicators.foundation.session import add_session_features
    from src.indicators.foundation.sr_levels import (
        add_sr_research_columns,
        build_sr_level_registry,
        build_sr_touch_audit_table,
        deserialize_sr_registry,
        project_sr_context,
        serialize_sr_registry,
        update_sr_lifecycle,
    )
    from src.indicators.foundation.value import add_prev_day_hl, add_prev_week_hl
    from src.indicators.foundation.volatility import add_atr
    from src.indicators.foundation.volume_profile import add_volume_profile
    from src.indicators.smc.equal_hl import add_equal_hl
    from src.indicators.structure.swings import add_swings

    graph_name = "validate_sr_levels"
    no_runtime_config_fingerprint = _runtime_config_fingerprint()
    chart_runtime_config_fingerprint = _runtime_config_fingerprint(
        "html", "out_dir", "date_from", "plot_label"
    )
    materialized_frame_policy = _node_cache_policy(
        materialize=True, artifact_kind="frame"
    )
    materialized_report_policy = _node_cache_policy(
        materialize=True, artifact_kind="report"
    )
    ephemeral_policy = _node_cache_policy(materialize=False, artifact_kind="ephemeral")

    def _build_projected_output(deps: dict[str, Any]) -> NodeOutput:
        enriched = deps["sr_enriched_context"].primary_frame()
        registry = build_sr_level_registry(enriched)
        ctx = update_sr_lifecycle(enriched, registry)
        projected = pd.concat(
            [enriched.copy(), pd.DataFrame(ctx, index=enriched.index)], axis=1
        )
        audit = build_sr_touch_audit_table(projected, registry)
        registry_serialized = serialize_sr_registry(registry)
        return NodeOutput(
            frames={"frame": projected, "audit": audit},
            payload={"registry_serialized": registry_serialized},
        )

    def _registry_from_deps(deps: dict[str, Any]) -> dict:
        payload = deps["sr_projected_context"].output.payload
        return deserialize_sr_registry(payload["registry_serialized"])

    def _audit_from_deps(deps: dict[str, Any]) -> pd.DataFrame:
        return deps["sr_projected_context"].output.frames["audit"]

    nodes = [
        _source_node(graph_name, "raw_input", "raw_input"),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_enriched_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(
                normalize_candle_schema,
                add_atr,
                add_swings,
                add_equal_hl,
                add_prev_day_hl,
                add_prev_week_hl,
                add_session_features,
                add_volume_profile,
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": deps["raw_input"]
                    .primary_frame()
                    .pipe(normalize_candle_schema, require_volume=True)
                    .pipe(add_atr)
                    .pipe(add_swings)
                    .pipe(add_equal_hl)
                    .pipe(add_prev_day_hl)
                    .pipe(add_prev_week_hl)
                    .pipe(add_session_features)
                    .pipe(add_volume_profile)
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_projected_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("sr_enriched_context",),
            output_artifacts=("frame", "audit"),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(
                build_sr_level_registry,
                update_sr_lifecycle,
                project_sr_context,
                build_sr_touch_audit_table,
                serialize_sr_registry,
            ),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: _build_projected_output(deps),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_research_context",
            node_kind="compute",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("sr_projected_context",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(add_sr_research_columns),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": add_sr_research_columns(
                        deps["sr_projected_context"].primary_frame().copy(),
                        _registry_from_deps(deps),
                        touch_rows=_audit_from_deps(deps),
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_summary",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("sr_projected_context", "sr_research_context"),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            config=_source_hash_config(summarize_sr_levels),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": summarize_sr_levels(
                        deps["sr_research_context"].primary_frame(),
                        _registry_from_deps(deps),
                        live_df=deps["sr_projected_context"].primary_frame(),
                        touch_rows=_audit_from_deps(deps),
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_main_chart",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "sr_enriched_context",
                "sr_projected_context",
            ),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config={
                **_source_hash_config(
                    vsr._build_sr_chart,
                    vsr._is_structural_zone,
                    vsr._select_visible_zones,
                    vsr._zone_lifecycle_bounds,
                    vsr._add_zone_lifecycle_shapes,
                    vsr._add_zone_terminal_markers,
                    vsr._add_zone_touch_markers,
                    vsr._add_primary_zone_band,
                    vsr._zone_rect_color,
                    vsr._terminal_label,
                    save_figure_html,
                ),
                # Module-level tuning knobs that don't live in any function
                # body — embed them so changing the constant invalidates the
                # cached chart on the next run.
                "structural_score_min": float(vsr._STRUCTURAL_SCORE_MIN),
                "structural_min_touches": int(vsr._STRUCTURAL_MIN_TOUCHES),
                "structural_high_info_families": sorted(
                    vsr._STRUCTURAL_HIGH_INFO_FAMILIES
                ),
                "zone_render_budget_per_side": int(vsr._ZONE_RENDER_BUDGET_PER_SIDE),
            },
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=lambda context, deps: (
                NodeOutput(payload={"html_path": None})
                if not context.config.get("html", False)
                else NodeOutput(
                    payload={
                        "html_path": str(
                            save_figure_html(
                                vsr._build_sr_chart(
                                    deps["sr_enriched_context"].primary_frame(),
                                    deps["sr_projected_context"].primary_frame(),
                                    _registry_from_deps(deps),
                                    title=(
                                        f"S/R Levels — {instrument} {timeframe}  |  "
                                        f"{context.config.get('plot_label', vsr.DATE_FROM + ' -> end')}"
                                    ),
                                    date_from=context.config.get(
                                        "date_from", vsr.DATE_FROM
                                    ),
                                    audit=_audit_from_deps(deps),
                                ),
                                Path(
                                    context.config.get(
                                        "out_dir", "notebooks/foundation"
                                    )
                                )
                                / (
                                    f"sr_levels_validation_{instrument}_{timeframe}"
                                    f"{context.config.get('plot_suffix', '')}.html"
                                ),
                            )
                        )
                    },
                    artifacts={
                        "html": Path(
                            context.config.get("out_dir", "notebooks/foundation")
                        )
                        / (
                            f"sr_levels_validation_{instrument}_{timeframe}"
                            f"{context.config.get('plot_suffix', '')}.html"
                        )
                    },
                )
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_validation_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("sr_summary", "sr_main_chart", "sr_projected_context"),
            output_artifacts=("payload",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": deps["sr_summary"].output.payload["summary"],
                    "row_count": int(len(deps["sr_projected_context"].primary_frame())),
                    "html_path": deps["sr_main_chart"].output.payload.get("html_path"),
                }
            ),
        ),
    ]
    return GraphManifest(
        graph_name=graph_name, nodes=tuple(nodes), default_target="sr_validation_bundle"
    )


def build_structure_context_validation_graph(
    *, instrument: str, timeframe: str
) -> GraphManifest:
    from scripts import validate_structure_context as vsc
    from src.validation.indicators.structure_context import (
        plot_structure_context_validation,
        summarize_structure_context,
    )

    graph_name = "validate_structure_context"
    no_runtime_config_fingerprint = _runtime_config_fingerprint()
    summary_runtime_config_fingerprint = _runtime_config_fingerprint("plot_start")
    chart_runtime_config_fingerprint = _runtime_config_fingerprint(
        "html", "plot_start", "out_dir"
    )
    materialized_frame_policy = _node_cache_policy(
        materialize=True, artifact_kind="frame"
    )
    materialized_report_policy = _node_cache_policy(
        materialize=True, artifact_kind="report"
    )
    ephemeral_policy = _node_cache_policy(materialize=False, artifact_kind="ephemeral")

    def _plot_start(context: GraphRunContext) -> pd.Timestamp:
        value = context.config.get("plot_start", str(vsc.PLOT_START))
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    nodes = [
        _source_node(graph_name, "raw_input", "raw_input"),
        NodeManifest(
            graph_name=graph_name,
            node_name="structure_context_frame",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(vsc._build_context),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={"frame": vsc._build_context(deps["raw_input"].primary_frame())}
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="structure_context_view",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("structure_context_frame",),
            output_artifacts=("frame",),
            fingerprint_fn=summary_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": deps["structure_context_frame"]
                    .primary_frame()
                    .loc[
                        pd.to_datetime(
                            deps["structure_context_frame"].primary_frame()[
                                "timestamp"
                            ],
                            utc=True,
                            errors="coerce",
                        )
                        >= _plot_start(context)
                    ]
                    .copy()
                    .reset_index(drop=True)
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="structure_context_summary",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("structure_context_view",),
            output_artifacts=("payload",),
            fingerprint_fn=summary_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            config=_source_hash_config(summarize_structure_context),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": summarize_structure_context(
                        deps["structure_context_view"].primary_frame()
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="structure_context_chart",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("structure_context_view",),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config=_source_hash_config(plot_structure_context_validation),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=lambda context, deps: (
                NodeOutput(payload={"html_path": None})
                if not context.config.get("html", True)
                else NodeOutput(
                    payload={
                        "html_path": str(
                            plot_structure_context_validation(
                                deps["structure_context_view"].primary_frame(),
                                outpath=Path(context.config.get("out_dir", vsc.OUT_DIR))
                                / f"structure_context_{instrument}_{timeframe}.html",
                                title=(
                                    "Structure Context Validation "
                                    f"- {instrument} {timeframe}"
                                ),
                            )
                        )
                    },
                    artifacts={
                        "html": Path(context.config.get("out_dir", vsc.OUT_DIR))
                        / f"structure_context_{instrument}_{timeframe}.html"
                    },
                )
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="structure_context_validation_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("structure_context_summary", "structure_context_chart"),
            output_artifacts=("payload",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": deps["structure_context_summary"].output.payload[
                        "summary"
                    ],
                    "html_path": deps["structure_context_chart"].output.payload.get(
                        "html_path"
                    ),
                }
            ),
        ),
    ]
    return GraphManifest(
        graph_name=graph_name,
        nodes=tuple(nodes),
        default_target="structure_context_validation_bundle",
    )


def build_swings_validation_graph(*, instrument: str, timeframe: str) -> GraphManifest:
    from scripts import validate_swings as vsw
    from src.validation.indicators.swings import (
        plot_swings_validation,
        summarize_swings,
        swing_event_windows,
    )

    graph_name = "validate_swings"
    no_runtime_config_fingerprint = _runtime_config_fingerprint()
    summary_runtime_config_fingerprint = _runtime_config_fingerprint(
        "n_windows", "start_ts", "end_ts"
    )
    chart_runtime_config_fingerprint = _runtime_config_fingerprint(
        "html", "start_ts", "end_ts", "out_dir"
    )
    materialized_frame_policy = _node_cache_policy(
        materialize=True, artifact_kind="frame"
    )
    materialized_report_policy = _node_cache_policy(
        materialize=True, artifact_kind="report"
    )
    ephemeral_policy = _node_cache_policy(materialize=False, artifact_kind="ephemeral")
    nodes = [
        _source_node(graph_name, "raw_input", "raw_input"),
        NodeManifest(
            graph_name=graph_name,
            node_name="swings_context_frame",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config=_source_hash_config(vsw._build_context),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={"frame": vsw._build_context(deps["raw_input"].primary_frame())}
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="swings_summary",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("swings_context_frame",),
            output_artifacts=("payload",),
            fingerprint_fn=summary_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            config=_source_hash_config(summarize_swings, swing_event_windows),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": summarize_swings(
                        deps["swings_context_frame"].primary_frame()
                    ),
                    "high_windows": swing_event_windows(
                        deps["swings_context_frame"].primary_frame(),
                        side="high",
                        limit=int(context.config.get("n_windows", 3)),
                    ),
                    "low_windows": swing_event_windows(
                        deps["swings_context_frame"].primary_frame(),
                        side="low",
                        limit=int(context.config.get("n_windows", 3)),
                    ),
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="swings_chart",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("swings_context_frame",),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config=_source_hash_config(plot_swings_validation),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=lambda context, deps: (
                NodeOutput(payload={"html_path": None})
                if not context.config.get("html", True)
                else NodeOutput(
                    payload={
                        "html_path": str(
                            plot_swings_validation(
                                deps["swings_context_frame"].primary_frame(),
                                outpath=Path(context.config.get("out_dir", vsw.OUT_DIR))
                                / f"swings_validation_{instrument}_{timeframe}.html",
                                title=(f"Swings Validation - {instrument} {timeframe}"),
                                start_ts=context.config.get("start_ts", "2026-01-01"),
                                end_ts=context.config.get("end_ts", "2026-03-15"),
                            )
                        )
                    },
                    artifacts={
                        "html": Path(context.config.get("out_dir", vsw.OUT_DIR))
                        / f"swings_validation_{instrument}_{timeframe}.html"
                    },
                )
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="swings_validation_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("swings_summary", "swings_chart"),
            output_artifacts=("payload",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": deps["swings_summary"].output.payload["summary"],
                    "high_windows": deps["swings_summary"].output.payload[
                        "high_windows"
                    ],
                    "low_windows": deps["swings_summary"].output.payload["low_windows"],
                    "html_path": deps["swings_chart"].output.payload.get("html_path"),
                }
            ),
        ),
    ]
    return GraphManifest(
        graph_name=graph_name,
        nodes=tuple(nodes),
        default_target="swings_validation_bundle",
    )


def build_ob_validation_graph(*, instrument: str, timeframe: str) -> GraphManifest:
    from scripts import validate_ob as vob
    from src.validation.indicators.ob import (
        _plot_endpoint_inventory_validation_html,
        _plot_equivalence_casebook_overlay_html,
        _plot_monitorability_series_html,
        build_ob_diagnostic_package,
        ob_event_windows,
        plot_ob_validation,
        summarize_ob,
    )

    graph_name = "validate_ob"
    no_runtime_config_fingerprint = _runtime_config_fingerprint()
    artifact_runtime_config_fingerprint = _runtime_config_fingerprint("out_dir", "html")
    chart_runtime_config_fingerprint = _runtime_config_fingerprint(
        "html", "out_dir", "plot_start"
    )
    materialized_frame_policy = _node_cache_policy(
        materialize=True, artifact_kind="frame"
    )
    materialized_bundle_policy = _node_cache_policy(
        materialize=True, artifact_kind="bundle"
    )
    materialized_report_policy = _node_cache_policy(
        materialize=True, artifact_kind="report"
    )
    ephemeral_policy = _node_cache_policy(materialize=False, artifact_kind="ephemeral")

    def _ob_context_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        return NodeOutput(
            frames={
                "frame": vob._build_context(deps["raw_input"].primary_frame().copy())
            }
        )

    def _ob_summary_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        df = deps["ob_context"].primary_frame()
        diagnostics = build_ob_diagnostic_package(
            df,
            instrument=instrument,
            timeframe=timeframe,
        )
        bull_windows = ob_event_windows(df, side="bull", limit=5)
        bear_windows = ob_event_windows(df, side="bear", limit=5)
        frames = {
            "bos_coverage_audit": diagnostics["bos_coverage_audit"],
            "event_audit": diagnostics["event_audit"],
            "monitorability_timeseries": diagnostics["monitorability_timeseries"],
            "inventory_timeseries": diagnostics["inventory_timeseries"],
            "distance_band_audit": diagnostics["distance_band_audit"],
            "live_vs_nonlive_casebook": diagnostics["live_vs_nonlive_casebook"],
            "execution_comparison": diagnostics["execution_comparison"],
            "accepted": diagnostics["accepted"],
        }
        for idx, window in enumerate(bull_windows):
            frames[f"bull_window_{idx}"] = window
        for idx, window in enumerate(bear_windows):
            frames[f"bear_window_{idx}"] = window
        return NodeOutput(
            frames=frames,
            payload={
                "summary": diagnostics["summary"],
                "coverage_summary": diagnostics["coverage_summary"],
                "inventory_summary": diagnostics["inventory_summary"],
                "equivalence_summary": diagnostics["equivalence_summary"],
                "execution_summary": diagnostics["execution_summary"],
                "redundancy_summary": diagnostics["redundancy_summary"],
                "canonical_contract_memo": diagnostics["canonical_contract_memo"],
                "redundancy_diagnostic_memo": diagnostics["redundancy_diagnostic_memo"],
                "decision_memo": diagnostics["decision_memo"],
                "bull_window_count": len(bull_windows),
                "bear_window_count": len(bear_windows),
            },
        )

    def _ob_artifacts_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        out_dir = Path(context.config.get("out_dir", "notebooks/smc"))
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{instrument}_{timeframe}"
        summary_output = deps["ob_summary_bundle"].output
        context_df = deps["ob_context"].primary_frame()
        artifact_frames = {
            f"ob_bos_coverage_audit_{suffix}.csv": summary_output.frames[
                "bos_coverage_audit"
            ],
            f"ob_inventory_timeseries_{suffix}.csv": summary_output.frames[
                "inventory_timeseries"
            ],
            f"ob_monitorability_timeseries_{suffix}.csv": summary_output.frames[
                "monitorability_timeseries"
            ],
            f"ob_distance_band_audit_{suffix}.csv": summary_output.frames[
                "distance_band_audit"
            ],
            f"ob_live_vs_nonlive_casebook_{suffix}.csv": summary_output.frames[
                "live_vs_nonlive_casebook"
            ],
            f"ob_bos_vs_ob_execution_comparison_{suffix}.csv": summary_output.frames[
                "execution_comparison"
            ],
        }
        artifacts: dict[str, Path] = {}
        for filename, frame in artifact_frames.items():
            artifacts[filename] = write_csv_atomic(frame, out_dir / filename)
        contract_name = f"ob_canonical_contract_memo_{suffix}.md"
        artifacts[contract_name] = write_text_atomic(
            summary_output.payload["canonical_contract_memo"], out_dir / contract_name
        )
        redundancy_name = f"ob_redundancy_diagnostic_{suffix}.md"
        artifacts[redundancy_name] = write_text_atomic(
            summary_output.payload["redundancy_diagnostic_memo"],
            out_dir / redundancy_name,
        )
        decision_name = f"ob_canonical_decision_memo_{suffix}.md"
        artifacts[decision_name] = write_text_atomic(
            summary_output.payload["decision_memo"], out_dir / decision_name
        )
        if context.config.get("html", False):
            monitorability_html = _plot_monitorability_series_html(
                summary_output.frames["inventory_timeseries"],
                outpath=out_dir / f"ob_monitorability_timeseries_{suffix}.html",
                title=f"OB Monitorability Over Time — {instrument} {timeframe}",
                y_columns=[
                    "top_active_distance_atr",
                    "top_fresh_distance_atr",
                    "raw_active_count",
                    "raw_fresh_count",
                ],
            )
            artifacts[monitorability_html.name] = monitorability_html
            casebook_html = _plot_equivalence_casebook_overlay_html(
                context_df,
                reference=summary_output.frames["live_vs_nonlive_casebook"],
                accepted=summary_output.frames["accepted"],
                outpath=out_dir / f"ob_live_vs_nonlive_casebook_{suffix}.html",
                title=f"OB Live vs Nonlive Casebook — {instrument} {timeframe}",
            )
            artifacts[casebook_html.name] = casebook_html
            inventory_html = _plot_endpoint_inventory_validation_html(
                context_df,
                inventory_df=summary_output.frames["inventory_timeseries"],
                accepted=summary_output.frames["accepted"],
                outpath=out_dir / f"ob_validation_inventory_canonical_{suffix}.html",
                title=f"OB Validation Inventory Canonical — {instrument} {timeframe}",
            )
            artifacts[inventory_html.name] = inventory_html
        artifact_paths = {name: str(path) for name, path in artifacts.items()}
        return NodeOutput(
            payload={"artifact_paths": artifact_paths}, artifacts=artifacts
        )

    def _ob_chart_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("html", False):
            return NodeOutput(payload={"html_path": None})
        df = deps["ob_context"].primary_frame()
        plot_start = pd.Timestamp(
            context.config.get("plot_start", str(vob.PLOT_START)), tz="UTC"
        )
        plot_end = pd.to_datetime(df["timestamp"], utc=True).max()
        plot_df = (
            df[pd.to_datetime(df["timestamp"], utc=True) >= plot_start]
            .copy()
            .reset_index(drop=True)
        )
        if plot_df.empty:
            raise ValueError(
                f"No {instrument} {timeframe} rows found for validation window starting {plot_start}."
            )
        title = (
            f"Order Blocks Validation — {instrument} {timeframe} "
            f"(swing w={vob.SWING_WINDOW}, ret={vob.SWING_RETRACE}, confirm={vob.SWING_CONFIRM_BARS}) "
            f"({plot_start.date()} to {plot_end.date()})"
        )
        outpath = (
            Path(context.config.get("out_dir", "notebooks/smc"))
            / f"ob_validation_{instrument}_{timeframe}.html"
        )
        html_path = plot_ob_validation(plot_df, outpath=outpath, title=title)
        return NodeOutput(
            payload={"html_path": str(html_path)},
            artifacts={"html": outpath},
        )

    nodes = [
        _source_node(graph_name, "raw_input", "raw_input"),
        NodeManifest(
            graph_name=graph_name,
            node_name="ob_context",
            node_kind="compute",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_frame_policy,
            config={
                **_source_hash_config(
                    vob._build_context,
                    vob.normalize_candle_schema,
                    vob.add_atr,
                    vob.add_swings,
                    vob.add_trend_state,
                    vob.add_bos,
                    vob.add_choch,
                    vob.add_displacement_candle,
                    vob.add_ob,
                    vob.add_ob_mitigation,
                ),
                "swing_window": int(vob.SWING_WINDOW),
                "swing_retrace": float(vob.SWING_RETRACE),
                "swing_confirm_bars": int(vob.SWING_CONFIRM_BARS),
                "ob_validation_context_contract": 1,
            },
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=_ob_context_compute,
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="ob_summary_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("ob_context",),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=materialized_bundle_policy,
            config={
                **_source_hash_config(
                    summarize_ob,
                    ob_event_windows,
                    build_ob_diagnostic_package,
                ),
                "n_windows": 5,
                "ob_validation_summary_contract": 6,
            },
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=_ob_summary_compute,
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="ob_artifact_bundle",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("ob_context", "ob_summary_bundle"),
            output_artifacts=("artifacts",),
            fingerprint_fn=artifact_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config={
                **_source_hash_config(
                    write_csv_atomic,
                    write_text_atomic,
                    _plot_monitorability_series_html,
                    _plot_equivalence_casebook_overlay_html,
                    _plot_endpoint_inventory_validation_html,
                ),
                "ob_validation_artifact_contract": 6,
            },
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=_ob_artifacts_compute,
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="ob_main_chart",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("ob_context",),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=materialized_report_policy,
            config={
                **_source_hash_config(plot_ob_validation),
                "swing_window": int(vob.SWING_WINDOW),
                "swing_retrace": float(vob.SWING_RETRACE),
                "swing_confirm_bars": int(vob.SWING_CONFIRM_BARS),
                "ob_validation_chart_contract": 1,
            },
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=_ob_chart_compute,
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="ob_validation_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "ob_context",
                "ob_summary_bundle",
                "ob_artifact_bundle",
                "ob_main_chart",
            ),
            output_artifacts=("payload",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=ephemeral_policy,
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames=dict(deps["ob_summary_bundle"].output.frames),
                payload={
                    "summary": deps["ob_summary_bundle"].output.payload["summary"],
                    "coverage_summary": deps["ob_summary_bundle"].output.payload[
                        "coverage_summary"
                    ],
                    "inventory_summary": deps["ob_summary_bundle"].output.payload[
                        "inventory_summary"
                    ],
                    "equivalence_summary": deps["ob_summary_bundle"].output.payload[
                        "equivalence_summary"
                    ],
                    "execution_summary": deps["ob_summary_bundle"].output.payload[
                        "execution_summary"
                    ],
                    "redundancy_summary": deps["ob_summary_bundle"].output.payload[
                        "redundancy_summary"
                    ],
                    "bull_window_count": deps["ob_summary_bundle"].output.payload[
                        "bull_window_count"
                    ],
                    "bear_window_count": deps["ob_summary_bundle"].output.payload[
                        "bear_window_count"
                    ],
                    "artifact_paths": deps["ob_artifact_bundle"].output.payload[
                        "artifact_paths"
                    ],
                    "html_path": deps["ob_main_chart"].output.payload.get("html_path"),
                    "row_count": int(len(deps["ob_context"].primary_frame())),
                },
            ),
        ),
    ]
    return GraphManifest(
        graph_name=graph_name,
        nodes=tuple(nodes),
        default_target="ob_validation_bundle",
    )


def get_builtin_graph(graph_name: str, **kwargs: Any) -> GraphManifest:
    if graph_name == "live_pipeline":
        return build_live_stage_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            swing_window=int(kwargs.get("swing_window", 6)),
            include_vp=bool(kwargs.get("include_vp", True)),
            timeframe=kwargs.get("timeframe", "H4"),
            include_cross_asset=bool(kwargs.get("include_cross_asset", False)),
        )
    if graph_name == "research_pipeline":
        return build_research_stage_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            swing_window=int(kwargs.get("swing_window", 6)),
            include_vp=bool(kwargs.get("include_vp", True)),
            include_avwap=bool(kwargs.get("include_avwap", False)),
            timeframe=kwargs.get("timeframe", "H4"),
            include_cross_asset=bool(kwargs.get("include_cross_asset", False)),
        )
    if graph_name == "validate_range_boundaries":
        return build_range_boundaries_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    if graph_name == "validate_regime":
        return build_regime_validation_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    if graph_name == "validate_trend_state":
        return build_trend_state_validation_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    if graph_name == "validate_sr_levels":
        return build_sr_levels_validation_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    if graph_name == "validate_structure_context":
        return build_structure_context_validation_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    if graph_name == "validate_swings":
        return build_swings_validation_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    if graph_name == "validate_ob":
        return build_ob_validation_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    raise KeyError(f"Unknown builtin graph {graph_name!r}")
