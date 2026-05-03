"""
pipelines/build_research.py

Full indicator pipeline for research and backtesting.

Uses the same canonical causal swing engine as live so that research,
training, backtesting, and deployment share the same structural backbone.

For live deployment, use ``build_live.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.dag_runtime import execute_graph, GraphRunContext, GraphRunResult
from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.features.cross_asset import (
    SMT_PARTNERS,
    SUPPORTED_CROSS_ASSET_TIMEFRAMES,
    cross_asset_runtime_config_hash,
    persist_market_context,
)
from src.indicators.research import (
    summarize_cross_asset_correlation_audit,
    summarize_smt_research,
)
from src.pipeline_runtime import (
    ArtifactWriteResult,
    PipelineMetadata,
    PipelineRunProfiler,
    PipelineStage,
    ReplayPolicy,
    cleanup_temp_artifacts,
    dataframe_fingerprint,
    fingerprint_mapping,
    load_partitioned_dataset,
    merge_recomputed_frontier,
    metadata_path,
    persist_partitioned_dataset,
    read_metadata,
    resolve_incremental_plan,
    slice_frame_for_plan,
    write_json_atomic,
    write_parquet_atomic,
    write_metadata_atomic,
)

# --- Foundation ---
from src.indicators.foundation.ema import add_emas
from src.indicators.foundation.adx import add_adx
from src.indicators.foundation.momentum import add_rsi, add_macd, add_rsi_divergence
from src.indicators.foundation.volatility import (
    add_atr,
    add_bb_width,
    add_rolling_atr_ratio,
    add_body_ratio,
)
from src.indicators.foundation.volume import (
    add_volume_features,
)
from src.indicators.foundation.value import (
    add_avwap_from_last_swing,
    add_asian_session_hl,
    add_prev_day_hl,
    add_prev_week_hl,
    add_round_number_flag,
)
from src.indicators.foundation.volume_profile import add_volume_profile
from src.indicators.foundation.session import add_session_features
from src.indicators.foundation.regime import add_regime

# --- Structure ---
from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state
from src.indicators.structure.bos import add_bos
from src.indicators.structure.choch import add_choch

# --- SMC ---
from src.indicators.smc.fvg import collect_fvg_debug_tables
from src.indicators.smc.fvg_fill import add_fvg_fill
from src.indicators.smc.ifvg import add_ifvg
from src.indicators.smc.ob import add_ob
from src.indicators.smc.ob_mitigation import add_ob_mitigation
from src.indicators.smc.equal_hl import add_equal_hl
from src.indicators.smc.displacement import add_displacement_candle
from src.indicators.smc.amd import add_amd_engine

# --- Sweeps v2 (Steps 9-11): unified liquidity sources + final sweeps ---
from src.indicators.foundation.sr_levels import add_sr_levels
from src.indicators.smc.sweeps.unified_sources import add_unified_liquidity_sources
from src.indicators.smc.sweeps.final_sweeps import add_final_sweeps

RESEARCH_PIPELINE_NAME = "build_research"
RESEARCH_SCHEMA_VERSION = 1
# Bumped 3 → 4 when sr_levels / unified_liquidity_sources / final_sweeps
# stages were added.
# Bumped 4 → 5 for Step 11B repair: persistent breach tracking, eligibility
# filter against bar open (not close), penetration / wick-prominence /
# cooldown / family-strength gates, and corrected followthrough column
# linkage. Output ladder + sweeps schemas changed observably, so the bump
# invalidates pipeline-level caches.
RESEARCH_FEATURE_CONTRACT_VERSION = 5


@dataclass(slots=True)
class PipelineExecutionResult:
    frame: pd.DataFrame
    plan: Any
    profiler: PipelineRunProfiler
    metadata_updates: dict[str, Any]
    market_context: pd.DataFrame | None = None
    graph_result: GraphRunResult | None = None


@dataclass(slots=True)
class MaterializationResult:
    frame: pd.DataFrame
    plan: Any
    profiler: PipelineRunProfiler
    metadata: PipelineMetadata | None
    artifacts: list[ArtifactWriteResult]
    metadata_file: str | None


def _run_stage(
    frame: pd.DataFrame,
    stage: PipelineStage,
    *,
    profiler: PipelineRunProfiler | None,
) -> pd.DataFrame:
    started_at = time.perf_counter()
    result = stage.fn(frame)
    if profiler is not None:
        profiler.record_stage(
            stage.name,
            started_at=started_at,
            input_frame=frame,
            output_frame=result,
            details={
                "classification": stage.policy.classification,
                "replay_bars": stage.policy.replay_bars,
                "warmup_bars": stage.policy.warmup_bars,
                "carried_state": stage.policy.carried_state,
            },
        )
    return result


def _record_partitioned_dataset_load(
    profiler: PipelineRunProfiler | None,
    *,
    kind: str,
):
    if profiler is None:
        return None

    def _observer(path: Path, frame: pd.DataFrame) -> None:
        profiler.record_read(
            path=path,
            rows=len(frame),
            bytes_read=path.stat().st_size if path.exists() else None,
            kind=kind,
        )

    return _observer


def _run_fvg_stack(df: pd.DataFrame) -> pd.DataFrame:
    fvg_debug = collect_fvg_debug_tables(df)
    out = fvg_debug["frame"]
    out = add_fvg_fill(out, debug_tables=fvg_debug)
    return add_ifvg(out, debug_tables=fvg_debug)


def _add_intraday_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    if ts.diff().median().total_seconds() < 86400:
        out = add_asian_session_hl(out)
        out = add_session_features(out, include_research_only=True)
    return out


def _coerce_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype(float)
    return out


def _partition_frontier_ts(metadata: PipelineMetadata | None) -> pd.Timestamp | None:
    if metadata is None or metadata.last_processed_ts is None:
        return None
    ts = pd.Timestamp(metadata.last_processed_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return (
        ts.tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
        .start_time.tz_localize("UTC")
    )


def _effective_frontier_from_ts(
    partition_frontier_ts: pd.Timestamp | None,
    replay_from_ts: pd.Timestamp | None,
) -> pd.Timestamp | None:
    if replay_from_ts is None:
        return partition_frontier_ts
    if partition_frontier_ts is None or replay_from_ts < partition_frontier_ts:
        return replay_from_ts
    return partition_frontier_ts


def _cross_asset_partner_fingerprint(
    processed_cross_asset_frames: Mapping[str, pd.DataFrame] | None,
) -> str | None:
    if not processed_cross_asset_frames:
        return None
    rows = []
    for symbol in sorted(processed_cross_asset_frames):
        rows.append(
            {
                "symbol": symbol,
                "fingerprint": dataframe_fingerprint(
                    processed_cross_asset_frames[symbol]
                ),
            }
        )
    return dataframe_fingerprint(pd.DataFrame(rows), strategy="content")


def _cross_asset_peer_identity_payload(
    *,
    instrument: str,
    timeframe: str,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None,
    raw_data_root: str | Path | None,
) -> dict[str, object]:
    from src.indicators.features.cross_asset import CONTEXT_SYMBOLS

    payload: dict[str, object] = {}
    provided = dict(peer_raw_frames or {})
    for symbol in CONTEXT_SYMBOLS:
        if symbol == instrument:
            continue
        frame = provided.get(symbol)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            payload[symbol] = {
                "source": "input",
                "fingerprint": dataframe_fingerprint(frame, strategy="content"),
            }
            continue
        if raw_data_root is None:
            payload[symbol] = {"source": "missing"}
            continue
        path = Path(raw_data_root) / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            payload[symbol] = {"source": "missing"}
            continue
        stat = path.stat()
        payload[symbol] = {
            "source": "file",
            "path": str(path.resolve()),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return payload


def _partner_source_hash(symbol: str) -> str:
    from src.dag_runtime.builtin_graphs import _live_partner_source_hash

    return _live_partner_source_hash(symbol)


def _pregraph_cross_asset_fingerprints(
    *,
    normalized: pd.DataFrame,
    instrument: str,
    timeframe: str,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None,
    raw_data_root: str | Path | None,
) -> tuple[str, str]:
    peer_inputs = _cross_asset_peer_identity_payload(
        instrument=instrument,
        timeframe=timeframe,
        peer_raw_frames=peer_raw_frames,
        raw_data_root=raw_data_root,
    )
    market_context_fp = fingerprint_mapping(
        {
            "peer_inputs": peer_inputs,
            "cross_asset_config_hash": cross_asset_runtime_config_hash(
                timeframe=timeframe,
                relevant_pairs=None,
            ),
            "full_pair_matrix": True,
        }
    )
    partner_fp = fingerprint_mapping(
        {
            "partners": [
                {
                    "symbol": symbol,
                    "peer_input": peer_inputs.get(symbol),
                    "source_hash": _partner_source_hash(symbol),
                }
                for symbol, _relation in SMT_PARTNERS.get(instrument, ())
            ]
        }
    )
    return market_context_fp, partner_fp


def _execute_research_graph(
    df: pd.DataFrame,
    *,
    instrument: str,
    swing_window: int,
    include_vp: bool,
    include_avwap: bool,
    timeframe: str,
    include_cross_asset: bool,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None,
    raw_data_root: str | Path | None,
    force_graph_recompute: bool,
    features_root: str | Path = "data/features",
    target: str | None = None,
) -> GraphRunResult:
    out = _coerce_ohlc(df)
    from src.dag_runtime.builtin_graphs import build_research_stage_graph

    graph = build_research_stage_graph(
        instrument=instrument,
        swing_window=swing_window,
        include_vp=include_vp,
        include_avwap=include_avwap,
        timeframe=timeframe,
        include_cross_asset=include_cross_asset,
    )
    inputs: dict[str, Any] = {"raw_input": out}
    if peer_raw_frames is not None:
        inputs["peer_raw_frames"] = peer_raw_frames
    return execute_graph(
        graph,
        target=target,
        context=GraphRunContext(
            graph_name=graph.graph_name,
            symbol=instrument,
            timeframe=timeframe,
            inputs=inputs,
            config={
                "instrument": instrument,
                "timeframe": timeframe,
                "swing_window": swing_window,
                "include_vp": include_vp,
                "include_avwap": include_avwap,
                "include_cross_asset": include_cross_asset,
                "raw_data_root": (
                    str(raw_data_root) if raw_data_root is not None else None
                ),
            },
            features_root=features_root,
            cache_root="data/dag_cache",
            force=force_graph_recompute,
            invalidate_cache=force_graph_recompute,
        ),
    )


def _research_stages(
    *,
    instrument: str,
    swing_window: int,
    include_vp: bool,
    include_avwap: bool,
    timeframe: str = "H4",
) -> list[PipelineStage]:
    stages: list[PipelineStage] = [
        PipelineStage(
            "normalize_candles",
            lambda df: normalize_candle_schema(df, require_volume=True),
            ReplayPolicy("normalize_candles", "A", replay_bars=0),
        ),
        PipelineStage(
            "atr", add_atr, ReplayPolicy("atr", "A", replay_bars=150, warmup_bars=14)
        ),
        PipelineStage(
            "ema", add_emas, ReplayPolicy("ema", "A", replay_bars=220, warmup_bars=200)
        ),
        PipelineStage(
            "adx", add_adx, ReplayPolicy("adx", "A", replay_bars=150, warmup_bars=14)
        ),
        PipelineStage(
            "rsi", add_rsi, ReplayPolicy("rsi", "A", replay_bars=150, warmup_bars=14)
        ),
        PipelineStage(
            "macd", add_macd, ReplayPolicy("macd", "A", replay_bars=200, warmup_bars=26)
        ),
        PipelineStage(
            "bb_width",
            add_bb_width,
            ReplayPolicy("bb_width", "A", replay_bars=150, warmup_bars=20),
        ),
        PipelineStage(
            "body_ratio",
            add_body_ratio,
            ReplayPolicy("body_ratio", "A", replay_bars=16, warmup_bars=1),
        ),
        PipelineStage(
            "swings",
            lambda df: add_swings(df, window=swing_window),
            ReplayPolicy(
                "swings",
                "B",
                replay_bars=max(400, swing_window * 30),
                warmup_bars=swing_window,
                carried_state=True,
            ),
        ),
        PipelineStage(
            "trend_state",
            add_trend_state,
            ReplayPolicy("trend_state", "B", replay_bars=400, carried_state=True),
        ),
        PipelineStage(
            "bos",
            add_bos,
            ReplayPolicy("bos", "B", replay_bars=400, carried_state=True),
        ),
        PipelineStage(
            "choch",
            add_choch,
            ReplayPolicy("choch", "B", replay_bars=400, carried_state=True),
        ),
        PipelineStage(
            "rsi_divergence",
            add_rsi_divergence,
            ReplayPolicy("rsi_divergence", "B", replay_bars=200, warmup_bars=14),
        ),
        PipelineStage(
            "rolling_atr_ratio",
            add_rolling_atr_ratio,
            ReplayPolicy("rolling_atr_ratio", "A", replay_bars=150, warmup_bars=100),
        ),
        PipelineStage(
            "volume_features",
            lambda df: add_volume_features(df, include_research_only=True),
            ReplayPolicy("volume_features", "A", replay_bars=240, warmup_bars=100),
        ),
        PipelineStage(
            "fvg_stack",
            _run_fvg_stack,
            ReplayPolicy("fvg_stack", "B", replay_bars=240, carried_state=True),
        ),
        PipelineStage(
            "displacement",
            add_displacement_candle,
            ReplayPolicy("displacement", "B", replay_bars=240, carried_state=True),
        ),
        PipelineStage(
            "order_blocks",
            add_ob,
            ReplayPolicy("order_blocks", "B", replay_bars=240, carried_state=True),
        ),
        PipelineStage(
            "ob_mitigation",
            add_ob_mitigation,
            ReplayPolicy("ob_mitigation", "B", replay_bars=240, carried_state=True),
        ),
        PipelineStage(
            "equal_hl",
            add_equal_hl,
            ReplayPolicy("equal_hl", "B", replay_bars=240, carried_state=True),
        ),
        PipelineStage(
            "amd_engine",
            lambda df: add_amd_engine(df, add_labels=False),
            ReplayPolicy("amd_engine", "B", replay_bars=240, carried_state=True),
        ),
        PipelineStage(
            "prev_day_hl",
            add_prev_day_hl,
            ReplayPolicy("prev_day_hl", "A", replay_bars=72, warmup_bars=24),
        ),
        PipelineStage(
            "prev_week_hl",
            add_prev_week_hl,
            ReplayPolicy("prev_week_hl", "A", replay_bars=240, warmup_bars=120),
        ),
        PipelineStage(
            "round_number_flag",
            lambda df: add_round_number_flag(df, instrument=instrument),
            ReplayPolicy("round_number_flag", "A", replay_bars=1, warmup_bars=1),
        ),
        PipelineStage(
            "intraday_context",
            _add_intraday_context,
            ReplayPolicy("intraday_context", "A", replay_bars=240, warmup_bars=24),
        ),
    ]
    if include_vp:
        stages.append(
            PipelineStage(
                "volume_profile",
                add_volume_profile,
                ReplayPolicy("volume_profile", "A", replay_bars=140, warmup_bars=80),
            )
        )
    if include_avwap:
        stages.append(
            PipelineStage(
                "anchored_vwap",
                lambda df: add_avwap_from_last_swing(df, include_research_only=True),
                ReplayPolicy("anchored_vwap", "B", replay_bars=200, carried_state=True),
            )
        )
    stages.append(
        PipelineStage(
            "regime",
            lambda df: add_regime(df, include_research_only=True),
            ReplayPolicy("regime", "C", replay_bars=240, carried_state=True),
        )
    )

    # ── Canonical sweep system ──────────────────────────────────────────
    # Run S/R level extraction → unified liquidity sources → final sweeps
    # *after* every source-producing stage so the unified framework sees
    # the latest live-safe view of every family.
    stages.append(
        PipelineStage(
            "sr_levels",
            lambda df: add_sr_levels(df, include_research_only=False),
            ReplayPolicy("sr_levels", "B", replay_bars=400, carried_state=True),
        )
    )
    stages.append(
        PipelineStage(
            "unified_liquidity_sources",
            lambda df: add_unified_liquidity_sources(
                df, scan_timeframe=timeframe, instrument=instrument
            ),
            ReplayPolicy(
                "unified_liquidity_sources",
                "B",
                replay_bars=240,
                carried_state=False,
            ),
        )
    )
    stages.append(
        PipelineStage(
            "final_sweeps",
            add_final_sweeps,
            ReplayPolicy("final_sweeps", "B", replay_bars=240, carried_state=True),
        )
    )
    return stages


def _smt_partner_stages(
    *,
    swing_window: int,
) -> list[PipelineStage]:
    """Return only the stages needed by SMT partners (through swings).

    SMT divergence reads: timestamp, OHLC, and swing_* columns.
    These are all produced by the first 9 stages of the research chain.
    Running the remaining 16+ stages (trend_state through regime) on
    partner frames is pure waste.
    """
    return [
        PipelineStage(
            "normalize_candles",
            lambda df: normalize_candle_schema(df, require_volume=True),
            ReplayPolicy("normalize_candles", "A", replay_bars=0),
        ),
        PipelineStage(
            "atr", add_atr, ReplayPolicy("atr", "A", replay_bars=150, warmup_bars=14)
        ),
        PipelineStage(
            "ema", add_emas, ReplayPolicy("ema", "A", replay_bars=220, warmup_bars=200)
        ),
        PipelineStage(
            "adx", add_adx, ReplayPolicy("adx", "A", replay_bars=150, warmup_bars=14)
        ),
        PipelineStage(
            "rsi", add_rsi, ReplayPolicy("rsi", "A", replay_bars=150, warmup_bars=14)
        ),
        PipelineStage(
            "macd",
            add_macd,
            ReplayPolicy("macd", "A", replay_bars=200, warmup_bars=26),
        ),
        PipelineStage(
            "bb_width",
            add_bb_width,
            ReplayPolicy("bb_width", "A", replay_bars=150, warmup_bars=20),
        ),
        PipelineStage(
            "body_ratio",
            add_body_ratio,
            ReplayPolicy("body_ratio", "A", replay_bars=16, warmup_bars=1),
        ),
        PipelineStage(
            "swings",
            lambda df: add_swings(df, window=swing_window),
            ReplayPolicy(
                "swings",
                "B",
                replay_bars=max(400, swing_window * 30),
                warmup_bars=swing_window,
                carried_state=True,
            ),
        ),
    ]


def build_smt_partner_indicators(
    df: pd.DataFrame,
    *,
    swing_window: int = 6,
) -> pd.DataFrame:
    """Build only the indicators needed for SMT partner frames.

    Runs the first 9 stages (normalize through swings) and stops.
    This is ~3x faster than running the full 25-stage chain since
    the downstream structural/SMC stages are never read by
    ``add_smt_divergence``.
    """
    out = _coerce_ohlc(df)
    for stage in _smt_partner_stages(swing_window=swing_window):
        out = stage.fn(out)
    return out


def build_research_indicators(
    df: pd.DataFrame,
    instrument: str = "XAU_USD",
    swing_window: int = 6,
    include_vp: bool = True,
    include_avwap: bool = False,
    profiler: PipelineRunProfiler | None = None,
    timeframe: str = "H4",
    include_cross_asset: bool = False,
    market_context: pd.DataFrame | None = None,
    processed_cross_asset_frames: Mapping[str, pd.DataFrame] | None = None,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None = None,
    raw_data_root: str | Path | None = "data/raw",
    force_graph_recompute: bool = False,
) -> pd.DataFrame:
    """Apply the full indicator stack in dependency order.

    Parameters
    ----------
    df : DataFrame
        Raw candle data with columns: timestamp, open, high, low, close,
        and either canonical ``volume`` or raw ``tickVolume``.
    instrument : str
        For round-number detection (XAU_USD or USOIL).
    swing_window : int
        Lookback window for canonical causal swing detection.
    include_vp : bool
        Whether to compute Volume Profile.
    include_avwap : bool
        Whether to compute Anchored VWAP from last swing.

    Returns
    -------
    DataFrame with all indicator columns added.
    """
    if include_cross_asset and timeframe not in SUPPORTED_CROSS_ASSET_TIMEFRAMES:
        raise ValueError(
            f"Cross-asset context only supports {sorted(SUPPORTED_CROSS_ASSET_TIMEFRAMES)}"
        )
    graph_result = _execute_research_graph(
        df,
        instrument=instrument,
        swing_window=swing_window,
        include_vp=include_vp,
        include_avwap=include_avwap,
        timeframe=timeframe,
        include_cross_asset=include_cross_asset,
        peer_raw_frames=peer_raw_frames,
        raw_data_root=raw_data_root,
        force_graph_recompute=force_graph_recompute,
        target=("research_feature_bundle" if include_cross_asset else None),
    )
    if profiler is not None and include_cross_asset:
        profiler.set_metric("cross_asset_off_graph_seconds", 0.0)
        profiler.set_metric("research_cross_asset_dag", True)
    return graph_result.primary_frame()


def run_research_pipeline(
    df: pd.DataFrame,
    *,
    instrument: str = "XAU_USD",
    timeframe: str = "H4",
    swing_window: int = 6,
    include_vp: bool = True,
    include_avwap: bool = False,
    existing_history: pd.DataFrame | None = None,
    metadata: Any = None,
    config: dict[str, Any] | None = None,
    force_rebuild: bool = False,
    profiler: PipelineRunProfiler | None = None,
    replay_bars_override: int | None = None,
    include_cross_asset: bool = False,
    market_context: pd.DataFrame | None = None,
    processed_cross_asset_frames: Mapping[str, pd.DataFrame] | None = None,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None = None,
    raw_data_root: str | Path | None = "data/raw",
    force_graph_recompute: bool = False,
) -> PipelineExecutionResult:
    runtime_profiler = profiler or PipelineRunProfiler(
        pipeline=RESEARCH_PIPELINE_NAME,
        symbol=instrument,
        timeframe=timeframe,
    )
    normalized = normalize_candle_schema(df, require_volume=True)
    input_fingerprint = dataframe_fingerprint(normalized)
    resolved_market_context: pd.DataFrame | None = None
    market_context_fingerprint: str | None = None
    partner_fingerprint: str | None = None
    build_cross_asset_audit = bool((config or {}).get("build_cross_asset_audit", False))
    if include_cross_asset:
        if timeframe not in SUPPORTED_CROSS_ASSET_TIMEFRAMES:
            raise ValueError(
                "Cross-asset context only supports "
                f"{sorted(SUPPORTED_CROSS_ASSET_TIMEFRAMES)}"
            )
        market_context_fingerprint, partner_fingerprint = (
            _pregraph_cross_asset_fingerprints(
                normalized=normalized,
                instrument=instrument,
                timeframe=timeframe,
                peer_raw_frames=peer_raw_frames,
                raw_data_root=raw_data_root,
            )
        )
        runtime_profiler.set_metric("cross_asset_off_graph_seconds", 0.0)
        runtime_profiler.set_metric("research_cross_asset_dag", True)
    extra_config = {
        key: value for key, value in (config or {}).items() if key != "features_root"
    }
    config_payload = {
        "instrument": instrument,
        "timeframe": timeframe,
        "swing_window": swing_window,
        "include_vp": include_vp,
        "include_avwap": include_avwap,
        "include_cross_asset": include_cross_asset,
        "build_cross_asset_audit": build_cross_asset_audit,
        "cross_asset_market_context_fp": market_context_fingerprint,
        "cross_asset_partner_fp": partner_fingerprint,
        **extra_config,
    }
    config_fingerprint = dataframe_fingerprint(
        pd.DataFrame([config_payload]), strategy="content"
    )
    runtime_profiler.set_metric("workload_class", "incremental_or_full_research")
    persist_from_ts = _partition_frontier_ts(metadata)
    max_replay_bars = (
        replay_bars_override
        if replay_bars_override is not None
        else max(
            stage.policy.replay_bars
            for stage in _research_stages(
                instrument=instrument,
                swing_window=swing_window,
                include_vp=include_vp,
                include_avwap=include_avwap,
                timeframe=timeframe,
            )
        )
    )
    plan = resolve_incremental_plan(
        normalized,
        metadata=metadata,
        schema_version=RESEARCH_SCHEMA_VERSION,
        feature_contract_version=RESEARCH_FEATURE_CONTRACT_VERSION,
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        replay_bars=max_replay_bars,
        force_rebuild=bool(force_rebuild or force_graph_recompute),
    )
    if plan.is_noop:
        return PipelineExecutionResult(
            frame=(
                existing_history.copy()
                if existing_history is not None
                else normalized.iloc[0:0].copy()
            ),
            plan=plan,
            profiler=runtime_profiler,
            metadata_updates={
                "input_fingerprint": input_fingerprint,
                "config_fingerprint": config_fingerprint,
            },
            market_context=resolved_market_context,
        )

    working = slice_frame_for_plan(normalized, plan)
    graph_result = _execute_research_graph(
        working,
        instrument=instrument,
        swing_window=swing_window,
        include_vp=include_vp,
        include_avwap=include_avwap,
        timeframe=timeframe,
        include_cross_asset=include_cross_asset,
        peer_raw_frames=peer_raw_frames,
        raw_data_root=raw_data_root,
        force_graph_recompute=force_graph_recompute,
        features_root=(config or {}).get("features_root", "data/features"),
        target=(
            "research_full_bundle"
            if include_cross_asset and build_cross_asset_audit
            else ("research_feature_bundle" if include_cross_asset else None)
        ),
    )
    computed = graph_result.primary_frame()
    if include_cross_asset:
        market_context_result = graph_result.node_results.get(
            "research_market_context_source"
        )
        if market_context_result is not None:
            resolved_market_context = market_context_result.primary_frame()
            runtime_profiler.set_metric(
                "market_context_cache_current",
                bool(market_context_result.cache_hit),
            )
    if existing_history is not None and plan.mode == "incremental":
        frontier_from_ts = _effective_frontier_from_ts(
            persist_from_ts,
            plan.replay_from_ts,
        )
        final_frame = merge_recomputed_frontier(
            existing_history,
            computed,
            frontier_from_ts=frontier_from_ts,
        )
    else:
        final_frame = computed
    last_ts = pd.to_datetime(final_frame["timestamp"], utc=True, errors="coerce").iloc[
        -1
    ]
    return PipelineExecutionResult(
        frame=final_frame,
        plan=plan,
        profiler=runtime_profiler,
        metadata_updates={
            "last_processed_ts": last_ts.isoformat(),
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "feature_contract_version": RESEARCH_FEATURE_CONTRACT_VERSION,
            "input_fingerprint": input_fingerprint,
            "config_fingerprint": config_fingerprint,
            "engine_version": "research-v1",
            "persist_from_ts": (
                _effective_frontier_from_ts(
                    persist_from_ts,
                    plan.replay_from_ts,
                ).isoformat()
                if _effective_frontier_from_ts(
                    persist_from_ts,
                    plan.replay_from_ts,
                )
                is not None
                else None
            ),
        },
        market_context=resolved_market_context,
        graph_result=graph_result,
    )


def materialize_research_features(
    raw_df: pd.DataFrame,
    *,
    instrument: str = "XAU_USD",
    timeframe: str = "H4",
    swing_window: int = 6,
    include_vp: bool = True,
    include_avwap: bool = False,
    features_root: str = "data/features",
    state_root: str | None = None,
    config: dict[str, Any] | None = None,
    force_rebuild: bool = False,
    profiler: PipelineRunProfiler | None = None,
    partition_writer=None,
    include_cross_asset: bool = False,
    market_context: pd.DataFrame | None = None,
    processed_cross_asset_frames: Mapping[str, pd.DataFrame] | None = None,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None = None,
    raw_data_root: str | Path | None = "data/raw",
    force_graph_recompute: bool = False,
    build_cross_asset_audit: bool = True,
) -> MaterializationResult:
    runtime_profiler = profiler or PipelineRunProfiler(
        pipeline=f"{RESEARCH_PIPELINE_NAME}_materialize",
        symbol=instrument,
        timeframe=timeframe,
    )
    cleanup_temp_artifacts(features_root)
    metadata_file = metadata_path(
        state_root or Path(features_root) / "_state",
        pipeline=RESEARCH_PIPELINE_NAME,
        symbol=instrument,
        timeframe=timeframe,
    )
    metadata = read_metadata(metadata_file)
    existing_history = None
    if metadata is not None and not force_rebuild:
        existing_history = load_partitioned_dataset(
            features_root,
            dataset="research",
            symbol=instrument,
            timeframe=timeframe,
            read_observer=_record_partitioned_dataset_load(
                runtime_profiler,
                kind="load-existing-research-history",
            ),
        )
    runtime_config = {
        "features_root": str(features_root),
        "build_cross_asset_audit": build_cross_asset_audit,
        **(config or {}),
    }
    result = run_research_pipeline(
        raw_df,
        instrument=instrument,
        timeframe=timeframe,
        swing_window=swing_window,
        include_vp=include_vp,
        include_avwap=include_avwap,
        existing_history=existing_history,
        metadata=metadata,
        config=runtime_config,
        force_rebuild=force_rebuild,
        profiler=runtime_profiler,
        replay_bars_override=(
            len(raw_df) if metadata is not None and not force_rebuild else None
        ),
        include_cross_asset=include_cross_asset,
        market_context=market_context,
        processed_cross_asset_frames=processed_cross_asset_frames,
        peer_raw_frames=peer_raw_frames,
        raw_data_root=raw_data_root,
        force_graph_recompute=force_graph_recompute,
    )
    if result.plan.is_noop:
        return MaterializationResult(
            frame=(
                existing_history.copy()
                if existing_history is not None
                else result.frame
            ),
            plan=result.plan,
            profiler=result.profiler,
            metadata=metadata,
            artifacts=[],
            metadata_file=str(metadata_file),
        )

    persist_kwargs: dict[str, Any] = {}
    if partition_writer is not None:
        persist_kwargs["writer"] = partition_writer
    artifacts = persist_partitioned_dataset(
        result.frame,
        base_dir=features_root,
        dataset="research",
        symbol=instrument,
        timeframe=timeframe,
        frontier_from_ts=(
            pd.Timestamp(result.metadata_updates["persist_from_ts"])
            if result.metadata_updates.get("persist_from_ts")
            else result.plan.replay_from_ts
        ),
        full_rebuild=bool(force_rebuild or result.plan.mode == "full"),
        **persist_kwargs,
    )
    for artifact in artifacts:
        result.profiler.record_artifact(
            path=artifact.path,
            rows=artifact.rows,
            bytes_written=artifact.bytes_written,
            kind="canonical-research",
        )
    if include_cross_asset and result.market_context is not None:
        market_context_artifacts = persist_market_context(
            result.market_context,
            features_root=features_root,
            timeframe=timeframe,
            variant="research",
            frontier_from_ts=(
                pd.Timestamp(result.metadata_updates["persist_from_ts"])
                if result.metadata_updates.get("persist_from_ts")
                else result.plan.replay_from_ts
            ),
            full_rebuild=bool(force_rebuild or result.plan.mode == "full"),
            relevant_pairs=None,
        )
        artifacts.extend(market_context_artifacts)
        for artifact in market_context_artifacts:
            result.profiler.record_artifact(
                path=artifact.path,
                rows=artifact.rows,
                bytes_written=artifact.bytes_written,
                kind="market-context-research",
            )
        graph_result = result.graph_result
        smt_node = (
            graph_result.node_results.get("research_smt_research_table")
            if graph_result is not None
            else None
        )
        smt_research = (
            smt_node.primary_frame()
            if smt_node is not None and smt_node.primary_frame() is not None
            else pd.DataFrame()
        )
        if not smt_research.empty:
            smt_research_partitioned = smt_research.rename(
                columns={"smt_detect_ts": "timestamp"}
            )
            smt_artifacts = persist_partitioned_dataset(
                smt_research_partitioned,
                base_dir=features_root,
                dataset="research_smt",
                symbol=instrument,
                timeframe=timeframe,
                frontier_from_ts=(
                    pd.Timestamp(result.metadata_updates["persist_from_ts"])
                    if result.metadata_updates.get("persist_from_ts")
                    else result.plan.replay_from_ts
                ),
                full_rebuild=bool(force_rebuild or result.plan.mode == "full"),
            )
            artifacts.extend(smt_artifacts)
            for artifact in smt_artifacts:
                result.profiler.record_artifact(
                    path=artifact.path,
                    rows=artifact.rows,
                    bytes_written=artifact.bytes_written,
                    kind="smt-research",
                )
            smt_summary = (
                smt_node.output.payload.get("summary")
                if smt_node is not None
                else summarize_smt_research(smt_research)
            )
            smt_summary_artifact = write_json_atomic(
                smt_summary,
                Path(features_root)
                / "research_smt_summary"
                / instrument
                / timeframe
                / "summary.json",
            )
            artifacts.append(smt_summary_artifact)
            result.profiler.record_artifact(
                path=smt_summary_artifact.path,
                rows=smt_summary_artifact.rows,
                bytes_written=smt_summary_artifact.bytes_written,
                kind="smt-research-summary",
            )

        if build_cross_asset_audit:
            audit_node = (
                graph_result.node_results.get("research_cross_asset_audit")
                if graph_result is not None
                else None
            )
            audit_tables = audit_node.output.frames if audit_node is not None else {}
            for table_name, table in audit_tables.items():
                if table.empty:
                    continue
                artifact = write_parquet_atomic(
                    table,
                    Path(features_root)
                    / "research_cross_asset_audit"
                    / instrument
                    / timeframe
                    / f"{table_name}.parquet",
                )
                artifacts.append(artifact)
                result.profiler.record_artifact(
                    path=artifact.path,
                    rows=artifact.rows,
                    bytes_written=artifact.bytes_written,
                    kind=f"cross-asset-audit-{table_name}",
                )
            audit_summary = (
                audit_node.output.payload.get("summary")
                if audit_node is not None
                else summarize_cross_asset_correlation_audit(audit_tables)
            )
            audit_summary_artifact = write_json_atomic(
                audit_summary,
                Path(features_root)
                / "research_cross_asset_audit"
                / instrument
                / timeframe
                / "summary.json",
            )
            artifacts.append(audit_summary_artifact)
            result.profiler.record_artifact(
                path=audit_summary_artifact.path,
                rows=audit_summary_artifact.rows,
                bytes_written=audit_summary_artifact.bytes_written,
                kind="cross-asset-audit-summary",
            )
    updated_metadata = PipelineMetadata(
        symbol=instrument,
        timeframe=timeframe,
        pipeline=RESEARCH_PIPELINE_NAME,
        last_processed_ts=result.metadata_updates["last_processed_ts"],
        schema_version=result.metadata_updates["schema_version"],
        feature_contract_version=result.metadata_updates["feature_contract_version"],
        input_fingerprint=result.metadata_updates["input_fingerprint"],
        config_fingerprint=result.metadata_updates["config_fingerprint"],
        engine_version=result.metadata_updates["engine_version"],
        updated_at=datetime.now(UTC).isoformat(),
        extra={
            "dataset": "research",
            "features_root": str(features_root),
            "frontier_from_ts": (
                result.metadata_updates.get("persist_from_ts")
                or (
                    result.plan.replay_from_ts.isoformat()
                    if result.plan.replay_from_ts is not None
                    else None
                )
            ),
            "plan_mode": result.plan.mode,
            "plan_reason": result.plan.reason,
            "include_avwap": include_avwap,
            "include_cross_asset": include_cross_asset,
            "build_cross_asset_audit": build_cross_asset_audit,
        },
    )
    write_metadata_atomic(metadata_file, updated_metadata)
    return MaterializationResult(
        frame=result.frame,
        plan=result.plan,
        profiler=result.profiler,
        metadata=updated_metadata,
        artifacts=artifacts,
        metadata_file=str(metadata_file),
    )


# Backward-compatible alias
build_all_indicators = build_research_indicators
