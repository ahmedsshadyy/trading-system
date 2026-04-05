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
from typing import Any

import pandas as pd

from src.dag_runtime import execute_graph, GraphRunContext
from src.indicators._helpers.schema import normalize_candle_schema
from src.pipeline_runtime import (
    ArtifactWriteResult,
    PipelineMetadata,
    PipelineRunProfiler,
    PipelineStage,
    ReplayPolicy,
    cleanup_temp_artifacts,
    dataframe_fingerprint,
    load_partitioned_dataset,
    merge_recomputed_frontier,
    metadata_path,
    persist_partitioned_dataset,
    read_metadata,
    resolve_incremental_plan,
    slice_frame_for_plan,
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
from src.indicators.smc.sweeps import add_liquidity_sweep
from src.indicators.smc.equal_hl import add_equal_hl
from src.indicators.smc.displacement import add_displacement_candle
from src.indicators.smc.amd import add_amd_engine

RESEARCH_PIPELINE_NAME = "build_research"
RESEARCH_SCHEMA_VERSION = 1
RESEARCH_FEATURE_CONTRACT_VERSION = 1


@dataclass(slots=True)
class PipelineExecutionResult:
    frame: pd.DataFrame
    plan: Any
    profiler: PipelineRunProfiler
    metadata_updates: dict[str, Any]


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
    return ts.to_period("M").start_time.tz_localize("UTC")


def _research_stages(
    *,
    instrument: str,
    swing_window: int,
    include_vp: bool,
    include_avwap: bool,
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
            ReplayPolicy("rsi_divergence", "B", replay_bars=220, warmup_bars=14),
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
            ReplayPolicy("fvg_stack", "B", replay_bars=300, carried_state=True),
        ),
        PipelineStage(
            "displacement",
            add_displacement_candle,
            ReplayPolicy("displacement", "B", replay_bars=300, carried_state=True),
        ),
        PipelineStage(
            "order_blocks",
            add_ob,
            ReplayPolicy("order_blocks", "B", replay_bars=300, carried_state=True),
        ),
        PipelineStage(
            "ob_mitigation",
            add_ob_mitigation,
            ReplayPolicy("ob_mitigation", "B", replay_bars=300, carried_state=True),
        ),
        PipelineStage(
            "liquidity_sweeps",
            add_liquidity_sweep,
            ReplayPolicy("liquidity_sweeps", "B", replay_bars=300, carried_state=True),
        ),
        PipelineStage(
            "equal_hl",
            add_equal_hl,
            ReplayPolicy("equal_hl", "B", replay_bars=300, carried_state=True),
        ),
        PipelineStage(
            "amd_engine",
            lambda df: add_amd_engine(df, add_labels=False),
            ReplayPolicy("amd_engine", "B", replay_bars=300, carried_state=True),
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
                ReplayPolicy("anchored_vwap", "B", replay_bars=300, carried_state=True),
            )
        )
    stages.append(
        PipelineStage(
            "regime",
            lambda df: add_regime(df, include_research_only=True),
            ReplayPolicy("regime", "C", replay_bars=240, carried_state=True),
        )
    )
    return stages


def build_research_indicators(
    df: pd.DataFrame,
    instrument: str = "XAU_USD",
    swing_window: int = 6,
    include_vp: bool = True,
    include_avwap: bool = False,
    profiler: PipelineRunProfiler | None = None,
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
    out = _coerce_ohlc(df)
    from src.dag_runtime.builtin_graphs import build_research_stage_graph

    graph = build_research_stage_graph(
        instrument=instrument,
        swing_window=swing_window,
        include_vp=include_vp,
        include_avwap=include_avwap,
    )
    result = execute_graph(
        graph,
        context=GraphRunContext(
            graph_name=graph.graph_name,
            symbol=instrument,
            timeframe="graph",
            inputs={"raw_input": out},
            config={
                "instrument": instrument,
                "swing_window": swing_window,
                "include_vp": include_vp,
                "include_avwap": include_avwap,
            },
            cache_root="data/dag_cache",
            force=True,
            invalidate_cache=True,
        ),
    )
    return result.primary_frame()


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
) -> PipelineExecutionResult:
    runtime_profiler = profiler or PipelineRunProfiler(
        pipeline=RESEARCH_PIPELINE_NAME,
        symbol=instrument,
        timeframe=timeframe,
    )
    normalized = normalize_candle_schema(df, require_volume=True)
    input_fingerprint = dataframe_fingerprint(normalized)
    config_payload = {
        "instrument": instrument,
        "timeframe": timeframe,
        "swing_window": swing_window,
        "include_vp": include_vp,
        "include_avwap": include_avwap,
        **(config or {}),
    }
    config_fingerprint = dataframe_fingerprint(
        pd.DataFrame([config_payload]), strategy="content"
    )
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
        force_rebuild=force_rebuild,
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
        )

    working = slice_frame_for_plan(normalized, plan)
    computed = build_research_indicators(
        working,
        instrument=instrument,
        swing_window=swing_window,
        include_vp=include_vp,
        include_avwap=include_avwap,
        profiler=runtime_profiler,
    )
    if existing_history is not None and plan.mode == "incremental":
        final_frame = merge_recomputed_frontier(
            existing_history,
            computed,
            frontier_from_ts=persist_from_ts,
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
                persist_from_ts.isoformat() if persist_from_ts is not None else None
            ),
        },
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
        )
    preview = run_research_pipeline(
        raw_df,
        instrument=instrument,
        timeframe=timeframe,
        swing_window=swing_window,
        include_vp=include_vp,
        include_avwap=include_avwap,
        existing_history=existing_history,
        metadata=metadata,
        config=config,
        force_rebuild=force_rebuild,
        profiler=runtime_profiler,
        replay_bars_override=(
            len(raw_df) if metadata is not None and not force_rebuild else None
        ),
    )
    if preview.plan.is_noop:
        return MaterializationResult(
            frame=(
                existing_history.copy()
                if existing_history is not None
                else preview.frame
            ),
            plan=preview.plan,
            profiler=preview.profiler,
            metadata=metadata,
            artifacts=[],
            metadata_file=str(metadata_file),
        )

    if metadata is not None and not force_rebuild:
        runtime_profiler = PipelineRunProfiler(
            pipeline=f"{RESEARCH_PIPELINE_NAME}_materialize",
            symbol=instrument,
            timeframe=timeframe,
        )
        result = run_research_pipeline(
            raw_df,
            instrument=instrument,
            timeframe=timeframe,
            swing_window=swing_window,
            include_vp=include_vp,
            include_avwap=include_avwap,
            existing_history=None,
            metadata=None,
            config=config,
            force_rebuild=True,
            profiler=runtime_profiler,
            replay_bars_override=len(raw_df),
        )
    else:
        result = preview

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
