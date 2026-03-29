from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.pipeline_runtime.metadata import PipelineMetadata

StageFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True, slots=True)
class ReplayPolicy:
    stage: str
    classification: str
    replay_bars: int
    warmup_bars: int = 0
    carried_state: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class PipelineStage:
    name: str
    fn: StageFn
    policy: ReplayPolicy


@dataclass(slots=True)
class IncrementalPlan:
    mode: str
    start_ts: pd.Timestamp | None
    replay_from_ts: pd.Timestamp | None
    replay_rows: int
    reason: str
    metadata: PipelineMetadata | None
    should_persist: bool
    is_noop: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)


def resolve_incremental_plan(
    df: pd.DataFrame,
    *,
    metadata: PipelineMetadata | None,
    schema_version: int,
    feature_contract_version: int,
    input_fingerprint: str,
    config_fingerprint: str,
    replay_bars: int,
    force_rebuild: bool = False,
    timestamp_col: str = "timestamp",
) -> IncrementalPlan:
    if df.empty:
        return IncrementalPlan(
            mode="noop",
            start_ts=None,
            replay_from_ts=None,
            replay_rows=0,
            reason="empty-input",
            metadata=metadata,
            should_persist=False,
            is_noop=True,
        )

    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    last_ts = ts.iloc[-1]

    if force_rebuild or metadata is None:
        return IncrementalPlan(
            mode="full",
            start_ts=ts.iloc[0],
            replay_from_ts=ts.iloc[0],
            replay_rows=len(df),
            reason="force-rebuild" if force_rebuild else "missing-metadata",
            metadata=metadata,
            should_persist=True,
        )

    if metadata.schema_version != schema_version:
        reason = "schema-version-changed"
    elif metadata.feature_contract_version != feature_contract_version:
        reason = "feature-contract-changed"
    elif metadata.config_fingerprint != config_fingerprint:
        reason = "config-changed"
    elif metadata.input_fingerprint == input_fingerprint and metadata.last_processed_ts:
        prior_ts = pd.Timestamp(metadata.last_processed_ts)
        if prior_ts.tzinfo is None:
            prior_ts = prior_ts.tz_localize("UTC")
        if prior_ts >= last_ts:
            return IncrementalPlan(
                mode="noop",
                start_ts=last_ts,
                replay_from_ts=last_ts,
                replay_rows=0,
                reason="fingerprint-match-no-new-bars",
                metadata=metadata,
                should_persist=False,
                is_noop=True,
            )
        reason = "new-bars"
    else:
        reason = "input-changed"

    prior_ts = (
        pd.Timestamp(metadata.last_processed_ts)
        if metadata.last_processed_ts
        else ts.iloc[0]
    )
    if prior_ts.tzinfo is None:
        prior_ts = prior_ts.tz_localize("UTC")
    start_idx = max(int(ts.searchsorted(prior_ts, side="right")) - replay_bars, 0)
    replay_from_ts = ts.iloc[start_idx]
    return IncrementalPlan(
        mode="incremental" if reason in {"new-bars", "input-changed"} else "full",
        start_ts=prior_ts,
        replay_from_ts=replay_from_ts,
        replay_rows=int(len(df.iloc[start_idx:])),
        reason=reason,
        metadata=metadata,
        should_persist=True,
        diagnostics={"last_input_ts": last_ts.isoformat()},
    )


def slice_frame_for_plan(
    df: pd.DataFrame,
    plan: IncrementalPlan,
    *,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    if plan.replay_from_ts is None or plan.mode == "full":
        return df.copy()
    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    return df.loc[ts >= plan.replay_from_ts].copy()


def merge_recomputed_frontier(
    history_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    *,
    frontier_from_ts: pd.Timestamp | None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    if frontier_from_ts is None or history_df.empty:
        return frontier_df.copy()
    ts = pd.to_datetime(history_df[timestamp_col], utc=True, errors="coerce")
    frozen = history_df.loc[ts < frontier_from_ts].copy()
    frontier_ts = pd.to_datetime(frontier_df[timestamp_col], utc=True, errors="coerce")
    trimmed_frontier = frontier_df.loc[frontier_ts >= frontier_from_ts].copy()
    merged = pd.concat([frozen, trimmed_frontier], ignore_index=True)
    merged = merged.drop_duplicates(subset=[timestamp_col], keep="last")
    return merged.sort_values(timestamp_col).reset_index(drop=True)
