from __future__ import annotations

import pandas as pd

from src.indicators.pipelines.build_live import build_live_indicators, run_live_pipeline
from src.indicators.pipelines.build_research import (
    build_research_indicators,
    run_research_pipeline,
)
from src.pipeline_runtime import PipelineMetadata


def _sample_ohlcv(rows: int = 260) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="4h", tz="UTC")
    close = pd.Series(range(rows), dtype=float).mul(0.8).add(2000.0)
    open_ = close.shift(1).fillna(close.iloc[0] - 0.2)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.4
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.4
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


def test_live_runtime_full_matches_direct_builder():
    raw = _sample_ohlcv()
    direct = build_live_indicators(raw, include_vp=False)
    runtime = run_live_pipeline(raw, include_vp=False)
    pd.testing.assert_frame_equal(runtime.frame, direct)
    assert runtime.plan.mode == "full"


def test_live_runtime_incremental_matches_full_rebuild():
    raw = _sample_ohlcv(320)
    baseline_raw = raw.iloc[:-5].copy()
    baseline = run_live_pipeline(baseline_raw, include_vp=False)
    metadata = PipelineMetadata(
        symbol="XAU_USD",
        timeframe="H4",
        pipeline="build_live",
        last_processed_ts=baseline.metadata_updates["last_processed_ts"],
        schema_version=baseline.metadata_updates["schema_version"],
        feature_contract_version=baseline.metadata_updates["feature_contract_version"],
        input_fingerprint="stale",
        config_fingerprint=baseline.metadata_updates["config_fingerprint"],
        engine_version=baseline.metadata_updates["engine_version"],
    )
    incremental = run_live_pipeline(
        raw,
        include_vp=False,
        existing_history=baseline.frame,
        metadata=metadata,
    )
    rebuilt = build_live_indicators(raw, include_vp=False)
    pd.testing.assert_frame_equal(
        incremental.frame.reset_index(drop=True), rebuilt.reset_index(drop=True)
    )
    assert incremental.plan.mode == "incremental"


def test_research_runtime_noop_when_metadata_matches():
    raw = _sample_ohlcv(240)
    baseline = run_research_pipeline(raw, include_vp=False, include_avwap=False)
    metadata = PipelineMetadata(
        symbol="XAU_USD",
        timeframe="H4",
        pipeline="build_research",
        last_processed_ts=baseline.metadata_updates["last_processed_ts"],
        schema_version=baseline.metadata_updates["schema_version"],
        feature_contract_version=baseline.metadata_updates["feature_contract_version"],
        input_fingerprint=baseline.metadata_updates["input_fingerprint"],
        config_fingerprint=baseline.metadata_updates["config_fingerprint"],
        engine_version=baseline.metadata_updates["engine_version"],
    )
    noop = run_research_pipeline(
        raw,
        include_vp=False,
        include_avwap=False,
        existing_history=baseline.frame,
        metadata=metadata,
    )
    assert noop.plan.is_noop is True
    pd.testing.assert_frame_equal(noop.frame, baseline.frame)


def test_research_runtime_full_matches_direct_builder():
    raw = _sample_ohlcv(240)
    direct = build_research_indicators(raw, include_vp=False, include_avwap=False)
    runtime = run_research_pipeline(raw, include_vp=False, include_avwap=False)
    pd.testing.assert_frame_equal(runtime.frame, direct)
