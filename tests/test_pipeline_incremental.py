from __future__ import annotations

import pandas as pd

from src.indicators.pipelines.build_live import (
    _live_stages,
    build_live_indicators,
    run_live_pipeline,
)
from src.indicators.pipelines.build_research import (
    _research_stages,
    build_research_indicators,
    run_research_pipeline,
)
from src.pipeline_runtime import PipelineMetadata


def _sample_ohlcv(
    rows: int = 260,
    *,
    start: str = "2026-01-01",
    close_base: float = 2000.0,
    slope: float = 0.8,
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=rows, freq="4h", tz="UTC")
    close = pd.Series(range(rows), dtype=float).mul(slope).add(close_base)
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


def _cross_asset_peers(rows: int = 320) -> dict[str, pd.DataFrame]:
    return {
        "DXY": _sample_ohlcv(rows, start="2025-12-20", close_base=100.0, slope=0.07),
        "USD_JPY": _sample_ohlcv(
            rows, start="2025-12-20", close_base=145.0, slope=0.03
        ),
        "USOIL": _sample_ohlcv(rows, start="2025-12-20", close_base=70.0, slope=0.11),
        "USD_CAD": _sample_ohlcv(
            rows, start="2025-12-20", close_base=1.30, slope=0.002
        ),
        "EUR_USD": _sample_ohlcv(
            rows, start="2025-12-20", close_base=1.10, slope=0.001
        ),
    }


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


def test_replay_window_regression_values_are_locked():
    live_policies = {
        stage.name: stage.policy.replay_bars
        for stage in _live_stages(
            instrument="XAU_USD", swing_window=6, include_vp=False
        )
    }
    research_policies = {
        stage.name: stage.policy.replay_bars
        for stage in _research_stages(
            instrument="XAU_USD",
            swing_window=6,
            include_vp=False,
            include_avwap=True,
        )
    }

    assert live_policies["swings"] == 400
    assert live_policies["trend_state"] == 400
    assert live_policies["bos"] == 400
    assert live_policies["choch"] == 400
    assert live_policies["fvg_stack"] == 240
    assert live_policies["displacement"] == 240
    assert live_policies["order_blocks"] == 240
    assert live_policies["ob_mitigation"] == 240
    assert live_policies["equal_hl"] == 240
    assert live_policies["amd_engine"] == 240
    assert live_policies["rsi_divergence"] == 200
    assert live_policies["regime"] == 200

    assert research_policies["swings"] == 400
    assert research_policies["trend_state"] == 400
    assert research_policies["bos"] == 400
    assert research_policies["choch"] == 400
    assert research_policies["fvg_stack"] == 240
    assert research_policies["displacement"] == 240
    assert research_policies["order_blocks"] == 240
    assert research_policies["ob_mitigation"] == 240
    assert research_policies["equal_hl"] == 240
    assert research_policies["amd_engine"] == 240
    assert research_policies["rsi_divergence"] == 200
    assert research_policies["anchored_vwap"] == 200
    assert research_policies["regime"] == 240


def test_live_runtime_incremental_matches_full_rebuild_across_month_boundary():
    raw = _sample_ohlcv(360, start="2025-12-20")
    baseline_raw = raw.iloc[:-12].copy()
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


def test_research_cross_asset_new_bars_stays_incremental_when_config_is_stable():
    raw = _sample_ohlcv(360, start="2025-12-20")
    peers = _cross_asset_peers(360)
    baseline_raw = raw.iloc[:-12].copy()
    baseline = run_research_pipeline(
        baseline_raw,
        include_vp=False,
        include_avwap=False,
        include_cross_asset=True,
        timeframe="H4",
        peer_raw_frames=peers,
        raw_data_root=None,
    )
    metadata = PipelineMetadata(
        symbol="XAU_USD",
        timeframe="H4",
        pipeline="build_research",
        last_processed_ts=baseline.metadata_updates["last_processed_ts"],
        schema_version=baseline.metadata_updates["schema_version"],
        feature_contract_version=baseline.metadata_updates["feature_contract_version"],
        input_fingerprint="stale",
        config_fingerprint=baseline.metadata_updates["config_fingerprint"],
        engine_version=baseline.metadata_updates["engine_version"],
    )
    incremental = run_research_pipeline(
        raw,
        include_vp=False,
        include_avwap=False,
        include_cross_asset=True,
        timeframe="H4",
        peer_raw_frames=peers,
        raw_data_root=None,
        existing_history=baseline.frame,
        metadata=metadata,
    )
    assert incremental.plan.mode == "incremental"
