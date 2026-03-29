from __future__ import annotations

import json
import time

import pandas as pd

from src.pipeline_runtime import (
    PipelineMetadata,
    PipelineRunProfiler,
    dataframe_fingerprint,
    metadata_path,
    monthly_partition_label,
    partitioned_parquet_path,
    read_metadata,
    report_is_current,
    resolve_incremental_plan,
    update_report_fingerprint,
    write_metadata_atomic,
    write_parquet_atomic,
)


def _sample_frame(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="4h", tz="UTC"),
            "open": range(rows),
            "high": [x + 1 for x in range(rows)],
            "low": [x - 1 for x in range(rows)],
            "close": [x + 0.5 for x in range(rows)],
            "volume": [100 + x for x in range(rows)],
        }
    )


def test_metadata_round_trip(tmp_path):
    path = metadata_path(
        tmp_path, pipeline="build_live", symbol="XAU_USD", timeframe="H4"
    )
    metadata = PipelineMetadata(
        symbol="XAU_USD",
        timeframe="H4",
        pipeline="build_live",
        last_processed_ts="2026-01-02T00:00:00+00:00",
        schema_version=3,
        feature_contract_version=5,
        input_fingerprint="abc",
        config_fingerprint="def",
        engine_version="runtime-v1",
    )
    write_metadata_atomic(path, metadata)
    loaded = read_metadata(path)
    assert loaded == metadata


def test_write_parquet_atomic(tmp_path):
    frame = _sample_frame()
    path = partitioned_parquet_path(
        tmp_path,
        dataset="live",
        symbol="XAU_USD",
        timeframe="H4",
        partition=monthly_partition_label(frame["timestamp"].iloc[-1]),
    )
    result = write_parquet_atomic(frame, path)
    reloaded = pd.read_parquet(result.path)
    pd.testing.assert_frame_equal(reloaded, frame)
    assert result.bytes_written > 0


def test_resolve_incremental_plan_detects_noop():
    frame = _sample_frame()
    input_fingerprint = dataframe_fingerprint(frame)
    metadata = PipelineMetadata(
        symbol="XAU_USD",
        timeframe="H4",
        pipeline="build_live",
        last_processed_ts=frame["timestamp"].iloc[-1].isoformat(),
        schema_version=1,
        feature_contract_version=1,
        input_fingerprint=input_fingerprint,
        config_fingerprint="cfg",
    )
    plan = resolve_incremental_plan(
        frame,
        metadata=metadata,
        schema_version=1,
        feature_contract_version=1,
        input_fingerprint=input_fingerprint,
        config_fingerprint="cfg",
        replay_bars=100,
    )
    assert plan.is_noop is True
    assert plan.reason == "fingerprint-match-no-new-bars"


def test_report_cache_round_trip(tmp_path):
    report_path = tmp_path / "report.html"
    report_path.write_text("<html/>", encoding="utf-8")
    fingerprint_path = tmp_path / "report.html.meta.json"
    update_report_fingerprint(
        report_path=report_path,
        fingerprint_path=fingerprint_path,
        fingerprint="digest-1",
    )
    assert report_is_current(
        report_path=report_path,
        fingerprint_path=fingerprint_path,
        fingerprint="digest-1",
    )
    assert not report_is_current(
        report_path=report_path,
        fingerprint_path=fingerprint_path,
        fingerprint="digest-2",
    )


def test_profiler_summary_contains_stages_and_artifacts(tmp_path):
    profiler = PipelineRunProfiler(
        pipeline="build_live", symbol="XAU_USD", timeframe="H4"
    )
    frame = _sample_frame()
    started_at = time.perf_counter()
    profiler.record_stage(
        "load_raw", started_at=started_at, input_frame=frame, output_frame=frame
    )
    profiler.record_artifact(
        path=tmp_path / "artifact.parquet", rows=len(frame), bytes_written=123
    )
    summary_path = profiler.write_json(tmp_path / "summary.json")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["pipeline"] == "build_live"
    assert payload["stages"][0]["name"] == "load_raw"
    assert payload["artifacts_written"][0]["bytes_written"] == 123
