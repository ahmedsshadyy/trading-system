from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.indicators.pipelines.build_live import materialize_live_features
from src.indicators.pipelines.build_research import (
    build_research_indicators,
    materialize_research_features,
)
from src.pipeline_runtime import load_partitioned_dataset, read_metadata


def _sample_ohlcv(rows: int = 420) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="4h", tz="UTC")
    close = pd.Series(range(rows), dtype=float).mul(0.7).add(2000.0)
    open_ = close.shift(1).fillna(close.iloc[0] - 0.1)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
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


def _partition_dir(root: Path, dataset: str) -> Path:
    return root / dataset / "XAU_USD" / "H4"


def _normalize_missing_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(
            out[column]
        ) or pd.api.types.is_datetime64_any_dtype(out[column]):
            continue
        out[column] = out[column].astype("object").where(out[column].notna(), None)
    return out


def test_noop_rerun_does_not_rewrite_historical_partitions(tmp_path):
    raw = _sample_ohlcv()
    first = materialize_live_features(
        raw, include_vp=False, features_root=str(tmp_path)
    )
    first_partitions = sorted(_partition_dir(tmp_path, "live").glob("*.parquet"))
    mtimes_before = {path.name: path.stat().st_mtime_ns for path in first_partitions}

    second = materialize_live_features(
        raw, include_vp=False, features_root=str(tmp_path)
    )
    mtimes_after = {path.name: path.stat().st_mtime_ns for path in first_partitions}
    persisted = load_partitioned_dataset(
        tmp_path,
        dataset="live",
        symbol="XAU_USD",
        timeframe="H4",
    )

    assert second.plan.is_noop is True
    assert second.artifacts == []
    assert mtimes_before == mtimes_after
    pd.testing.assert_frame_equal(
        second.frame.reset_index(drop=True),
        persisted.reset_index(drop=True),
        check_dtype=False,
    )


def test_one_bar_append_touches_only_frontier_partition(tmp_path):
    raw = _sample_ohlcv()
    baseline_raw = raw.iloc[:-1].copy()
    materialize_live_features(
        baseline_raw, include_vp=False, features_root=str(tmp_path)
    )
    partitions = sorted(_partition_dir(tmp_path, "live").glob("*.parquet"))
    mtimes_before = {path.name: path.stat().st_mtime_ns for path in partitions}

    updated = materialize_live_features(
        raw, include_vp=False, features_root=str(tmp_path)
    )
    mtimes_after = {path.name: path.stat().st_mtime_ns for path in partitions}
    touched = {
        name for name, mtime in mtimes_after.items() if mtime != mtimes_before[name]
    }
    written = {artifact.path.name for artifact in updated.artifacts}

    assert updated.plan.mode == "incremental"
    assert len(touched) == 1
    assert touched == written


def test_multi_bar_append_preserves_parity_with_full_rebuild(tmp_path):
    raw = _sample_ohlcv(430)
    baseline_raw = raw.iloc[:-7].copy()
    materialize_research_features(
        baseline_raw,
        include_vp=False,
        include_avwap=False,
        features_root=str(tmp_path),
    )
    updated = materialize_research_features(
        raw,
        include_vp=False,
        include_avwap=False,
        features_root=str(tmp_path),
    )

    persisted = load_partitioned_dataset(
        tmp_path,
        dataset="research",
        symbol="XAU_USD",
        timeframe="H4",
    )
    rebuilt = build_research_indicators(raw, include_vp=False, include_avwap=False)

    pd.testing.assert_frame_equal(
        _normalize_missing_non_numeric(updated.frame.reset_index(drop=True)),
        _normalize_missing_non_numeric(rebuilt.reset_index(drop=True)),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        _normalize_missing_non_numeric(persisted.reset_index(drop=True)),
        _normalize_missing_non_numeric(rebuilt.reset_index(drop=True)),
        check_dtype=False,
    )


def test_interrupted_write_does_not_advance_metadata_or_corrupt_artifacts(tmp_path):
    raw = _sample_ohlcv(250)
    baseline = materialize_live_features(
        raw.iloc[:-1].copy(), include_vp=False, features_root=str(tmp_path)
    )
    metadata_before = read_metadata(baseline.metadata_file)
    persisted_before = load_partitioned_dataset(
        tmp_path,
        dataset="live",
        symbol="XAU_USD",
        timeframe="H4",
    )

    def failing_writer(df: pd.DataFrame, path: str | Path):
        raise RuntimeError("simulated write failure")

    with pytest.raises(RuntimeError):
        materialize_live_features(
            raw,
            include_vp=False,
            features_root=str(tmp_path),
            partition_writer=failing_writer,
        )

    metadata_after = read_metadata(baseline.metadata_file)
    persisted_after = load_partitioned_dataset(
        tmp_path,
        dataset="live",
        symbol="XAU_USD",
        timeframe="H4",
    )
    assert metadata_after == metadata_before
    pd.testing.assert_frame_equal(
        persisted_after.reset_index(drop=True), persisted_before.reset_index(drop=True)
    )


def test_historical_immutable_partitions_remain_untouched_on_ordinary_run(tmp_path):
    raw = _sample_ohlcv(450)
    baseline = materialize_live_features(
        raw.iloc[:-4].copy(), include_vp=False, features_root=str(tmp_path)
    )
    partitions = sorted(_partition_dir(tmp_path, "live").glob("*.parquet"))
    assert len(partitions) >= 2
    historical = partitions[:-1]
    frontier = partitions[-1]
    hist_mtimes_before = {path.name: path.stat().st_mtime_ns for path in historical}
    frontier_mtime_before = frontier.stat().st_mtime_ns

    updated = materialize_live_features(
        raw, include_vp=False, features_root=str(tmp_path)
    )

    hist_mtimes_after = {path.name: path.stat().st_mtime_ns for path in historical}
    frontier_mtime_after = frontier.stat().st_mtime_ns
    assert updated.plan.mode == "incremental"
    assert hist_mtimes_before == hist_mtimes_after
    assert frontier_mtime_after != frontier_mtime_before
