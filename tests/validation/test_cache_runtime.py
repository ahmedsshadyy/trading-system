from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.common.cache_runtime import (
    cleanup_validation_artifacts,
    load_or_build_context,
    load_or_build_validation_result,
    load_or_skip_report,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="4h", tz="UTC"),
            "open": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "high": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
            "low": [0.8, 1.8, 2.8, 3.8, 4.8, 5.8],
            "close": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
        }
    )


def test_load_or_build_context_hits_cache_on_second_run(tmp_path: Path) -> None:
    calls = {"count": 0}
    source = _sample_frame()

    def builder() -> pd.DataFrame:
        calls["count"] += 1
        return source.assign(marker=1)

    first = load_or_build_context(
        validator="unit_validator",
        symbol="XAU_USD",
        timeframe="H4",
        input_df=source,
        config_payload={"variant": "context"},
        build_fn=builder,
        cache_root=tmp_path,
    )
    second = load_or_build_context(
        validator="unit_validator",
        symbol="XAU_USD",
        timeframe="H4",
        input_df=source,
        config_payload={"variant": "context"},
        build_fn=builder,
        cache_root=tmp_path,
    )

    assert calls["count"] == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    pd.testing.assert_frame_equal(first.frame, second.frame, check_dtype=False)


def test_validation_result_invalidates_when_params_change(tmp_path: Path) -> None:
    calls = {"count": 0}

    def builder(multiplier: int) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
        calls["count"] += 1
        frame = _sample_frame().assign(score=lambda df: df["close"] * multiplier)
        return (
            {"summary": {"multiplier": multiplier}},
            {
                "frame": frame,
                "event_table": frame.tail(2),
                "candidate_table": frame.head(2),
            },
        )

    first = load_or_build_validation_result(
        validator="unit_validator",
        symbol="XAU_USD",
        timeframe="H4",
        stage="debug",
        context_fingerprint="ctx-1",
        config_payload={"multiplier": 2},
        build_fn=lambda: builder(2),
        cache_root=tmp_path,
    )
    second = load_or_build_validation_result(
        validator="unit_validator",
        symbol="XAU_USD",
        timeframe="H4",
        stage="debug",
        context_fingerprint="ctx-1",
        config_payload={"multiplier": 2},
        build_fn=lambda: builder(2),
        cache_root=tmp_path,
    )
    third = load_or_build_validation_result(
        validator="unit_validator",
        symbol="XAU_USD",
        timeframe="H4",
        stage="debug",
        context_fingerprint="ctx-1",
        config_payload={"multiplier": 3},
        build_fn=lambda: builder(3),
        cache_root=tmp_path,
    )

    assert calls["count"] == 2
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert third.cache_hit is False
    assert second.payload["summary"]["multiplier"] == 2
    assert third.payload["summary"]["multiplier"] == 3


def test_load_or_skip_report_skips_rewrite_when_fingerprint_matches(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.html"
    writes = {"count": 0}

    def writer(path: Path) -> Path:
        writes["count"] += 1
        path.write_text(f"write-{writes['count']}", encoding="utf-8")
        return path

    first_path, first_skipped = load_or_skip_report(
        report_path,
        fingerprint="abc",
        writer=writer,
    )
    second_path, second_skipped = load_or_skip_report(
        report_path,
        fingerprint="abc",
        writer=writer,
    )

    assert first_path == report_path
    assert second_path == report_path
    assert first_skipped is False
    assert second_skipped is True
    assert writes["count"] == 1


def test_cleanup_validation_artifacts_prunes_cache_only(tmp_path: Path) -> None:
    cache_root = tmp_path / "validation_cache"
    cache_file = cache_root / "validator" / "artifact.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("cache", encoding="utf-8")

    report_root = tmp_path / "reports"
    report_file = report_root / "report.html"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("report", encoding="utf-8")

    removed = cleanup_validation_artifacts(
        cache_root=cache_root,
        max_age_days=0,
        report_roots=[report_root],
    )

    assert cache_file in removed
    assert not cache_file.exists()
    assert report_file.exists()
