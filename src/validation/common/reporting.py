from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.pipeline_runtime import (
    dataframe_fingerprint,
    report_is_current,
    update_report_fingerprint,
)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    return logging.getLogger(name)


def load_windowed_parquet(
    path: str | Path,
    *,
    full: bool = False,
    tail_rows: int | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if full or tail_rows is None:
        return df
    return df.tail(tail_rows).copy()


def report_fingerprint(
    df: pd.DataFrame, *, extra: dict[str, object] | None = None
) -> str:
    payload = dataframe_fingerprint(df, strategy="content")
    if not extra:
        return payload
    extra_df = pd.DataFrame([extra])
    return dataframe_fingerprint(extra_df, strategy="content") + ":" + payload


def report_cache_metadata_path(report_path: str | Path) -> Path:
    report = Path(report_path)
    return report.with_suffix(report.suffix + ".meta.json")


def should_write_report(
    report_path: str | Path,
    *,
    fingerprint: str,
    force: bool = False,
) -> bool:
    return not report_is_current(
        report_path=report_path,
        fingerprint_path=report_cache_metadata_path(report_path),
        fingerprint=fingerprint,
        force=force,
    )


def mark_report_written(report_path: str | Path, *, fingerprint: str) -> None:
    update_report_fingerprint(
        report_path=report_path,
        fingerprint_path=report_cache_metadata_path(report_path),
        fingerprint=fingerprint,
    )
