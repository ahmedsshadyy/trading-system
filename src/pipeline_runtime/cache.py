from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.pipeline_runtime.fingerprint import fingerprint_mapping
from src.pipeline_runtime.metadata import (
    read_metadata,
    write_metadata_atomic,
    PipelineMetadata,
)


@dataclass(frozen=True, slots=True)
class CacheKey:
    symbol: str
    timeframe: str
    stage: str
    schema_version: int
    config_fingerprint: str
    upstream_fingerprint: str
    time_range: str

    def digest(self) -> str:
        return fingerprint_mapping(self.__dict__)


def report_is_current(
    *,
    report_path: str | Path,
    fingerprint_path: str | Path,
    fingerprint: str,
    force: bool = False,
) -> bool:
    if force:
        return False
    report_target = Path(report_path)
    if not report_target.exists():
        return False
    metadata = read_metadata(fingerprint_path)
    return bool(metadata and metadata.input_fingerprint == fingerprint)


def update_report_fingerprint(
    *,
    report_path: str | Path,
    fingerprint_path: str | Path,
    fingerprint: str,
) -> None:
    target = Path(report_path)
    metadata = PipelineMetadata(
        symbol="report",
        timeframe="report",
        pipeline=target.stem,
        input_fingerprint=fingerprint,
        config_fingerprint=fingerprint,
        extra={"report_path": str(target)},
    )
    write_metadata_atomic(fingerprint_path, metadata)
