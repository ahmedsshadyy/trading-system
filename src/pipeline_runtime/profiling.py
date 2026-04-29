from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StageProfile:
    name: str
    seconds: float
    rows_in: int | None = None
    rows_out: int | None = None
    estimated_memory_bytes: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactRecord:
    path: str
    rows: int | None = None
    bytes_written: int | None = None
    kind: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReadRecord:
    path: str
    rows: int | None = None
    bytes_read: int | None = None
    kind: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class PipelineRunProfiler:
    def __init__(
        self,
        *,
        pipeline: str,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.symbol = symbol
        self.timeframe = timeframe
        self.started_at = datetime.now(UTC)
        self._started_perf_counter = time.perf_counter()
        self._started_process_cpu = time.process_time()
        self._stages: list[StageProfile] = []
        self._artifacts: list[ArtifactRecord] = []
        self._reads: list[ReadRecord] = []
        self._counters: dict[str, int | float] = {}

    def measure_frame(self, frame: Any) -> tuple[int | None, int | None]:
        if frame is None:
            return None, None
        rows = len(frame) if hasattr(frame, "__len__") else None
        memory = None
        if hasattr(frame, "memory_usage"):
            try:
                memory = int(frame.memory_usage(deep=True).sum())
            except Exception:
                memory = None
        return rows, memory

    def record_stage(
        self,
        name: str,
        *,
        started_at: float,
        input_frame: Any = None,
        output_frame: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        rows_in, memory = self.measure_frame(input_frame)
        rows_out, _ = self.measure_frame(output_frame)
        self._stages.append(
            StageProfile(
                name=name,
                seconds=time.perf_counter() - started_at,
                rows_in=rows_in,
                rows_out=rows_out,
                estimated_memory_bytes=memory,
                details=details or {},
            )
        )

    def record_artifact(
        self,
        *,
        path: str | Path,
        rows: int | None = None,
        bytes_written: int | None = None,
        kind: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._artifacts.append(
            ArtifactRecord(
                path=str(path),
                rows=rows,
                bytes_written=bytes_written,
                kind=kind,
                details=details or {},
            )
        )

    def record_read(
        self,
        *,
        path: str | Path,
        rows: int | None = None,
        bytes_read: int | None = None,
        kind: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._reads.append(
            ReadRecord(
                path=str(path),
                rows=rows,
                bytes_read=bytes_read,
                kind=kind,
                details=details or {},
            )
        )

    def increment_counter(self, name: str, value: int | float = 1) -> None:
        current = self._counters.get(name, 0)
        self._counters[name] = current + value

    def set_metric(self, name: str, value: int | float | str | bool | None) -> None:
        self._counters[name] = value

    def summary(self) -> dict[str, Any]:
        ended_at = datetime.now(UTC)
        total_seconds = time.perf_counter() - self._started_perf_counter
        process_cpu_seconds = time.process_time() - self._started_process_cpu
        bytes_written = sum(record.bytes_written or 0 for record in self._artifacts)
        bytes_read = sum(record.bytes_read or 0 for record in self._reads)
        return {
            "pipeline": self.pipeline,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "total_seconds": total_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "avg_cpu_utilization_pct": (
                (process_cpu_seconds / total_seconds) * 100.0
                if total_seconds > 0
                else None
            ),
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "parquet_reads": sum(
                1 for record in self._reads if record.path.endswith(".parquet")
            ),
            "artifact_writes": len(self._artifacts),
            "counters": dict(self._counters),
            "stages": [asdict(stage) for stage in self._stages],
            "artifacts_read": [asdict(record) for record in self._reads],
            "artifacts_written": [asdict(artifact) for artifact in self._artifacts],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.summary(), sort_keys=True, indent=2), encoding="utf-8"
        )
        return target
