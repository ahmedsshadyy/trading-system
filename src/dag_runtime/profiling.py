from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class NodeProfile:
    node_name: str
    node_kind: str
    seconds: float
    cache_hit: bool
    rows_out: int | None = None
    estimated_memory_bytes: int | None = None
    fingerprint: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NodeArtifactRecord:
    node_name: str
    path: str
    bytes_written: int | None = None
    rows: int | None = None
    kind: str | None = None


class GraphRunProfiler:
    def __init__(
        self,
        *,
        graph_name: str,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        self.graph_name = graph_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.started_at = datetime.now(UTC)
        self._nodes: list[NodeProfile] = []
        self._artifacts: list[NodeArtifactRecord] = []
        self.scheduler_mode = "serial"
        self.worker_count = 1
        self.metrics: dict[str, Any] = {}

    def set_scheduler(self, *, scheduler_mode: str, worker_count: int) -> None:
        self.scheduler_mode = scheduler_mode
        self.worker_count = worker_count

    def set_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def record_node(
        self,
        *,
        node_name: str,
        node_kind: str,
        started_at: float | None = None,
        cache_hit: bool,
        seconds: float | None = None,
        rows_out: int | None = None,
        estimated_memory_bytes: int | None = None,
        fingerprint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._nodes.append(
            NodeProfile(
                node_name=node_name,
                node_kind=node_kind,
                seconds=(
                    seconds
                    if seconds is not None
                    else time.perf_counter() - (started_at or time.perf_counter())
                ),
                cache_hit=cache_hit,
                rows_out=rows_out,
                estimated_memory_bytes=estimated_memory_bytes,
                fingerprint=fingerprint,
                details=details or {},
            )
        )

    def record_artifact(
        self,
        *,
        node_name: str,
        path: str | Path,
        bytes_written: int | None = None,
        rows: int | None = None,
        kind: str | None = None,
    ) -> None:
        self._artifacts.append(
            NodeArtifactRecord(
                node_name=node_name,
                path=str(path),
                bytes_written=bytes_written,
                rows=rows,
                kind=kind,
            )
        )

    def summary(self) -> dict[str, Any]:
        ended_at = datetime.now(UTC)
        return {
            "graph_name": self.graph_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "scheduler_mode": self.scheduler_mode,
            "worker_count": self.worker_count,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "total_seconds": (ended_at - self.started_at).total_seconds(),
            "metrics": dict(self.metrics),
            "nodes": [asdict(node) for node in self._nodes],
            "artifacts_written": [asdict(artifact) for artifact in self._artifacts],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.summary(), sort_keys=True, indent=2), encoding="utf-8"
        )
        return target
