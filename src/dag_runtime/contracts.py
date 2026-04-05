from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NodeKind = Literal[
    "source", "compute", "aggregate", "selection", "materialize", "report"
]
SemanticClass = Literal["A", "B", "C"]


@dataclass(frozen=True, slots=True)
class ReplayPolicyContract:
    mode: Literal[
        "full_history_only", "bounded_replay", "carried_state", "append_only_safe"
    ]
    replay_bars: int | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class WindowPolicy:
    mode: Literal["full", "tail", "frontier", "explicit_range"] = "full"
    tail_rows: int | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CachePolicy:
    materialize: bool = True
    artifact_kind: Literal["frame", "bundle", "report", "ephemeral"] = "bundle"
    allow_noop_reuse: bool = True
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    level: Literal["unit", "node_parity", "graph_parity", "incremental", "report"] = (
        "node_parity"
    )
    exact: bool = True
    tolerance: float | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class MutableScope:
    scope: Literal[
        "immutable", "frontier_only", "append_only", "explicit_rebuild_only"
    ] = "immutable"
    notes: str = ""


@dataclass(frozen=True, slots=True)
class FailureRecoveryPolicy:
    atomic_write: bool = True
    temp_cleanup: bool = True
    metadata_after_artifacts: bool = True
    notes: str = ""
