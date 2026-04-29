from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.dag_runtime.contracts import (
    CachePolicy,
    FailureRecoveryPolicy,
    MutableScope,
    NodeKind,
    ReplayPolicyContract,
    SemanticClass,
    ValidationPolicy,
    WindowPolicy,
)


@dataclass(slots=True)
class NodeOutput:
    payload: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    profile_details: dict[str, Any] = field(default_factory=dict)

    def primary_frame(self) -> pd.DataFrame | None:
        if "frame" in self.frames:
            return self.frames["frame"]
        return next(iter(self.frames.values()), None)


@dataclass(slots=True)
class ExecutionPolicy:
    scheduler_mode: str = "serial"
    max_workers: int = 1
    max_concurrent_cache_writes: int = 1


@dataclass(slots=True)
class GraphRunContext:
    graph_name: str
    symbol: str
    timeframe: str
    inputs: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    cache_root: str | Path = "data/dag_cache"
    state_root: str | Path | None = None
    features_root: str | Path = "data/features"
    force: bool = False
    invalidate_cache: bool = False
    explain_only: bool = False
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)


ComputeFn = Callable[[GraphRunContext, dict[str, "NodeExecutionResult"]], NodeOutput]
FingerprintFn = Callable[
    ["NodeManifest", GraphRunContext, dict[str, "NodeExecutionResult"]],
    dict[str, Any],
]


@dataclass(frozen=True, slots=True)
class NodeManifest:
    graph_name: str
    node_name: str
    node_kind: NodeKind
    semantic_class: SemanticClass
    inputs: tuple[str, ...]
    upstream_nodes: tuple[str, ...]
    output_artifacts: tuple[str, ...]
    compute_fn: ComputeFn
    fingerprint_fn: FingerprintFn | None = None
    schema_version: int = 1
    feature_contract_version: int = 1
    engine_version: str = "1"
    replay_policy: ReplayPolicyContract | None = None
    window_policy: WindowPolicy = field(default_factory=WindowPolicy)
    cache_policy: CachePolicy = field(default_factory=CachePolicy)
    validation_policy: ValidationPolicy = field(default_factory=ValidationPolicy)
    mutable_scope: MutableScope = field(default_factory=MutableScope)
    failure_recovery_policy: FailureRecoveryPolicy = field(
        default_factory=FailureRecoveryPolicy
    )
    description: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NodeExecutionResult:
    manifest: NodeManifest
    output: NodeOutput
    fingerprint: str
    cache_hit: bool = False
    cache_path: Path | None = None

    def primary_frame(self) -> pd.DataFrame | None:
        return self.output.primary_frame()
