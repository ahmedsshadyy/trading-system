from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable
from dataclasses import asdict

import pandas as pd

from src.dag_runtime.node import GraphRunContext, NodeExecutionResult, NodeManifest
from src.pipeline_runtime import dataframe_fingerprint, fingerprint_mapping


def compute_source_hash(fn: Callable) -> str:
    """Return a stable short hash of ``fn``'s source code.

    The hash captures every textual change to the function — including
    whitespace and comments. This is intentional: any source-level change
    invalidates the node cache so stale output is never served.

    Falls back to ``repr(fn)`` for builtins and C-implemented callables
    where source inspection fails.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def compute_multi_source_hash(*fns: Callable) -> str:
    """Combine source hashes of multiple functions into one stable hash.

    Use for composite stages whose output depends on more than one
    underlying function (e.g. ``fvg_stack`` calls three).
    """
    joined = "|".join(compute_source_hash(fn) for fn in fns)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return {
            "__frame__": True,
            "fingerprint": dataframe_fingerprint(value, strategy="content"),
        }
    if isinstance(value, dict):
        return {str(key): _jsonify(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(child) for child in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def source_input_fingerprints(
    context: GraphRunContext, names: tuple[str, ...]
) -> dict[str, str]:
    payload: dict[str, str] = {}
    for name in names:
        value = context.inputs.get(name)
        if isinstance(value, pd.DataFrame):
            payload[name] = dataframe_fingerprint(value, strategy="content")
        else:
            payload[name] = fingerprint_mapping({"value": _jsonify(value)})
    return payload


def default_node_fingerprint_payload(
    manifest: NodeManifest,
    context: GraphRunContext,
    dependency_results: dict[str, NodeExecutionResult],
) -> dict[str, Any]:
    return {
        "graph_name": manifest.graph_name,
        "node_name": manifest.node_name,
        "node_kind": manifest.node_kind,
        "semantic_class": manifest.semantic_class,
        "schema_version": manifest.schema_version,
        "feature_contract_version": manifest.feature_contract_version,
        "engine_version": manifest.engine_version,
        "config": _jsonify(dict(manifest.config)),
        "runtime_config": _jsonify(context.config),
        "input_fingerprints": source_input_fingerprints(context, manifest.inputs),
        "upstream_fingerprints": {
            name: dependency_results[name].fingerprint
            for name in manifest.upstream_nodes
        },
        "window_policy": _jsonify(asdict(manifest.window_policy)),
        "replay_policy": _jsonify(
            asdict(manifest.replay_policy) if manifest.replay_policy is not None else {}
        ),
    }


def node_fingerprint(
    manifest: NodeManifest,
    context: GraphRunContext,
    dependency_results: dict[str, NodeExecutionResult],
) -> str:
    payload = (
        manifest.fingerprint_fn(manifest, context, dependency_results)
        if manifest.fingerprint_fn is not None
        else default_node_fingerprint_payload(manifest, context, dependency_results)
    )
    return fingerprint_mapping(_jsonify(payload))
