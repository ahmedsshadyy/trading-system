from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import asdict

import pandas as pd

from src.dag_runtime.node import GraphRunContext, NodeExecutionResult, NodeManifest
from src.pipeline_runtime import dataframe_fingerprint, fingerprint_mapping


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
