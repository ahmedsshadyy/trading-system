from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.dag_runtime.node import NodeExecutionResult, NodeManifest, NodeOutput
from src.pipeline_runtime import (
    PipelineMetadata,
    read_metadata,
    write_json_atomic,
    write_metadata_atomic,
    write_parquet_atomic,
)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return {
            "__dataframe__": True,
            "rows": int(len(value)),
            "columns": list(value.columns),
        }
    if isinstance(value, dict):
        return {str(key): _jsonify(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(child) for child in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def node_cache_dir(
    cache_root: str | Path,
    *,
    graph_name: str,
    symbol: str,
    timeframe: str,
    node_name: str,
) -> Path:
    return Path(cache_root) / graph_name / symbol / timeframe / node_name


def node_cache_paths(
    cache_root: str | Path,
    *,
    graph_name: str,
    symbol: str,
    timeframe: str,
    node_name: str,
    fingerprint: str,
) -> tuple[Path, Path]:
    base = node_cache_dir(
        cache_root,
        graph_name=graph_name,
        symbol=symbol,
        timeframe=timeframe,
        node_name=node_name,
    )
    return base / f"{fingerprint}.json", base / f"{fingerprint}.meta.json"


def load_cached_node(
    cache_root: str | Path,
    *,
    manifest: NodeManifest,
    symbol: str,
    timeframe: str,
    fingerprint: str,
) -> NodeExecutionResult | None:
    payload_path, metadata_path = node_cache_paths(
        cache_root,
        graph_name=manifest.graph_name,
        symbol=symbol,
        timeframe=timeframe,
        node_name=manifest.node_name,
        fingerprint=fingerprint,
    )
    metadata = read_metadata(metadata_path)
    if metadata is None or not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    frame_paths = payload.get("__frame_paths__", {})
    artifact_paths = {
        name: Path(path) for name, path in payload.get("__artifact_paths__", {}).items()
    }
    for rel_path in frame_paths.values():
        if not (payload_path.parent / rel_path).exists():
            return None
    frames = {
        name: pd.read_parquet(payload_path.parent / rel_path).reset_index(drop=True)
        for name, rel_path in frame_paths.items()
    }
    return NodeExecutionResult(
        manifest=manifest,
        output=NodeOutput(
            payload={
                key: value for key, value in payload.items() if not key.startswith("__")
            },
            frames=frames,
            artifacts=artifact_paths,
            profile_details=payload.get("__profile_details__", {}),
        ),
        fingerprint=fingerprint,
        cache_hit=True,
        cache_path=payload_path,
    )


def save_cached_node(
    cache_root: str | Path,
    *,
    manifest: NodeManifest,
    symbol: str,
    timeframe: str,
    fingerprint: str,
    output: NodeOutput,
    metadata_extra: dict[str, Any] | None = None,
) -> NodeExecutionResult:
    payload_path, metadata_path = node_cache_paths(
        cache_root,
        graph_name=manifest.graph_name,
        symbol=symbol,
        timeframe=timeframe,
        node_name=manifest.node_name,
        fingerprint=fingerprint,
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    frame_paths: dict[str, str] = {}
    for name, frame in output.frames.items():
        frame_path = payload_path.parent / f"{fingerprint}.{name}.parquet"
        write_parquet_atomic(frame.reset_index(drop=True), frame_path)
        frame_paths[name] = frame_path.name
    serializable = {
        **_jsonify(output.payload),
        "__frame_paths__": frame_paths,
        "__artifact_paths__": {
            name: str(path) for name, path in output.artifacts.items()
        },
        "__profile_details__": _jsonify(output.profile_details),
    }
    write_json_atomic(serializable, payload_path)
    metadata = PipelineMetadata(
        symbol=symbol,
        timeframe=timeframe,
        pipeline=f"{manifest.graph_name}:{manifest.node_name}",
        input_fingerprint=fingerprint,
        config_fingerprint=fingerprint,
        schema_version=manifest.schema_version,
        feature_contract_version=manifest.feature_contract_version,
        engine_version=manifest.engine_version,
        extra=metadata_extra or {},
    )
    write_metadata_atomic(metadata_path, metadata)
    return NodeExecutionResult(
        manifest=manifest,
        output=output,
        fingerprint=fingerprint,
        cache_hit=False,
        cache_path=payload_path,
    )


def invalidate_node_cache(
    cache_root: str | Path,
    *,
    graph_name: str,
    symbol: str,
    timeframe: str,
    node_name: str,
) -> list[Path]:
    root = node_cache_dir(
        cache_root,
        graph_name=graph_name,
        symbol=symbol,
        timeframe=timeframe,
        node_name=node_name,
    )
    removed: list[Path] = []
    if not root.exists():
        return removed
    for path in root.glob("*"):
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed
