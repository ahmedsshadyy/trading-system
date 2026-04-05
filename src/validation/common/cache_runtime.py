from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.pipeline_runtime import (
    PipelineRunProfiler,
    PipelineMetadata,
    dataframe_fingerprint,
    fingerprint_mapping,
    read_metadata,
    report_is_current,
    update_report_fingerprint,
    write_json_atomic,
    write_metadata_atomic,
    write_parquet_atomic,
)

DEFAULT_VALIDATION_CACHE_ROOT = Path("data/validation_cache")
VALIDATION_RUNTIME_VERSION = "validation-cache-v1"


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
    if isinstance(value, Mapping):
        return {str(key): _jsonify(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(child) for child in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def validation_cache_key(
    *,
    validator: str,
    symbol: str,
    timeframe: str,
    stage: str,
    input_fingerprint: str,
    config_fingerprint: str,
    upstream_fingerprint: str,
    time_range: str,
    schema_version: int = 1,
    feature_contract_version: int = 1,
    runtime_version: str = VALIDATION_RUNTIME_VERSION,
) -> str:
    return fingerprint_mapping(
        {
            "validator": validator,
            "symbol": symbol,
            "timeframe": timeframe,
            "stage": stage,
            "schema_version": schema_version,
            "feature_contract_version": feature_contract_version,
            "runtime_version": runtime_version,
            "input_fingerprint": input_fingerprint,
            "config_fingerprint": config_fingerprint,
            "upstream_fingerprint": upstream_fingerprint,
            "time_range": time_range,
        }
    )


def validation_cache_dir(
    *,
    validator: str,
    symbol: str,
    timeframe: str,
    stage: str,
    cache_root: str | Path = DEFAULT_VALIDATION_CACHE_ROOT,
) -> Path:
    return Path(cache_root) / validator / symbol / timeframe / stage


def _metadata_for_stage(
    *,
    validator: str,
    symbol: str,
    timeframe: str,
    stage: str,
    cache_key: str,
    input_fingerprint: str,
    config_fingerprint: str,
    extra: dict[str, Any],
    schema_version: int,
    feature_contract_version: int,
) -> PipelineMetadata:
    return PipelineMetadata(
        symbol=symbol,
        timeframe=timeframe,
        pipeline=f"{validator}:{stage}",
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        schema_version=schema_version,
        feature_contract_version=feature_contract_version,
        engine_version=VALIDATION_RUNTIME_VERSION,
        extra={"cache_key": cache_key, **_jsonify(extra)},
    )


@dataclass(slots=True)
class CachedFrameResult:
    frame: pd.DataFrame
    fingerprint: str
    cache_hit: bool
    data_path: Path
    metadata_path: Path


@dataclass(slots=True)
class CachedValidationResult:
    payload: dict[str, Any]
    frames: dict[str, pd.DataFrame]
    fingerprint: str
    cache_hit: bool
    payload_path: Path
    frame_paths: dict[str, Path]
    metadata_path: Path


def load_or_build_stage_artifact(
    *,
    validator: str,
    symbol: str,
    timeframe: str,
    stage: str,
    input_fingerprint: str,
    config_payload: Mapping[str, Any],
    upstream_fingerprint: str,
    time_range: str,
    build_fn: Callable[[], dict[str, Any]],
    cache_root: str | Path = DEFAULT_VALIDATION_CACHE_ROOT,
    schema_version: int = 1,
    feature_contract_version: int = 1,
    invalidate_cache: bool = False,
    profiler: PipelineRunProfiler | None = None,
) -> CachedValidationResult:
    config_fingerprint = fingerprint_mapping(_jsonify(dict(config_payload)))
    cache_key = validation_cache_key(
        validator=validator,
        symbol=symbol,
        timeframe=timeframe,
        stage=stage,
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        upstream_fingerprint=upstream_fingerprint,
        time_range=time_range,
        schema_version=schema_version,
        feature_contract_version=feature_contract_version,
    )
    stage_dir = validation_cache_dir(
        validator=validator,
        symbol=symbol,
        timeframe=timeframe,
        stage=stage,
        cache_root=cache_root,
    )
    payload_path = stage_dir / f"{cache_key}.json"
    metadata_path = stage_dir / f"{cache_key}.meta.json"

    metadata = read_metadata(metadata_path) if not invalidate_cache else None
    if (
        metadata is not None
        and metadata.input_fingerprint == input_fingerprint
        and metadata.config_fingerprint == config_fingerprint
        and payload_path.exists()
    ):
        with payload_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frame_paths = {
            name: stage_dir / rel_path
            for name, rel_path in payload.get("__frame_paths__", {}).items()
        }
        if all(path.exists() for path in frame_paths.values()):
            frames = {
                name: pd.read_parquet(path).reset_index(drop=True)
                for name, path in frame_paths.items()
            }
            if profiler is not None:
                profiler.record_stage(
                    stage,
                    started_at=time.perf_counter(),
                    output_frame=next(iter(frames.values()), None),
                    details={"cache_hit": True},
                )
            return CachedValidationResult(
                payload={
                    key: value
                    for key, value in payload.items()
                    if key != "__frame_paths__"
                },
                frames=frames,
                fingerprint=cache_key,
                cache_hit=True,
                payload_path=payload_path,
                frame_paths=frame_paths,
                metadata_path=metadata_path,
            )

    started_at = time.perf_counter()
    built = build_fn()
    payload = dict(built.get("payload", {}))
    frames = {
        name: frame.reset_index(drop=True)
        for name, frame in built.get("frames", {}).items()
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: dict[str, Path] = {}
    for name, frame in frames.items():
        frame_path = stage_dir / f"{cache_key}.{name}.parquet"
        result = write_parquet_atomic(frame, frame_path)
        frame_paths[name] = frame_path
        if profiler is not None:
            profiler.record_artifact(
                path=result.path,
                rows=result.rows,
                bytes_written=result.bytes_written,
                kind=f"{stage}:frame",
            )
    serializable_payload = {
        **_jsonify(payload),
        "__frame_paths__": {name: path.name for name, path in frame_paths.items()},
    }
    result = write_json_atomic(serializable_payload, payload_path)
    metadata = _metadata_for_stage(
        validator=validator,
        symbol=symbol,
        timeframe=timeframe,
        stage=stage,
        cache_key=cache_key,
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        extra={
            "time_range": time_range,
            "upstream_fingerprint": upstream_fingerprint,
            "payload_path": str(payload_path),
            "frame_paths": {name: str(path) for name, path in frame_paths.items()},
        },
        schema_version=schema_version,
        feature_contract_version=feature_contract_version,
    )
    write_metadata_atomic(metadata_path, metadata)
    if profiler is not None:
        profiler.record_artifact(
            path=result.path,
            bytes_written=result.bytes_written,
            kind=f"{stage}:payload",
        )
        profiler.record_stage(
            stage,
            started_at=started_at,
            output_frame=next(iter(frames.values()), None),
            details={"cache_hit": False},
        )
    return CachedValidationResult(
        payload=payload,
        frames=frames,
        fingerprint=cache_key,
        cache_hit=False,
        payload_path=payload_path,
        frame_paths=frame_paths,
        metadata_path=metadata_path,
    )


def load_or_build_context(
    *,
    validator: str,
    symbol: str,
    timeframe: str,
    input_df: pd.DataFrame,
    config_payload: Mapping[str, Any],
    build_fn: Callable[[], pd.DataFrame],
    cache_root: str | Path = DEFAULT_VALIDATION_CACHE_ROOT,
    schema_version: int = 1,
    feature_contract_version: int = 1,
    invalidate_cache: bool = False,
    profiler: PipelineRunProfiler | None = None,
) -> CachedFrameResult:
    input_fingerprint = dataframe_fingerprint(input_df, strategy="content")
    config_fingerprint = fingerprint_mapping(_jsonify(dict(config_payload)))
    time_range = "full"
    if "timestamp" in input_df.columns and not input_df.empty:
        ts = pd.to_datetime(input_df["timestamp"], utc=True, errors="coerce")
        time_range = f"{ts.iloc[0].isoformat()}:{ts.iloc[-1].isoformat()}"
    cache_key = validation_cache_key(
        validator=validator,
        symbol=symbol,
        timeframe=timeframe,
        stage="context",
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        upstream_fingerprint=input_fingerprint,
        time_range=time_range,
        schema_version=schema_version,
        feature_contract_version=feature_contract_version,
    )
    stage_dir = validation_cache_dir(
        validator=validator,
        symbol=symbol,
        timeframe=timeframe,
        stage="context",
        cache_root=cache_root,
    )
    data_path = stage_dir / f"{cache_key}.parquet"
    metadata_path = stage_dir / f"{cache_key}.meta.json"

    metadata = read_metadata(metadata_path) if not invalidate_cache else None
    if (
        metadata is not None
        and metadata.input_fingerprint == input_fingerprint
        and metadata.config_fingerprint == config_fingerprint
        and data_path.exists()
    ):
        frame = pd.read_parquet(data_path).reset_index(drop=True)
        if profiler is not None:
            profiler.record_stage(
                "context",
                started_at=time.perf_counter(),
                input_frame=input_df,
                output_frame=frame,
                details={"cache_hit": True},
            )
        return CachedFrameResult(
            frame=frame,
            fingerprint=cache_key,
            cache_hit=True,
            data_path=data_path,
            metadata_path=metadata_path,
        )

    started_at = time.perf_counter()
    frame = build_fn().reset_index(drop=True)
    result = write_parquet_atomic(frame, data_path)
    metadata = _metadata_for_stage(
        validator=validator,
        symbol=symbol,
        timeframe=timeframe,
        stage="context",
        cache_key=cache_key,
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        extra={"time_range": time_range, "data_path": str(data_path)},
        schema_version=schema_version,
        feature_contract_version=feature_contract_version,
    )
    write_metadata_atomic(metadata_path, metadata)
    if profiler is not None:
        profiler.record_artifact(
            path=result.path,
            rows=result.rows,
            bytes_written=result.bytes_written,
            kind="context",
        )
        profiler.record_stage(
            "context",
            started_at=started_at,
            input_frame=input_df,
            output_frame=frame,
            details={"cache_hit": False},
        )
    return CachedFrameResult(
        frame=frame,
        fingerprint=cache_key,
        cache_hit=False,
        data_path=data_path,
        metadata_path=metadata_path,
    )


def load_or_build_validation_result(
    *,
    validator: str,
    symbol: str,
    timeframe: str,
    stage: str,
    context_fingerprint: str,
    config_payload: Mapping[str, Any],
    build_fn: Callable[[], tuple[dict[str, Any], dict[str, pd.DataFrame]]],
    cache_root: str | Path = DEFAULT_VALIDATION_CACHE_ROOT,
    schema_version: int = 1,
    feature_contract_version: int = 1,
    invalidate_cache: bool = False,
    profiler: PipelineRunProfiler | None = None,
) -> CachedValidationResult:
    return load_or_build_stage_artifact(
        validator=validator,
        symbol=symbol,
        timeframe=timeframe,
        stage=stage,
        input_fingerprint=context_fingerprint,
        config_payload=config_payload,
        upstream_fingerprint=context_fingerprint,
        time_range="derived",
        build_fn=lambda: _wrap_validation_builder(build_fn),
        cache_root=cache_root,
        schema_version=schema_version,
        feature_contract_version=feature_contract_version,
        invalidate_cache=invalidate_cache,
        profiler=profiler,
    )


def _wrap_validation_builder(
    build_fn: Callable[[], tuple[dict[str, Any], dict[str, pd.DataFrame]]],
) -> dict[str, Any]:
    payload, frames = build_fn()
    return {"payload": payload, "frames": frames}


def load_or_skip_report(
    report_path: str | Path,
    *,
    fingerprint: str,
    force: bool = False,
    writer: Callable[[Path], Path | None],
    profiler: PipelineRunProfiler | None = None,
    kind: str = "report",
) -> tuple[Path | None, bool]:
    report_target = Path(report_path)
    metadata_path = report_target.with_suffix(report_target.suffix + ".meta.json")
    if report_is_current(
        report_path=report_target,
        fingerprint_path=metadata_path,
        fingerprint=fingerprint,
        force=force,
    ):
        return report_target, True

    written_path = writer(report_target)
    if written_path is None:
        return None, False

    update_report_fingerprint(
        report_path=written_path,
        fingerprint_path=metadata_path,
        fingerprint=fingerprint,
    )
    if profiler is not None and written_path.exists():
        profiler.record_artifact(
            path=written_path,
            bytes_written=written_path.stat().st_size,
            kind=kind,
        )
    return written_path, False


def write_csv_atomic(df: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, target)
    return target


def write_text_atomic(text: str, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)
    return target


def cleanup_validation_artifacts(
    *,
    cache_root: str | Path = DEFAULT_VALIDATION_CACHE_ROOT,
    max_age_days: int = 30,
    report_roots: list[str | Path] | None = None,
) -> list[Path]:
    now = time.time()
    cutoff_seconds = max_age_days * 86400
    removed: list[Path] = []

    for root in [Path(cache_root), *(Path(root) for root in (report_roots or []))]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix == ".tmp":
                path.unlink(missing_ok=True)
                removed.append(path)
                continue
            if now - path.stat().st_mtime <= cutoff_seconds:
                continue
            if root == Path(cache_root):
                path.unlink(missing_ok=True)
                removed.append(path)
                continue
            metadata_path = path.with_suffix(path.suffix + ".meta.json")
            if metadata_path.exists():
                path.unlink(missing_ok=True)
                removed.append(path)
                metadata_path.unlink(missing_ok=True)
                removed.append(metadata_path)
    return removed
