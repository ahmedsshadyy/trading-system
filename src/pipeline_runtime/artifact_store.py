from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass(slots=True)
class ArtifactWriteResult:
    path: Path
    rows: int | None
    bytes_written: int
    partition: str | None = None


def _json_safe_frame_attrs(frame: pd.DataFrame) -> dict[str, object]:
    safe_attrs: dict[str, object] = {}
    for key, value in frame.attrs.items():
        if isinstance(value, pd.DataFrame):
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        safe_attrs[str(key)] = value
    return safe_attrs


def _parquet_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy(deep=False)
    safe.attrs = _json_safe_frame_attrs(df)
    return safe


def canonical_dataset_root(
    base_dir: str | Path,
    *,
    dataset: str,
    symbol: str,
    timeframe: str,
) -> Path:
    return Path(base_dir) / dataset / symbol / timeframe


def monthly_partition_label(timestamp: pd.Timestamp | None) -> str:
    if timestamp is None or pd.isna(timestamp):
        return "unknown"
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return f"{ts.year:04d}-{ts.month:02d}"


def partitioned_parquet_path(
    base_dir: str | Path,
    *,
    dataset: str,
    symbol: str,
    timeframe: str,
    partition: str,
) -> Path:
    return (
        canonical_dataset_root(
            base_dir,
            dataset=dataset,
            symbol=symbol,
            timeframe=timeframe,
        )
        / f"{partition}.parquet"
    )


def list_partition_paths(
    base_dir: str | Path,
    *,
    dataset: str,
    symbol: str,
    timeframe: str,
) -> list[Path]:
    root = canonical_dataset_root(
        base_dir,
        dataset=dataset,
        symbol=symbol,
        timeframe=timeframe,
    )
    if not root.exists():
        return []
    return sorted(root.glob("*.parquet"))


def load_partitioned_dataset(
    base_dir: str | Path,
    *,
    dataset: str,
    symbol: str,
    timeframe: str,
    read_observer: Callable[[Path, pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    parts = list_partition_paths(
        base_dir,
        dataset=dataset,
        symbol=symbol,
        timeframe=timeframe,
    )
    if not parts:
        return pd.DataFrame()
    frames = []
    for path in parts:
        frame = pd.read_parquet(path)
        if read_observer is not None:
            read_observer(path, frame)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
    return combined.reset_index(drop=True)


def write_parquet_atomic(df: pd.DataFrame, path: str | Path) -> ArtifactWriteResult:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    _parquet_safe_frame(df).to_parquet(tmp_path, index=False)
    os.replace(tmp_path, target)
    return ArtifactWriteResult(
        path=target,
        rows=int(len(df)),
        bytes_written=target.stat().st_size,
    )


def write_json_atomic(
    payload: dict[str, object], path: str | Path
) -> ArtifactWriteResult:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)
    return ArtifactWriteResult(
        path=target, rows=None, bytes_written=target.stat().st_size
    )


def cleanup_temp_artifacts(base_dir: str | Path) -> list[Path]:
    removed: list[Path] = []
    for path in Path(base_dir).rglob("*.tmp"):
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def persist_partitioned_dataset(
    df: pd.DataFrame,
    *,
    base_dir: str | Path,
    dataset: str,
    symbol: str,
    timeframe: str,
    frontier_from_ts: pd.Timestamp | None,
    full_rebuild: bool = False,
    writer=write_parquet_atomic,
) -> list[ArtifactWriteResult]:
    if df.empty:
        return []

    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["__partition"] = ts.dt.strftime("%Y-%m")
    out = out.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )

    last_ts = ts.iloc[-1]
    frontier_partition = monthly_partition_label(frontier_from_ts or last_ts)
    partitions = sorted(out["__partition"].dropna().unique())
    write_partitions = (
        partitions
        if full_rebuild
        else [partition for partition in partitions if partition >= frontier_partition]
    )

    if full_rebuild:
        existing = {
            path.stem: path
            for path in list_partition_paths(
                base_dir,
                dataset=dataset,
                symbol=symbol,
                timeframe=timeframe,
            )
        }
        for partition, path in existing.items():
            if partition not in partitions:
                path.unlink(missing_ok=True)

    results: list[ArtifactWriteResult] = []
    for partition in write_partitions:
        partition_df = (
            out.loc[out["__partition"] == partition]
            .drop(columns=["__partition"])
            .reset_index(drop=True)
        )
        result = writer(
            partition_df,
            partitioned_parquet_path(
                base_dir,
                dataset=dataset,
                symbol=symbol,
                timeframe=timeframe,
                partition=partition,
            ),
        )
        result.partition = partition
        results.append(result)
    return results
