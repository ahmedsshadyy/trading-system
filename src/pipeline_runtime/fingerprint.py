from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported fingerprint value: {type(value)!r}")


def stable_digest_bytes(payload: bytes, *, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    digest.update(payload)
    return digest.hexdigest()


def stable_digest_text(text: str, *, algorithm: str = "sha256") -> str:
    return stable_digest_bytes(text.encode("utf-8"), algorithm=algorithm)


def fingerprint_mapping(payload: Mapping[str, Any]) -> str:
    return stable_digest_text(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=_json_default
        )
    )


def fingerprint_path(path: str | Path) -> str:
    target = Path(path)
    stat = target.stat()
    return fingerprint_mapping(
        {
            "path": str(target.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )


def dataframe_fingerprint(
    df: pd.DataFrame,
    *,
    include_index: bool = False,
    strategy: str = "cheap",
) -> str:
    if df.empty:
        return fingerprint_mapping(
            {
                "rows": 0,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
        )

    if strategy == "content":
        hashed = (
            pd.util.hash_pandas_object(df, index=include_index).to_numpy().tobytes()
        )
        return stable_digest_bytes(hashed)

    tail_ts = None
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        tail_ts = (
            ts.iloc[-1].isoformat() if not ts.empty and pd.notna(ts.iloc[-1]) else None
        )

    numeric_sum = None
    numeric = df.select_dtypes(include=["number"])
    if not numeric.empty:
        numeric_sum = float(numeric.tail(min(len(numeric), 64)).sum().sum())

    return fingerprint_mapping(
        {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "last_timestamp": tail_ts,
            "numeric_tail_sum": numeric_sum,
        }
    )
