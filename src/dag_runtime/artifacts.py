from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.pipeline_runtime import write_json_atomic, write_parquet_atomic


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


__all__ = [
    "write_csv_atomic",
    "write_json_atomic",
    "write_parquet_atomic",
    "write_text_atomic",
]
