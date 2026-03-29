from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PipelineMetadata:
    symbol: str
    timeframe: str
    pipeline: str
    last_processed_ts: str | None = None
    schema_version: int = 1
    feature_contract_version: int = 1
    input_fingerprint: str | None = None
    config_fingerprint: str | None = None
    engine_version: str = "1"
    metadata_version: int = 1
    updated_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.extra:
            payload.pop("extra", None)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineMetadata":
        known = {
            "symbol",
            "timeframe",
            "pipeline",
            "last_processed_ts",
            "schema_version",
            "feature_contract_version",
            "input_fingerprint",
            "config_fingerprint",
            "engine_version",
            "metadata_version",
            "updated_at",
            "extra",
        }
        extra = dict(payload.get("extra", {}))
        for key, value in payload.items():
            if key not in known:
                extra[key] = value
        return cls(
            symbol=payload["symbol"],
            timeframe=payload["timeframe"],
            pipeline=payload["pipeline"],
            last_processed_ts=payload.get("last_processed_ts"),
            schema_version=int(payload.get("schema_version", 1)),
            feature_contract_version=int(payload.get("feature_contract_version", 1)),
            input_fingerprint=payload.get("input_fingerprint"),
            config_fingerprint=payload.get("config_fingerprint"),
            engine_version=str(payload.get("engine_version", "1")),
            metadata_version=int(payload.get("metadata_version", 1)),
            updated_at=payload.get("updated_at"),
            extra=extra,
        )


def read_metadata(path: str | Path) -> PipelineMetadata | None:
    target = Path(path)
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    return PipelineMetadata.from_dict(payload)


def write_metadata_atomic(path: str | Path, metadata: PipelineMetadata) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, target)
    return target


def metadata_path(
    base_dir: str | Path, *, pipeline: str, symbol: str, timeframe: str
) -> Path:
    return Path(base_dir) / pipeline / symbol / timeframe / "metadata.json"
