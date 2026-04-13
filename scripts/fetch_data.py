from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from metaapi_cloud_sdk import MetaApi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
AUDIT_DIR = RAW_DIR / "_fetch_audit"
SYMBOL_MAP_PATH = AUDIT_DIR / "metaapi_symbol_map.json"

load_dotenv(ROOT / ".env")

TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")

START = datetime(2014, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 4, 11, 23, 59, 59, tzinfo=timezone.utc)
PAGE_LIMIT = 1000

TIMEFRAMES = ["1d", "4h", "1h", "15m"]
TF_MAP = {
    "1d": "D",
    "4h": "H4",
    "1h": "H1",
    "15m": "M15",
}
TF_DELTAS = {
    "1d": timedelta(days=1),
    "4h": timedelta(hours=4),
    "1h": timedelta(hours=1),
    "15m": timedelta(minutes=15),
}
MAX_ALLOWED_GAP = {
    "D": timedelta(days=35),
    "H4": timedelta(days=35),
    "H1": timedelta(days=35),
    "M15": timedelta(days=35),
}
END_COVERAGE_TOLERANCE = {
    "D": timedelta(days=3),
    "H4": timedelta(days=3),
    "H1": timedelta(days=3),
    "M15": timedelta(days=3),
}

DXY_COMPONENT_WEIGHTS = {
    "EUR_USD": -0.576,
    "USD_JPY": 0.136,
    "GBP_USD": -0.119,
    "USD_CAD": 0.091,
    "USD_SEK": 0.042,
    "USD_CHF": 0.036,
}
DXY_SCALE = 50.14348112


class InstrumentSpec(NamedTuple):
    canonical: str
    aliases: tuple[str, ...]
    synthetic: bool = False


INSTRUMENT_SPECS: tuple[InstrumentSpec, ...] = (
    InstrumentSpec("XAU_USD", ("XAUUSD", "XAU_USD", "GOLD")),
    InstrumentSpec("USOIL", ("USOIL", "WTI", "OIL", "US_OIL")),
    InstrumentSpec("EUR_USD", ("EURUSD", "EUR_USD")),
    InstrumentSpec("GBP_USD", ("GBPUSD", "GBP_USD")),
    InstrumentSpec("USD_JPY", ("USDJPY", "USD_JPY")),
    InstrumentSpec("AUD_USD", ("AUDUSD", "AUD_USD")),
    InstrumentSpec("NZD_USD", ("NZDUSD", "NZD_USD")),
    InstrumentSpec("USD_CAD", ("USDCAD", "USD_CAD")),
    InstrumentSpec("USD_CHF", ("USDCHF", "USD_CHF")),
    InstrumentSpec("USD_SEK", ("USDSEK", "USD_SEK")),
    InstrumentSpec("DXY", ("DXY",), synthetic=True),
)

CLI_TIMEFRAME_MAP = {
    "D": "1d",
    "1D": "1d",
    "1d": "1d",
    "4H": "4h",
    "H4": "4h",
    "4h": "4h",
    "1H": "1h",
    "H1": "1h",
    "1h": "1h",
    "15M": "15m",
    "M15": "15m",
    "15m": "15m",
}


def _ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_symbol_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _symbol_bases(symbol: str) -> tuple[str, ...]:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", symbol.upper()) if part]
    bases = []
    if parts:
        bases.append(parts[0])
    full = _normalize_symbol_token(symbol)
    if full and full not in bases:
        bases.append(full)
    return tuple(bases)


def _symbol_sort_key(symbol: str, alias: str) -> tuple[int, int, int, str]:
    upper = symbol.upper()
    alias_norm = _normalize_symbol_token(alias)
    bases = _symbol_bases(symbol)
    base = bases[0] if bases else ""
    suffix_penalty = 0 if ".SML" in upper else 1
    exact_penalty = 0 if base == alias_norm else 1
    length_penalty = len(symbol)
    return (exact_penalty, suffix_penalty, length_penalty, symbol)


def resolve_broker_symbol(alias_candidates: tuple[str, ...], symbols: list[str]) -> str:
    for alias in alias_candidates:
        alias_norm = _normalize_symbol_token(alias)
        matches = []
        for symbol in symbols:
            bases = _symbol_bases(symbol)
            if not bases:
                continue
            if any(base == alias_norm or base.startswith(alias_norm) for base in bases):
                matches.append(symbol)
        if matches:
            return sorted(matches, key=lambda value: _symbol_sort_key(value, alias))[0]
    joined = ", ".join(alias_candidates)
    raise ValueError(f"Could not resolve broker symbol for aliases: {joined}")


def resolve_broker_symbols(
    instruments: tuple[InstrumentSpec, ...], symbols: list[str]
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for spec in instruments:
        if spec.synthetic:
            continue
        resolved[spec.canonical] = resolve_broker_symbol(spec.aliases, symbols)
    return resolved


def _raw_columns_for_frame(df: pd.DataFrame) -> list[str]:
    preferred = [
        "symbol",
        "timestamp",
        "brokerTime",
        "timeframe",
        "open",
        "close",
        "high",
        "low",
        "tickVolume",
        "volume",
        "spread",
        "endTime",
        "endBrokerTime",
        "state",
    ]
    ordered = [column for column in preferred if column in df.columns]
    extras = [column for column in df.columns if column not in ordered]
    return ordered + extras


def load_existing_raw_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pq.read_table(path).to_pandas()
    if "timestamp" not in df.columns:
        raise ValueError(f"Existing raw file missing timestamp column: {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError(f"Existing raw file contains invalid timestamps: {path}")
    return df.sort_values("timestamp").reset_index(drop=True)


def clip_frame_to_window(
    df: pd.DataFrame,
    *,
    start: datetime = START,
    end: datetime = END,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    clipped = df.copy()
    clipped["timestamp"] = pd.to_datetime(
        clipped["timestamp"], utc=True, errors="coerce"
    )
    clipped = clipped[
        (clipped["timestamp"] >= pd.Timestamp(start))
        & (clipped["timestamp"] <= pd.Timestamp(end))
    ]
    if clipped.empty:
        return pd.DataFrame(columns=df.columns)
    return clipped.sort_values("timestamp").reset_index(drop=True)


def first_timestamp(df: pd.DataFrame) -> pd.Timestamp | None:
    if df.empty:
        return None
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return None
    return ts.iloc[0]


def last_timestamp(df: pd.DataFrame) -> pd.Timestamp | None:
    if df.empty:
        return None
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return None
    return ts.iloc[-1]


def build_fetch_plan(
    *,
    raw_path: Path,
    timeframe: str,
    start: datetime = START,
    end: datetime = END,
) -> dict[str, Any]:
    existing_raw = load_existing_raw_frame(raw_path)
    existing = clip_frame_to_window(existing_raw, start=start, end=end)
    tf_label = TF_MAP[timeframe]
    last_ts = last_timestamp(existing)
    plan: dict[str, Any] = {
        "mode": "full",
        "existing": existing,
        "trimmed": len(existing_raw) != len(existing),
        "fetch_start": start,
        "skip": False,
    }
    if last_ts is None:
        return plan

    if last_ts >= pd.Timestamp(end) - END_COVERAGE_TOLERANCE[tf_label]:
        plan["mode"] = "skip"
        plan["skip"] = True
        plan["fetch_start"] = last_ts.to_pydatetime()
        return plan

    fetch_start = max(pd.Timestamp(start), last_ts - pd.Timedelta(TF_DELTAS[timeframe]))
    plan["mode"] = "incremental"
    plan["fetch_start"] = fetch_start.to_pydatetime()
    return plan


def build_edge_repair_plan(
    *,
    raw_path: Path,
    timeframe: str,
    start: datetime = START,
    end: datetime = END,
) -> dict[str, Any]:
    existing_raw = load_existing_raw_frame(raw_path)
    existing = clip_frame_to_window(existing_raw, start=start, end=end)
    if existing.empty:
        return {
            "mode": "full",
            "existing": existing,
            "trimmed": len(existing_raw) != len(existing),
            "fetch_head": False,
            "fetch_tail": False,
            "head_end": None,
            "tail_start": start,
        }

    first_ts = first_timestamp(existing)
    last_ts = last_timestamp(existing)
    if first_ts is None or last_ts is None:
        return {
            "mode": "full",
            "existing": existing,
            "fetch_head": False,
            "fetch_tail": False,
            "head_end": None,
            "tail_start": start,
        }

    tf_label = TF_MAP[timeframe]
    head_needed = first_ts > pd.Timestamp(start)
    tail_needed = last_ts < pd.Timestamp(end) - END_COVERAGE_TOLERANCE[tf_label]
    tail_start = max(pd.Timestamp(start), last_ts - pd.Timedelta(TF_DELTAS[timeframe]))

    return {
        "mode": "edge-repair",
        "existing": existing,
        "trimmed": len(existing_raw) != len(existing),
        "fetch_head": head_needed,
        "fetch_tail": tail_needed,
        "head_end": first_ts.to_pydatetime(),
        "tail_start": tail_start.to_pydatetime(),
    }


def normalize_fetched_candles(
    candles: list[dict[str, Any]],
    *,
    symbol: str,
    timeframe: str,
    start: datetime = START,
    end: datetime = END,
) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles).copy()
    if "time" not in df.columns:
        raise ValueError("MetaApi response missing 'time' field")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().any():
        bad_idx = int(df["time"].isna().idxmax())
        raise ValueError(f"MetaApi response contains invalid time at row {bad_idx}")

    df = df.rename(columns={"time": "timestamp"})
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df[
        (df["timestamp"] >= pd.Timestamp(start))
        & (df["timestamp"] <= pd.Timestamp(end))
    ]
    if df.empty:
        return df

    if "symbol" not in df.columns:
        df.insert(0, "symbol", symbol)
    else:
        df["symbol"] = df["symbol"].fillna(symbol)
    if "timeframe" not in df.columns:
        df["timeframe"] = timeframe

    return df[_raw_columns_for_frame(df)].reset_index(drop=True)


def merge_raw_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = incoming.copy()
    elif incoming.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, incoming], ignore_index=True, sort=False)
    if merged.empty:
        return merged
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    merged = merged.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    return merged[_raw_columns_for_frame(merged)].reset_index(drop=True)


def _iso_z(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _broker_time(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S.000")


def synthesize_dxy_frame(
    component_frames: dict[str, pd.DataFrame],
    *,
    timeframe: str,
) -> pd.DataFrame:
    required = tuple(DXY_COMPONENT_WEIGHTS)
    missing = [name for name in required if name not in component_frames]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing DXY components for {timeframe}: {missing_list}")

    merged: pd.DataFrame | None = None
    for instrument in required:
        base = component_frames[instrument].copy()
        scoped = base[["timestamp", "open", "high", "low", "close"]].rename(
            columns={
                "open": f"{instrument}_open",
                "high": f"{instrument}_high",
                "low": f"{instrument}_low",
                "close": f"{instrument}_close",
            }
        )
        merged = (
            scoped
            if merged is None
            else merged.merge(scoped, on="timestamp", how="inner")
        )

    if merged is None or merged.empty:
        raise ValueError(
            f"No overlapping timestamps available to synthesize DXY {timeframe}"
        )

    out = pd.DataFrame({"timestamp": merged["timestamp"]})
    for price_field in ("open", "close"):
        value = pd.Series(DXY_SCALE, index=merged.index, dtype=float)
        for instrument, weight in DXY_COMPONENT_WEIGHTS.items():
            value = value * pd.to_numeric(
                merged[f"{instrument}_{price_field}"], errors="coerce"
            ).pow(weight)
        out[price_field] = value

    high_value = pd.Series(DXY_SCALE, index=merged.index, dtype=float)
    low_value = pd.Series(DXY_SCALE, index=merged.index, dtype=float)
    for instrument, weight in DXY_COMPONENT_WEIGHTS.items():
        high_source = "high" if weight > 0 else "low"
        low_source = "low" if weight > 0 else "high"
        high_value = high_value * pd.to_numeric(
            merged[f"{instrument}_{high_source}"], errors="coerce"
        ).pow(weight)
        low_value = low_value * pd.to_numeric(
            merged[f"{instrument}_{low_source}"], errors="coerce"
        ).pow(weight)
    out["high"] = high_value
    out["low"] = low_value
    out["tickVolume"] = 0
    out["volume"] = 0
    out["spread"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out["symbol"] = "DXY"
    out["brokerTime"] = out["timestamp"].map(_broker_time)
    out["timeframe"] = timeframe
    delta = TF_DELTAS[timeframe]
    end_ts = out["timestamp"] + delta - timedelta(milliseconds=1)
    out["endTime"] = end_ts.map(_iso_z)
    out["endBrokerTime"] = end_ts.map(_broker_time)
    out["state"] = "complete"
    return out[_raw_columns_for_frame(out)].reset_index(drop=True)


def audit_candle_frame(
    df: pd.DataFrame,
    *,
    instrument: str,
    timeframe_label: str,
    start: datetime = START,
    end: datetime = END,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "instrument": instrument,
        "timeframe": timeframe_label,
        "status": "pass",
        "row_count": 0,
        "duplicate_timestamps": 0,
        "largest_gap": None,
        "largest_gap_seconds": None,
        "start_coverage_ok": False,
        "end_coverage_ok": False,
        "messages": [],
    }

    if df.empty:
        audit["status"] = "fail"
        audit["messages"].append("No candles returned after clipping")
        return audit

    ts = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .sort_values()
        .reset_index(drop=True)
    )
    if ts.isna().any():
        audit["status"] = "fail"
        audit["messages"].append("Invalid timestamps present in frame")
        return audit

    min_ts = ts.iloc[0]
    max_ts = ts.iloc[-1]
    duplicate_count = int(ts.duplicated().sum())
    gap_series = ts.diff().dropna()
    largest_gap = gap_series.max() if not gap_series.empty else pd.Timedelta(0)

    audit["row_count"] = int(len(df))
    audit["duplicate_timestamps"] = duplicate_count
    audit["min_timestamp"] = min_ts.isoformat()
    audit["max_timestamp"] = max_ts.isoformat()
    audit["largest_gap"] = str(largest_gap)
    audit["largest_gap_seconds"] = float(largest_gap.total_seconds())
    expected_start = pd.Timestamp(start)
    start_tolerance = pd.Timedelta(
        TF_DELTAS.get(
            {"D": "1d", "H4": "4h", "H1": "1h", "M15": "15m"}[timeframe_label],
            timedelta(0),
        )
    )
    audit["start_coverage_ok"] = (
        min_ts <= expected_start + start_tolerance and min_ts >= expected_start
    )
    audit["end_coverage_ok"] = (
        max_ts >= pd.Timestamp(end) - END_COVERAGE_TOLERANCE[timeframe_label]
    )

    if duplicate_count:
        audit["status"] = "fail"
        audit["messages"].append(f"Found {duplicate_count} duplicate timestamps")

    if min_ts < pd.Timestamp(start):
        audit["status"] = "fail"
        audit["messages"].append(
            f"Data starts before configured lower bound: {min_ts.isoformat()}"
        )

    if max_ts > pd.Timestamp(end):
        audit["status"] = "fail"
        audit["messages"].append(
            f"Data ends after configured upper bound: {max_ts.isoformat()}"
        )

    if expected_start == pd.Timestamp(START) and min_ts.year != start.year:
        audit["status"] = "fail"
        audit["messages"].append(
            f"Expected data to begin in {start.year}; got {min_ts.isoformat()}"
        )

    if largest_gap > MAX_ALLOWED_GAP[timeframe_label]:
        audit["status"] = "fail"
        audit["messages"].append(f"Interior gap too large: {largest_gap}")

    if not audit["end_coverage_ok"]:
        audit["status"] = "fail"
        audit["messages"].append(
            f"Latest candle {max_ts.isoformat()} does not reach the {end.date()} target window"
        )

    return audit


async def fetch_candles_for_symbol(
    account: Any,
    *,
    symbol: str,
    timeframe: str,
    start: datetime = START,
    end: datetime = END,
    limit: int = PAGE_LIMIT,
    progress: tqdm | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_candles: list[dict[str, Any]] = []
    cursor = end
    previous_oldest: pd.Timestamp | None = None

    while True:
        candles = await account.get_historical_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_time=cursor,
            limit=limit,
        )
        if not candles:
            break

        batch = pd.DataFrame(candles).copy()
        batch["time"] = pd.to_datetime(batch["time"], utc=True, errors="coerce")
        batch = batch.sort_values("time").reset_index(drop=True)
        oldest = batch["time"].iloc[0]
        newest = batch["time"].iloc[-1]

        if previous_oldest is not None and oldest >= previous_oldest:
            raise RuntimeError(
                f"Pagination stalled for {symbol} {timeframe}: oldest candle repeated at {oldest.isoformat()}"
            )

        all_candles.extend(batch.to_dict("records"))

        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                {
                    "candles": f"{len(all_candles):,}",
                    "from": str(oldest)[:10],
                    "to": str(newest)[:10],
                }
            )

        if oldest <= pd.Timestamp(start) or len(batch) < limit:
            break

        previous_oldest = oldest
        cursor = (oldest - TF_DELTAS[timeframe]).to_pydatetime()

    normalized = normalize_fetched_candles(
        all_candles,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
    )
    audit = audit_candle_frame(
        normalized,
        instrument=symbol,
        timeframe_label=TF_MAP[timeframe],
        start=start,
        end=end,
    )
    return normalized, audit


def audit_existing_coverage(
    df: pd.DataFrame,
    *,
    instrument: str,
    timeframe_label: str,
    end: datetime = END,
) -> dict[str, Any]:
    if df.empty:
        return {
            "instrument": instrument,
            "timeframe": timeframe_label,
            "status": "fail",
            "messages": ["Existing raw file is empty"],
        }
    ts = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .sort_values()
        .reset_index(drop=True)
    )
    max_ts = ts.iloc[-1]
    return {
        "instrument": instrument,
        "timeframe": timeframe_label,
        "status": "pass",
        "messages": ["Existing raw file already covers target window; skipped refetch"],
        "row_count": int(len(df)),
        "min_timestamp": ts.iloc[0].isoformat(),
        "max_timestamp": max_ts.isoformat(),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "largest_gap": None,
        "largest_gap_seconds": None,
        "start_coverage_ok": True,
        "end_coverage_ok": max_ts
        >= pd.Timestamp(end) - END_COVERAGE_TOLERANCE[timeframe_label],
    }


def audit_boundary_coverage(
    df: pd.DataFrame,
    *,
    instrument: str,
    timeframe_label: str,
    start: datetime = START,
    end: datetime = END,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "instrument": instrument,
        "timeframe": timeframe_label,
        "status": "pass",
        "row_count": 0,
        "duplicate_timestamps": 0,
        "largest_gap": None,
        "largest_gap_seconds": None,
        "start_coverage_ok": False,
        "end_coverage_ok": False,
        "messages": [],
    }

    if df.empty:
        audit["status"] = "fail"
        audit["messages"].append("No candles available after merge")
        return audit

    ts = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .sort_values()
        .reset_index(drop=True)
    )
    if ts.isna().any():
        audit["status"] = "fail"
        audit["messages"].append("Invalid timestamps present in frame")
        return audit

    min_ts = ts.iloc[0]
    max_ts = ts.iloc[-1]
    duplicate_count = int(ts.duplicated().sum())
    gap_series = ts.diff().dropna()
    largest_gap = gap_series.max() if not gap_series.empty else pd.Timedelta(0)
    expected_start = pd.Timestamp(start)
    expected_end = pd.Timestamp(end)
    start_tolerance = pd.Timedelta(
        TF_DELTAS[{"D": "1d", "H4": "4h", "H1": "1h", "M15": "15m"}[timeframe_label]]
    )

    audit["row_count"] = int(len(df))
    audit["duplicate_timestamps"] = duplicate_count
    audit["min_timestamp"] = min_ts.isoformat()
    audit["max_timestamp"] = max_ts.isoformat()
    audit["largest_gap"] = str(largest_gap)
    audit["largest_gap_seconds"] = float(largest_gap.total_seconds())
    audit["start_coverage_ok"] = (
        min_ts <= expected_start + start_tolerance and min_ts >= expected_start
    )
    audit["end_coverage_ok"] = (
        max_ts >= expected_end - END_COVERAGE_TOLERANCE[timeframe_label]
    )

    if duplicate_count:
        audit["status"] = "fail"
        audit["messages"].append(f"Found {duplicate_count} duplicate timestamps")
    if min_ts < expected_start:
        audit["status"] = "fail"
        audit["messages"].append(
            f"Data starts before configured lower bound: {min_ts.isoformat()}"
        )
    if max_ts > expected_end:
        audit["status"] = "fail"
        audit["messages"].append(
            f"Data ends after configured upper bound: {max_ts.isoformat()}"
        )
    if not audit["start_coverage_ok"]:
        audit["status"] = "fail"
        audit["messages"].append(
            f"Earliest candle {min_ts.isoformat()} does not align to the {expected_start.date()} start"
        )
    if not audit["end_coverage_ok"]:
        audit["status"] = "fail"
        audit["messages"].append(
            f"Latest candle {max_ts.isoformat()} does not reach the {expected_end.date()} target window"
        )
    if largest_gap > MAX_ALLOWED_GAP[timeframe_label]:
        audit["messages"].append(
            f"Interior gap retained in existing history: {largest_gap}"
        )

    return audit


def write_raw_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df), path)


def write_audit_summary(audit: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(audit, indent=2, sort_keys=True))


def select_instrument_specs(values: list[str] | None) -> tuple[InstrumentSpec, ...]:
    if not values:
        return INSTRUMENT_SPECS

    selected: list[InstrumentSpec] = []
    seen: set[str] = set()
    for value in values:
        token = _normalize_symbol_token(value)
        match = next(
            (
                spec
                for spec in INSTRUMENT_SPECS
                if token == _normalize_symbol_token(spec.canonical)
                or any(
                    token == _normalize_symbol_token(alias) for alias in spec.aliases
                )
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Unknown instrument selection: {value}")
        if match.canonical not in seen:
            selected.append(match)
            seen.add(match.canonical)
    return tuple(selected)


def select_timeframes(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return tuple(TIMEFRAMES)

    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = CLI_TIMEFRAME_MAP.get(value)
        if normalized is None:
            raise ValueError(f"Unsupported timeframe selection: {value}")
        if normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
    return tuple(selected)


def parse_cli_datetime(value: str, *, is_end: bool) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    if len(value) == 10 and is_end:
        parsed = parsed + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return parsed.to_pydatetime()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and repair raw MetaApi candle history."
    )
    parser.add_argument(
        "--instrument",
        action="append",
        dest="instruments",
        help="Canonical instrument or alias. Repeat for multiple values.",
    )
    parser.add_argument(
        "--timeframe",
        action="append",
        dest="timeframes",
        help="Timeframe to fetch. Accepts D, H4, H1, M15, 1d, 4h, 1h, 15m.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "full", "edge-repair"),
        default="auto",
        help="Fetch mode. auto=skip/current append, full=rebuild, edge-repair=repair start and end only.",
    )
    parser.add_argument(
        "--start",
        default=START.date().isoformat(),
        help="UTC lower bound. Date or ISO datetime. Defaults to 2014-01-01.",
    )
    parser.add_argument(
        "--end",
        default=END.date().isoformat(),
        help="UTC upper bound. Date or ISO datetime. Defaults to 2026-04-11.",
    )
    parser.add_argument(
        "--build-dxy",
        action="store_true",
        help="Build synthetic DXY for the selected timeframes after fetch.",
    )
    return parser.parse_args(argv)


async def fetch(
    *,
    instruments: tuple[InstrumentSpec, ...] = INSTRUMENT_SPECS,
    timeframes: tuple[str, ...] = tuple(TIMEFRAMES),
    start: datetime = START,
    end: datetime = END,
    mode: str = "auto",
    build_dxy: bool = True,
) -> dict[str, Any]:
    _ensure_data_dirs()

    if not TOKEN or not ACCOUNT_ID:
        raise RuntimeError("METAAPI_TOKEN and METAAPI_ACCOUNT_ID must be set in .env")

    api = MetaApi(TOKEN)
    summary: dict[str, Any] = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "mode": mode,
        "timeframes": [TF_MAP[timeframe] for timeframe in timeframes],
        "instruments": [spec.canonical for spec in instruments],
    }
    try:
        print("Connecting to MetaApi...")
        account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

        if account.state != "DEPLOYED":
            print("Deploying account...")
            await account.deploy()

        await account.wait_connected()
        print("Connected.\n")

        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()

        print("Resolving broker symbols...")
        symbols = await connection.get_symbols()
        resolved = resolve_broker_symbols(instruments, symbols)
        SYMBOL_MAP_PATH.write_text(json.dumps(resolved, indent=2, sort_keys=True))
        print(json.dumps(resolved, indent=2, sort_keys=True))
        print()

        await connection.close()

        component_cache: dict[str, dict[str, pd.DataFrame]] = {}
        non_synthetic = [spec for spec in instruments if not spec.synthetic]
        total_tasks = len(non_synthetic) * len(timeframes)

        with tqdm(
            total=total_tasks, desc="Fetching history", unit="file"
        ) as overall_bar:
            for spec in non_synthetic:
                broker_symbol = resolved[spec.canonical]
                component_cache[spec.canonical] = {}

                for timeframe in timeframes:
                    tf_label = TF_MAP[timeframe]
                    overall_bar.set_description(f"Fetching {broker_symbol} {tf_label}")
                    audit_path = AUDIT_DIR / f"{spec.canonical}_{tf_label}.json"
                    raw_path = RAW_DIR / f"{spec.canonical}_{tf_label}.parquet"
                    existing = load_existing_raw_frame(raw_path)

                    try:
                        if mode == "auto":
                            plan = build_fetch_plan(
                                raw_path=raw_path,
                                timeframe=timeframe,
                                start=start,
                                end=end,
                            )
                            existing = plan["existing"]
                            if plan["skip"]:
                                if plan["trimmed"]:
                                    write_raw_parquet(existing, raw_path)
                                audit = audit_existing_coverage(
                                    existing,
                                    instrument=spec.canonical,
                                    timeframe_label=tf_label,
                                    end=end,
                                )
                                audit["canonical_instrument"] = spec.canonical
                                audit["broker_symbol"] = broker_symbol
                                audit["fetch_mode"] = plan["mode"]
                                if plan["trimmed"]:
                                    audit["messages"].append(
                                        f"Trimmed candles outside configured window before {start.date()}"
                                    )
                                write_audit_summary(audit, audit_path)
                                component_cache[spec.canonical][tf_label] = existing
                                tqdm.write(
                                    f"• {spec.canonical} {tf_label}: already current through target window"
                                )
                                overall_bar.update(1)
                                continue

                            with tqdm(
                                desc=f"  {broker_symbol} {tf_label}",
                                leave=False,
                                unit="chunk",
                            ) as chunk_bar:
                                frame, audit = await fetch_candles_for_symbol(
                                    account,
                                    symbol=broker_symbol,
                                    timeframe=timeframe,
                                    start=plan["fetch_start"],
                                    end=end,
                                    progress=chunk_bar,
                                )
                            audit["canonical_instrument"] = spec.canonical
                            audit["broker_symbol"] = broker_symbol
                            audit["fetch_mode"] = plan["mode"]
                            audit["fetch_start"] = pd.Timestamp(
                                plan["fetch_start"]
                            ).isoformat()
                            write_audit_summary(audit, audit_path)

                            if audit["status"] != "pass":
                                tqdm.write(
                                    f"✗ {spec.canonical} {tf_label}: audit failed"
                                )
                                for message in audit["messages"]:
                                    tqdm.write(f"  - {message}")
                            else:
                                merged = merge_raw_frames(existing, frame)
                                write_raw_parquet(merged, raw_path)
                                component_cache[spec.canonical][tf_label] = merged
                                tqdm.write(
                                    f"✓ {spec.canonical} {tf_label}: {len(merged):,} candles → {raw_path}"
                                )
                        elif mode == "full":
                            with tqdm(
                                desc=f"  {broker_symbol} {tf_label}",
                                leave=False,
                                unit="chunk",
                            ) as chunk_bar:
                                frame, audit = await fetch_candles_for_symbol(
                                    account,
                                    symbol=broker_symbol,
                                    timeframe=timeframe,
                                    start=start,
                                    end=end,
                                    progress=chunk_bar,
                                )
                            audit["canonical_instrument"] = spec.canonical
                            audit["broker_symbol"] = broker_symbol
                            audit["fetch_mode"] = mode
                            audit["fetch_start"] = pd.Timestamp(start).isoformat()
                            write_audit_summary(audit, audit_path)

                            if audit["status"] != "pass":
                                tqdm.write(
                                    f"✗ {spec.canonical} {tf_label}: audit failed"
                                )
                                for message in audit["messages"]:
                                    tqdm.write(f"  - {message}")
                            else:
                                write_raw_parquet(frame, raw_path)
                                component_cache[spec.canonical][tf_label] = frame
                                tqdm.write(
                                    f"✓ {spec.canonical} {tf_label}: {len(frame):,} candles → {raw_path}"
                                )
                        elif mode == "edge-repair":
                            plan = build_edge_repair_plan(
                                raw_path=raw_path,
                                timeframe=timeframe,
                                start=start,
                                end=end,
                            )
                            existing = plan["existing"]
                            if plan["mode"] == "full":
                                with tqdm(
                                    desc=f"  {broker_symbol} {tf_label}",
                                    leave=False,
                                    unit="chunk",
                                ) as chunk_bar:
                                    frame, audit = await fetch_candles_for_symbol(
                                        account,
                                        symbol=broker_symbol,
                                        timeframe=timeframe,
                                        start=start,
                                        end=end,
                                        progress=chunk_bar,
                                    )
                                audit["canonical_instrument"] = spec.canonical
                                audit["broker_symbol"] = broker_symbol
                                audit["fetch_mode"] = "full"
                                write_audit_summary(audit, audit_path)
                                if audit["status"] != "pass":
                                    tqdm.write(
                                        f"✗ {spec.canonical} {tf_label}: audit failed"
                                    )
                                    for message in audit["messages"]:
                                        tqdm.write(f"  - {message}")
                                else:
                                    write_raw_parquet(frame, raw_path)
                                    component_cache[spec.canonical][tf_label] = frame
                                    tqdm.write(
                                        f"✓ {spec.canonical} {tf_label}: {len(frame):,} candles → {raw_path}"
                                    )
                            else:
                                head_frame = pd.DataFrame()
                                tail_frame = pd.DataFrame()
                                fetch_messages: list[str] = []

                                if plan["fetch_head"]:
                                    with tqdm(
                                        desc=f"  {broker_symbol} {tf_label} head",
                                        leave=False,
                                        unit="chunk",
                                    ) as chunk_bar:
                                        head_frame, head_audit = (
                                            await fetch_candles_for_symbol(
                                                account,
                                                symbol=broker_symbol,
                                                timeframe=timeframe,
                                                start=start,
                                                end=plan["head_end"],
                                                progress=chunk_bar,
                                            )
                                        )
                                    if head_audit["status"] != "pass":
                                        raise RuntimeError(
                                            f"Head repair failed: {'; '.join(head_audit['messages'])}"
                                        )
                                    fetch_messages.append(
                                        f"Head repaired through {pd.Timestamp(plan['head_end']).isoformat()}"
                                    )

                                if plan["fetch_tail"]:
                                    with tqdm(
                                        desc=f"  {broker_symbol} {tf_label} tail",
                                        leave=False,
                                        unit="chunk",
                                    ) as chunk_bar:
                                        tail_frame, tail_audit = (
                                            await fetch_candles_for_symbol(
                                                account,
                                                symbol=broker_symbol,
                                                timeframe=timeframe,
                                                start=plan["tail_start"],
                                                end=end,
                                                progress=chunk_bar,
                                            )
                                        )
                                    if tail_audit["status"] != "pass":
                                        raise RuntimeError(
                                            f"Tail repair failed: {'; '.join(tail_audit['messages'])}"
                                        )
                                    fetch_messages.append(
                                        f"Tail repaired from {pd.Timestamp(plan['tail_start']).isoformat()}"
                                    )

                                merged = merge_raw_frames(existing, head_frame)
                                merged = merge_raw_frames(merged, tail_frame)
                                audit = audit_boundary_coverage(
                                    merged,
                                    instrument=spec.canonical,
                                    timeframe_label=tf_label,
                                    start=start,
                                    end=end,
                                )
                                audit["canonical_instrument"] = spec.canonical
                                audit["broker_symbol"] = broker_symbol
                                audit["fetch_mode"] = mode
                                audit["messages"] = fetch_messages + audit["messages"]
                                write_audit_summary(audit, audit_path)

                                if audit["status"] != "pass":
                                    tqdm.write(
                                        f"✗ {spec.canonical} {tf_label}: edge repair failed"
                                    )
                                    for message in audit["messages"]:
                                        tqdm.write(f"  - {message}")
                                else:
                                    write_raw_parquet(merged, raw_path)
                                    component_cache[spec.canonical][tf_label] = merged
                                    tqdm.write(
                                        f"✓ {spec.canonical} {tf_label}: {len(merged):,} candles → {raw_path}"
                                    )
                        else:
                            raise ValueError(f"Unsupported fetch mode: {mode}")
                    except Exception as exc:
                        failure = {
                            "instrument": spec.canonical,
                            "broker_symbol": broker_symbol,
                            "timeframe": tf_label,
                            "status": "fail",
                            "fetch_mode": mode,
                            "messages": [str(exc)],
                        }
                        write_audit_summary(failure, audit_path)
                        tqdm.write(f"✗ {spec.canonical} {tf_label}: {exc}")

                    overall_bar.update(1)

        if build_dxy:
            for timeframe in timeframes:
                tf_label = TF_MAP[timeframe]
                dxy_components = {
                    name: frames[tf_label]
                    for name, frames in component_cache.items()
                    if name in DXY_COMPONENT_WEIGHTS and tf_label in frames
                }
                audit_path = AUDIT_DIR / f"DXY_{tf_label}.json"
                raw_path = RAW_DIR / f"DXY_{tf_label}.parquet"
                try:
                    if len(dxy_components) != len(DXY_COMPONENT_WEIGHTS):
                        missing = sorted(
                            set(DXY_COMPONENT_WEIGHTS) - set(dxy_components)
                        )
                        raise ValueError(
                            f"Cannot build DXY {tf_label}; missing components: {missing}"
                        )

                    dxy_frame = synthesize_dxy_frame(
                        dxy_components, timeframe=timeframe
                    )
                    audit = audit_candle_frame(
                        dxy_frame,
                        instrument="DXY",
                        timeframe_label=tf_label,
                        start=start,
                        end=end,
                    )
                    write_audit_summary(audit, audit_path)
                    if audit["status"] != "pass":
                        tqdm.write(f"✗ DXY {tf_label}: audit failed")
                        for message in audit["messages"]:
                            tqdm.write(f"  - {message}")
                        continue

                    write_raw_parquet(dxy_frame, raw_path)
                    tqdm.write(
                        f"✓ DXY {tf_label}: {len(dxy_frame):,} candles → {raw_path}"
                    )
                except Exception as exc:
                    failure = {
                        "instrument": "DXY",
                        "timeframe": tf_label,
                        "status": "fail",
                        "messages": [str(exc)],
                    }
                    write_audit_summary(failure, audit_path)
                    tqdm.write(f"✗ DXY {tf_label}: {exc}")

        print("\nDone. UNDEPLOY your MetaApi account now!")
        summary["status"] = "ok"
        return summary
    finally:
        api.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    instruments = select_instrument_specs(args.instruments)
    timeframes = select_timeframes(args.timeframes)
    start = parse_cli_datetime(args.start, is_end=False)
    end = parse_cli_datetime(args.end, is_end=True)
    build_dxy = args.build_dxy or not args.instruments

    if end < start:
        raise ValueError("End must be on or after start")

    asyncio.run(
        fetch(
            instruments=instruments,
            timeframes=timeframes,
            start=start,
            end=end,
            mode=args.mode,
            build_dxy=build_dxy,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
