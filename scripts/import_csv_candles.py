from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

START = datetime(2014, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 4, 11, 23, 59, 59, tzinfo=timezone.utc)

TIMEFRAME_MAP = {
    "D1": "D",
    "D": "D",
    "H1": "H1",
    "H4": "H4",
    "M15": "M15",
}

RAW_TIMEFRAME_MAP = {
    "D": "1d",
    "H4": "4h",
    "H1": "1h",
    "M15": "15m",
}

TIMEFRAME_DELTAS = {
    "D": pd.Timedelta(days=1),
    "H4": pd.Timedelta(hours=4),
    "H1": pd.Timedelta(hours=1),
    "M15": pd.Timedelta(minutes=15),
}

INSTRUMENT_ALIASES = {
    "AUDUSD": "AUD_USD",
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "NZDUSD": "NZD_USD",
    "USDCAD": "USD_CAD",
    "USDCHF": "USD_CHF",
    "USDJPY": "USD_JPY",
    "USDSEK": "USD_SEK",
    "XAUUSD": "XAU_USD",
    "USOIL": "USOIL",
}


def _normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def infer_identity(path: Path) -> tuple[str, str]:
    stem = path.stem.upper()
    match = re.match(r"^(?P<instrument>[A-Z0-9]+)_(?P<timeframe>D1|D|H1|H4|M15)$", stem)
    if match is None:
        raise ValueError(f"Could not infer instrument/timeframe from {path.name}")

    instrument_token = _normalize_token(match.group("instrument"))
    canonical = INSTRUMENT_ALIASES.get(instrument_token)
    if canonical is None:
        raise ValueError(
            f"Unsupported instrument token in {path.name}: {instrument_token}"
        )

    timeframe = TIMEFRAME_MAP[match.group("timeframe")]
    return canonical, timeframe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import headerless OHLCV CSV files into project parquet candle format."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing CSV files like USDJPY_D1.csv, USDJPY_H1.csv, USDJPY_H4.csv, USDJPY_M15.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help="Parquet output directory. Defaults to data/raw.",
    )
    parser.add_argument(
        "--start",
        default=START.date().isoformat(),
        help="UTC lower bound. Defaults to 2014-01-01.",
    )
    parser.add_argument(
        "--end",
        default=END.date().isoformat(),
        help="UTC upper bound. Defaults to 2026-04-11.",
    )
    return parser.parse_args(argv)


def parse_cli_datetime(value: str, *, is_end: bool) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    if len(value) == 10 and is_end:
        parsed = parsed + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return parsed.to_pydatetime()


def load_csv_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume"],
        encoding="utf-8-sig",
    )
    return df


def _iso_z(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _broker_time(
    ts: pd.Timestamp, *, offset_hours: int = 2, millisecond: str = "000"
) -> str:
    shifted = ts.tz_convert("UTC") + pd.Timedelta(hours=offset_hours)
    return shifted.strftime("%Y-%m-%d %H:%M:%S") + f".{millisecond}"


def build_raw_frame(
    df: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
    source_file: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    out = out[
        (out["timestamp"] >= pd.Timestamp(start))
        & (out["timestamp"] <= pd.Timestamp(end))
    ]
    out = out.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    if out.empty:
        return out

    raw_timeframe = RAW_TIMEFRAME_MAP[timeframe]
    delta = TIMEFRAME_DELTAS[timeframe]
    end_ts = out["timestamp"] + delta - pd.Timedelta(milliseconds=1)

    raw = pd.DataFrame(
        {
            "symbol": instrument.replace("_", ""),
            "timestamp": out["timestamp"],
            "brokerTime": out["timestamp"].map(
                lambda ts: _broker_time(ts, millisecond="000")
            ),
            "timeframe": raw_timeframe,
            "open": out["open"],
            "close": out["close"],
            "high": out["high"],
            "low": out["low"],
            "tickVolume": out["volume"].astype(int),
            "spread": 0,
            "endTime": end_ts.map(_iso_z),
            "endBrokerTime": end_ts.map(lambda ts: _broker_time(ts, millisecond="999")),
            "state": "complete",
            "source_file": source_file,
        }
    )
    return raw.reset_index(drop=True)


def convert_existing_project_frame_to_raw(
    df: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
    source_file: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    working = df.copy()
    if "volume" not in working.columns and "tickVolume" in working.columns:
        working["volume"] = working["tickVolume"]
    return build_raw_frame(
        working[["timestamp", "open", "high", "low", "close", "volume"]],
        instrument=instrument,
        timeframe=timeframe,
        source_file=source_file,
        start=start,
        end=end,
    )


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path)


def import_directory(
    input_dir: Path,
    *,
    output_dir: Path,
    start: datetime,
    end: datetime,
) -> list[tuple[Path, int]]:
    results: list[tuple[Path, int]] = []
    csv_paths = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise ValueError(f"No CSV files found in {input_dir}")

    for csv_path in csv_paths:
        instrument, timeframe = infer_identity(csv_path)
        raw = build_raw_frame(
            load_csv_frame(csv_path),
            instrument=instrument,
            timeframe=timeframe,
            source_file=csv_path.name,
            start=start,
            end=end,
        )
        out_path = output_dir / f"{instrument}_{timeframe}.parquet"
        write_parquet(raw, out_path)
        results.append((out_path, len(raw)))
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start = parse_cli_datetime(args.start, is_end=False)
    end = parse_cli_datetime(args.end, is_end=True)
    if end < start:
        raise ValueError("End must be on or after start")

    results = import_directory(
        args.input_dir,
        output_dir=args.output_dir,
        start=start,
        end=end,
    )
    for out_path, rows in results:
        print(f"✓ {out_path.name}: {rows:,} rows")


if __name__ == "__main__":
    main()
