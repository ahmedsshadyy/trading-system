from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.indicators._helpers.schema import normalize_candle_schema

DATA_DIR = ROOT / "data" / "raw"
TIMEFRAME_MAP = {"D": "D", "H4": "H4", "H1": "H1", "M15": "M15"}


def parse_parquet_identity(path: Path) -> tuple[str, str]:
    stem = path.stem
    instrument, separator, tf_key = stem.rpartition("_")
    if not separator or not instrument:
        raise ValueError(f"Could not parse instrument/timeframe from {path.name}")

    timeframe = TIMEFRAME_MAP.get(tf_key)
    if timeframe is None:
        raise ValueError(f"Unsupported timeframe suffix in {path.name}")

    return instrument, timeframe


def prepare_candle_frame(raw_path: Path, *, instrument: str, timeframe: str):
    df = pq.read_table(raw_path).to_pandas()
    df = normalize_candle_schema(df, require_volume=True)
    out = df[["timestamp", "open", "high", "low", "close", "volume", "spread"]].copy()
    out["instrument"] = instrument
    out["timeframe"] = timeframe
    out = out.dropna(subset=["volume"])
    out["volume"] = out["volume"].astype(int)
    return out


def select_parquet_files(
    *,
    data_dir: Path = DATA_DIR,
    instruments: set[str] | None = None,
) -> list[Path]:
    parquet_files: list[Path] = []
    for path in sorted(
        path
        for path in data_dir.glob("*.parquet")
        if path.is_file() and "_fetch_audit" not in path.parts
    ):
        instrument, _ = parse_parquet_identity(path)
        if instruments is not None and instrument not in instruments:
            continue
        parquet_files.append(path)
    return parquet_files


def load_all_candles(
    *, data_dir: Path = DATA_DIR, instruments: set[str] | None = None
) -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set in .env")

    engine = create_engine(database_url)
    parquet_files = select_parquet_files(data_dir=data_dir, instruments=instruments)

    with tqdm(total=len(parquet_files), desc="Loading candles", unit="file") as bar:
        for path in parquet_files:
            try:
                instrument, timeframe = parse_parquet_identity(path)
                frame = prepare_candle_frame(
                    path, instrument=instrument, timeframe=timeframe
                )

                with engine.begin() as connection:
                    connection.execute(
                        text("""
                            DELETE FROM candles
                            WHERE instrument = :instrument AND timeframe = :timeframe
                            """),
                        {"instrument": instrument, "timeframe": timeframe},
                    )
                    frame.to_sql(
                        "candles",
                        connection,
                        if_exists="append",
                        index=False,
                        method="multi",
                        chunksize=1000,
                    )
                tqdm.write(f"✓ {path.name}: {len(frame):,} rows loaded")
            except Exception as exc:
                tqdm.write(f"✗ {path.name}: {exc}")
            bar.update(1)

    print("\nDone.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load parquet candle files into the candles table."
    )
    parser.add_argument(
        "--instrument",
        action="append",
        dest="instruments",
        help="Canonical instrument name to load. Repeat for multiple instruments.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    instruments = set(args.instruments) if args.instruments else None
    load_all_candles(instruments=instruments)


if __name__ == "__main__":
    main(sys.argv[1:])
