import os
from pathlib import Path
import pyarrow.parquet as pq
from sqlalchemy import create_engine
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

engine = create_engine(os.getenv("DATABASE_URL"))

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Map filename prefix to instrument name
INSTRUMENT_MAP = {
    "XAU_USD": "XAU_USD",
    "USOIL": "USOIL",
    "EUR_USD": "EUR_USD",
    "USD_JPY": "USD_JPY",
}

TIMEFRAME_MAP = {
    "D": "D",
    "H4": "H4",
    "H1": "H1",
    "M15": "M15",
}

parquet_files = sorted(DATA_DIR.glob("*.parquet"))

with tqdm(total=len(parquet_files), desc="Loading candles", unit="file") as bar:
    for f in parquet_files:
        # Parse instrument and timeframe from filename e.g. XAU_USD_H4.parquet
        stem = f.stem  # e.g. XAU_USD_H4

        # Find matching instrument
        instrument = None
        timeframe = None
        for inst_key in INSTRUMENT_MAP:
            if stem.startswith(inst_key):
                instrument = INSTRUMENT_MAP[inst_key]
                tf_key = stem[len(inst_key) + 1 :]  # everything after instrument_
                timeframe = TIMEFRAME_MAP.get(tf_key)
                break

        if not instrument or not timeframe:
            tqdm.write(f"Skipping {f.name} — could not parse instrument/timeframe")
            bar.update(1)
            continue

        df = pq.read_table(f).to_pandas()

        # Rename columns to match schema
        df = df.rename(columns={"tickVolume": "volume"})

        # Keep only schema columns
        df = df[
            ["timestamp", "open", "high", "low", "close", "volume", "spread"]
        ].copy()

        # Add instrument and timeframe columns
        df["instrument"] = instrument
        df["timeframe"] = timeframe

        # Drop rows with null volume
        df = df.dropna(subset=["volume"])
        df["volume"] = df["volume"].astype(int)

        # Load to PostgreSQL — skip duplicates via on_conflict
        try:
            df.to_sql(
                "candles",
                engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
            )
            tqdm.write(f"✓ {f.name}: {len(df):,} rows loaded")
        except Exception as e:
            tqdm.write(f"✗ {f.name}: {e}")

        bar.update(1)

print("\nDone.")
