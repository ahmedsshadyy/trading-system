"""Export the first candle per instrument/timeframe from the normalized DB.

Regenerates data/validation/first_rows_normalized.csv from the current
state of the candles table (which already has tickVolume → volume applied).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set in .env")

    engine = create_engine(database_url)

    query = text("""
        SELECT c.*
        FROM candles c
        JOIN (
            SELECT instrument, timeframe, MIN(timestamp) AS min_ts
            FROM candles
            GROUP BY instrument, timeframe
        ) m
            ON c.instrument = m.instrument
            AND c.timeframe = m.timeframe
            AND c.timestamp = m.min_ts
        ORDER BY c.instrument, c.timeframe
    """)

    df = pd.read_sql(query, engine)

    out = ROOT / "data" / "validation" / "first_rows_normalized.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows → {out}")


if __name__ == "__main__":
    main()
