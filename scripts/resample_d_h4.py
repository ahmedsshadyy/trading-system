"""Resample D and H4 candles from H1 data using NY-session-aligned boundaries.

The broker-provided D and H4 parquets use inconsistent timezone conventions
(DST shifts, different grids per symbol).  This script rebuilds them from
the already-aligned H1 data using explicit session assignment in
America/New_York so that DST is handled correctly.

Session boundaries (in ET, i.e. America/New_York):
  Forex  (EUR_USD, USD_JPY, GBP_USD, USD_CAD, USD_SEK, USD_CHF, AUD_USD, NZD_USD):
    D  session start: 17:00 ET  (= NY close)
    H4 grid:         17:00 / 21:00 / 01:00 / 05:00 / 09:00 / 13:00 ET

  Commodities (USOIL, XAU_USD):
    D  session start: 18:00 ET  (1 h after forex — daily break)
    H4 grid:         18:00 / 22:00 / 02:00 / 06:00 / 10:00 / 14:00 ET
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_data import write_raw_parquet, _broker_time, _iso_z

RAW_DIR = ROOT / "data" / "raw"
NY = "America/New_York"

FOREX = (
    "EUR_USD",
    "USD_JPY",
    "GBP_USD",
    "USD_CAD",
    "USD_SEK",
    "USD_CHF",
    "AUD_USD",
    "NZD_USD",
)
COMMODITIES = ("USOIL", "XAU_USD")


def _assign_session_start(
    ts_utc_index: pd.DatetimeIndex,
    anchor_hour_et: int,
    freq_hours: int,
) -> pd.DatetimeIndex:
    """Assign each UTC timestamp to its session-start time (returned in UTC).

    Works correctly across DST transitions because all arithmetic is done
    in America/New_York local time, then converted back to UTC.
    """
    ts_ny = ts_utc_index.tz_convert(NY)
    hour = np.array(ts_ny.hour)

    h_since_anchor = (hour - anchor_hour_et) % 24
    bin_offset = (h_since_anchor // freq_hours) * freq_hours
    session_hour = (anchor_hour_et + bin_offset) % 24

    # If session_hour > current hour, the session started on the previous calendar day
    rolled_back = session_hour > hour
    midnight_ny = ts_ny.normalize()
    session_date = midnight_ny - pd.to_timedelta(rolled_back.astype(int), unit="D")
    session_ts_ny = session_date + pd.to_timedelta(session_hour, unit="h")

    return pd.DatetimeIndex(session_ts_ny).tz_convert("UTC")


def _resample_from_h1(
    h1: pd.DataFrame,
    *,
    anchor_hour_et: int,
    freq_hours: int,
    tf_raw_label: str,
    delta: timedelta,
) -> pd.DataFrame:
    """Group H1 bars by session and aggregate to OHLCV."""
    df = h1.copy()
    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else ""
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    session_starts = _assign_session_start(
        pd.DatetimeIndex(df["timestamp"]),
        anchor_hour_et=anchor_hour_et,
        freq_hours=freq_hours,
    )
    df["session"] = session_starts

    agg = df.groupby("session").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        tickVolume=("tickVolume", "sum"),
        spread=("spread", "last"),
    )
    out = agg.reset_index().rename(columns={"session": "timestamp"})

    out["symbol"] = symbol
    out["brokerTime"] = out["timestamp"].map(_broker_time)
    out["timeframe"] = tf_raw_label
    end_ts = out["timestamp"] + delta - timedelta(milliseconds=1)
    out["endTime"] = end_ts.map(_iso_z)
    out["endBrokerTime"] = end_ts.map(_broker_time)
    out["state"] = "complete"
    out["volume"] = out["tickVolume"]

    return out.reset_index(drop=True)


def main() -> None:
    all_instruments = FOREX + COMMODITIES

    for instrument in all_instruments:
        anchor = 18 if instrument in COMMODITIES else 17
        h1_path = RAW_DIR / f"{instrument}_H1.parquet"
        if not h1_path.exists():
            print(f"✗ {instrument}: missing H1 parquet")
            continue

        h1 = pq.read_table(h1_path).to_pandas()

        # --- Daily ---
        daily = _resample_from_h1(
            h1,
            anchor_hour_et=anchor,
            freq_hours=24,
            tf_raw_label="1d",
            delta=timedelta(days=1),
        )
        write_raw_parquet(daily, RAW_DIR / f"{instrument}_D.parquet")

        # --- H4 ---
        h4 = _resample_from_h1(
            h1,
            anchor_hour_et=anchor,
            freq_hours=4,
            tf_raw_label="4h",
            delta=timedelta(hours=4),
        )
        write_raw_parquet(h4, RAW_DIR / f"{instrument}_H4.parquet")

        print(
            f"✓ {instrument:<10} D: {len(daily):>5,}  H4: {len(h4):>6,}  (anchor: {anchor}:00 ET)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
