import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from metaapi_cloud_sdk import MetaApi
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")

TIMEFRAMES = ["1d", "4h", "1h", "15m"]

TF_MAP = {
    "1d": "D",
    "4h": "H4",
    "1h": "H1",
    "15m": "M15",
}

START = datetime(2015, 1, 1, tzinfo=timezone.utc)

os.makedirs("data/raw", exist_ok=True)


async def fetch():
    api = MetaApi(TOKEN)

    print("Connecting to MetaApi...")
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

    if account.state != "DEPLOYED":
        print("Deploying account...")
        await account.deploy()

    await account.wait_connected()
    print("Connected.\n")

    # Check symbols first
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()

    print("Checking available symbols...")
    symbols = await connection.get_symbols()
    relevant = [
        s
        for s in symbols
        if any(x in s.upper() for x in ["XAU", "OIL", "WTI", "BRENT", "EUR", "JPY"])
    ]
    print("Relevant symbols found:", relevant)
    print()

    await connection.close()

    INSTRUMENTS = {
        "XAU_USD": "XAUUSD.sml",
        "USOIL": "USOIL.sml",
        "EUR_USD": "EURUSD.sml",
        "USD_JPY": "USDJPY.sml",
    }

    total_tasks = len(INSTRUMENTS) * len(TIMEFRAMES)

    with tqdm(total=total_tasks, desc="Overall", unit="file") as overall_bar:
        for instrument_key, symbol in INSTRUMENTS.items():
            for tf in TIMEFRAMES:
                tf_key = TF_MAP[tf]
                overall_bar.set_description(f"Fetching {symbol} {tf_key}")

                try:
                    all_candles = []

                    with tqdm(
                        desc=f"  {symbol} {tf_key}", leave=False, unit="chunk"
                    ) as chunk_bar:
                        # MetaApi paginates BACKWARDS — start from now, go back to START
                        end_time = None  # None = start from most recent candle

                        while True:
                            candles = await account.get_historical_candles(
                                symbol=symbol,
                                timeframe=tf,
                                start_time=end_time,
                                limit=1000,
                            )

                            if not candles:
                                break

                            all_candles.extend(candles)
                            chunk_bar.update(1)
                            chunk_bar.set_postfix(
                                {
                                    "candles": f"{len(all_candles):,}",
                                    "year": str(candles[0]["time"])[:4],
                                    "date": str(candles[0]["time"])[:10],
                                }
                            )

                            # Move end_time back to just before the oldest candle in this batch
                            oldest_time = pd.to_datetime(candles[0]["time"])

                            # Stop if we've gone back far enough
                            if oldest_time <= pd.Timestamp(START):
                                break

                            if len(candles) < 1000:
                                break

                            end_time = oldest_time.to_pydatetime().replace(
                                tzinfo=timezone.utc
                            )

                    if not all_candles:
                        tqdm.write(f"{symbol} {tf_key}: No data — check symbol name")
                        overall_bar.update(1)
                        continue

                    df = pd.DataFrame(all_candles)
                    df["time"] = pd.to_datetime(df["time"])
                    df = df.rename(columns={"time": "timestamp"})
                    df = df.drop_duplicates(subset=["timestamp"]).sort_values(
                        "timestamp"
                    )

                    path = f"data/raw/{instrument_key}_{tf_key}.parquet"
                    pq.write_table(pa.Table.from_pandas(df), path)
                    tqdm.write(f"✓ {symbol} {tf_key}: {len(df)} candles → {path}")

                except Exception as e:
                    tqdm.write(f"✗ {symbol} {tf_key}: Error: {e}")

                overall_bar.update(1)

    print("\nDone. UNDEPLOY your MetaApi account now!")
    api.close()


asyncio.run(fetch())
