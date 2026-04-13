from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tests.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_directory_converts_headerless_ohlcv_csv_to_project_parquet(
    tmp_path: Path,
) -> None:
    module = _load_script_module("import_csv_candles")
    input_dir = tmp_path / "USDJPY"
    output_dir = tmp_path / "raw"
    input_dir.mkdir()

    (input_dir / "USDJPY_D1.csv").write_text(
        "\n".join(
            [
                "2010-03-28 00:00,92.455,92.47,92.395,92.4,2412",
                "2014-01-01 00:00,105.0,105.2,104.8,105.1,1000",
                "2014-01-02 00:00,105.1,105.3,104.9,105.0,1100",
                "2026-04-12 00:00,159.6,159.8,159.5,159.7,1200",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    results = module.import_directory(
        input_dir,
        output_dir=output_dir,
        start=pd.Timestamp("2014-01-01T00:00:00Z").to_pydatetime(),
        end=pd.Timestamp("2026-04-11T23:59:59Z").to_pydatetime(),
    )

    assert results == [(output_dir / "USD_JPY_D.parquet", 2)]

    df = pq.read_table(output_dir / "USD_JPY_D.parquet").to_pandas()
    assert list(df.columns) == [
        "symbol",
        "timestamp",
        "brokerTime",
        "timeframe",
        "open",
        "close",
        "high",
        "low",
        "tickVolume",
        "spread",
        "endTime",
        "endBrokerTime",
        "state",
        "source_file",
    ]
    assert df["timestamp"].tolist() == [
        pd.Timestamp("2014-01-01T00:00:00Z"),
        pd.Timestamp("2014-01-02T00:00:00Z"),
    ]
    assert df["symbol"].tolist() == ["USDJPY", "USDJPY"]
    assert df["brokerTime"].tolist() == [
        "2014-01-01 02:00:00.000",
        "2014-01-02 02:00:00.000",
    ]
    assert df["timeframe"].tolist() == ["1d", "1d"]
    assert df["tickVolume"].tolist() == [1000, 1100]
    assert df["spread"].tolist() == [0, 0]
    assert df["endTime"].tolist() == [
        "2014-01-01T23:59:59.999Z",
        "2014-01-02T23:59:59.999Z",
    ]
    assert df["endBrokerTime"].tolist() == [
        "2014-01-02 01:59:59.999",
        "2014-01-03 01:59:59.999",
    ]
    assert df["state"].tolist() == ["complete", "complete"]
    assert df["source_file"].tolist() == ["USDJPY_D1.csv", "USDJPY_D1.csv"]


def test_convert_existing_project_frame_to_raw_uses_tick_volume_and_symbol(
    tmp_path: Path,
) -> None:
    module = _load_script_module("import_csv_candles")
    project = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-01-01T22:00:00Z"], utc=True),
            "open": [105.229],
            "high": [105.381],
            "low": [105.229],
            "close": [105.363],
            "volume": [654],
            "spread": [0],
            "instrument": ["USD_JPY"],
            "timeframe": ["H1"],
        }
    )

    raw = module.convert_existing_project_frame_to_raw(
        project,
        instrument="USD_JPY",
        timeframe="H1",
        source_file="USDJPY_H1.csv",
        start=pd.Timestamp("2014-01-01T00:00:00Z").to_pydatetime(),
        end=pd.Timestamp("2026-04-11T23:59:59Z").to_pydatetime(),
    )

    assert list(raw.columns) == [
        "symbol",
        "timestamp",
        "brokerTime",
        "timeframe",
        "open",
        "close",
        "high",
        "low",
        "tickVolume",
        "spread",
        "endTime",
        "endBrokerTime",
        "state",
        "source_file",
    ]
    assert raw.loc[0, "symbol"] == "USDJPY"
    assert raw.loc[0, "brokerTime"] == "2014-01-02 00:00:00.000"
    assert raw.loc[0, "timeframe"] == "1h"
    assert raw.loc[0, "tickVolume"] == 654
    assert raw.loc[0, "source_file"] == "USDJPY_H1.csv"
