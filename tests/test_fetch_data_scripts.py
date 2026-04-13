from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tests.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeAccount:
    def __init__(self, pages):
        self._pages = list(pages)

    async def get_historical_candles(self, **kwargs):
        if not self._pages:
            return []
        return self._pages.pop(0)


def _metaapi_candle(ts: str, *, open_: float, close: float, high: float, low: float):
    ts_obj = pd.Timestamp(ts, tz="UTC")
    return {
        "symbol": "EURUSD.sml",
        "time": ts_obj.isoformat(),
        "brokerTime": ts_obj.strftime("%Y-%m-%d %H:%M:%S.000"),
        "timeframe": "1h",
        "open": open_,
        "close": close,
        "high": high,
        "low": low,
        "tickVolume": 1000,
        "spread": 3,
        "endTime": (
            ts_obj + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z",
        "endBrokerTime": (
            ts_obj + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        ).strftime("%Y-%m-%d %H:%M:%S.000"),
        "state": "complete",
    }


def test_resolve_broker_symbols_prefers_matching_suffix_family() -> None:
    module = _load_script_module("fetch_data")

    resolved = module.resolve_broker_symbols(
        (
            module.InstrumentSpec("EUR_USD", ("EURUSD", "EUR_USD")),
            module.InstrumentSpec("USD_JPY", ("USDJPY", "USD_JPY")),
        ),
        ["EURUSD", "EURUSD.sml", "USDJPY.sml"],
    )

    assert resolved == {"EUR_USD": "EURUSD.sml", "USD_JPY": "USDJPY.sml"}


def test_fetch_candles_for_symbol_dedupes_inclusive_overlap() -> None:
    module = _load_script_module("fetch_data")
    account = _FakeAccount(
        [
            [
                _metaapi_candle(
                    "2026-04-10T03:00:00+00:00",
                    open_=1.04,
                    close=1.05,
                    high=1.06,
                    low=1.03,
                ),
                _metaapi_candle(
                    "2026-04-10T02:00:00+00:00",
                    open_=1.03,
                    close=1.04,
                    high=1.05,
                    low=1.02,
                ),
                _metaapi_candle(
                    "2026-04-10T01:00:00+00:00",
                    open_=1.02,
                    close=1.03,
                    high=1.04,
                    low=1.01,
                ),
            ],
            [
                _metaapi_candle(
                    "2026-04-10T01:00:00+00:00",
                    open_=1.02,
                    close=1.03,
                    high=1.04,
                    low=1.01,
                ),
                _metaapi_candle(
                    "2026-04-10T00:00:00+00:00",
                    open_=1.01,
                    close=1.02,
                    high=1.03,
                    low=1.00,
                ),
            ],
            [],
        ]
    )

    frame, audit = asyncio.run(
        module.fetch_candles_for_symbol(
            account,
            symbol="EURUSD.sml",
            timeframe="1h",
            start=pd.Timestamp("2026-04-10T00:00:00Z").to_pydatetime(),
            end=pd.Timestamp("2026-04-10T03:00:00Z").to_pydatetime(),
            limit=3,
        )
    )

    assert list(frame["timestamp"]) == [
        pd.Timestamp("2026-04-10T00:00:00Z"),
        pd.Timestamp("2026-04-10T01:00:00Z"),
        pd.Timestamp("2026-04-10T02:00:00Z"),
        pd.Timestamp("2026-04-10T03:00:00Z"),
    ]
    assert audit["duplicate_timestamps"] == 0


def test_fetch_candles_for_symbol_detects_stalled_pagination() -> None:
    module = _load_script_module("fetch_data")
    repeated_page = [
        _metaapi_candle(
            "2026-04-10T03:00:00+00:00", open_=1.04, close=1.05, high=1.06, low=1.03
        ),
        _metaapi_candle(
            "2026-04-10T02:00:00+00:00", open_=1.03, close=1.04, high=1.05, low=1.02
        ),
        _metaapi_candle(
            "2026-04-10T01:00:00+00:00", open_=1.02, close=1.03, high=1.04, low=1.01
        ),
    ]
    account = _FakeAccount([repeated_page, repeated_page])

    with pytest.raises(RuntimeError, match="Pagination stalled"):
        asyncio.run(
            module.fetch_candles_for_symbol(
                account,
                symbol="EURUSD.sml",
                timeframe="1h",
                start=pd.Timestamp("2026-04-10T00:00:00Z").to_pydatetime(),
                end=pd.Timestamp("2026-04-10T03:00:00Z").to_pydatetime(),
                limit=3,
            )
        )


def test_audit_candle_frame_rejects_large_interior_gap() -> None:
    module = _load_script_module("fetch_data")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2014-01-01T00:00:00Z",
                    "2014-01-01T01:00:00Z",
                    "2026-04-10T23:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "tickVolume": [100, 120, 140],
            "spread": [2, 2, 2],
        }
    )

    audit = module.audit_candle_frame(
        frame,
        instrument="USD_JPY",
        timeframe_label="H1",
        start=module.START,
        end=module.END,
    )

    assert audit["status"] == "fail"
    assert any("Interior gap too large" in message for message in audit["messages"])


def test_synthesize_dxy_frame_uses_expected_formula_and_structure() -> None:
    module = _load_script_module("fetch_data")
    ts = pd.to_datetime(["2026-04-10T00:00:00Z"], utc=True)

    def frame(open_, high, low, close):
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": [open_],
                "high": [high],
                "low": [low],
                "close": [close],
            }
        )

    components = {
        "EUR_USD": frame(1.10, 1.20, 1.00, 1.15),
        "USD_JPY": frame(150.0, 151.0, 149.0, 150.5),
        "GBP_USD": frame(1.30, 1.40, 1.20, 1.35),
        "USD_CAD": frame(1.25, 1.26, 1.24, 1.255),
        "USD_SEK": frame(10.50, 10.70, 10.30, 10.60),
        "USD_CHF": frame(0.90, 0.91, 0.89, 0.905),
    }

    out = module.synthesize_dxy_frame(components, timeframe="1h")
    row = out.iloc[0]

    expected_open = module.DXY_SCALE
    expected_high = module.DXY_SCALE
    expected_low = module.DXY_SCALE
    expected_close = module.DXY_SCALE
    for instrument, weight in module.DXY_COMPONENT_WEIGHTS.items():
        expected_open *= components[instrument]["open"].iloc[0] ** weight
        expected_close *= components[instrument]["close"].iloc[0] ** weight
        expected_high *= (
            components[instrument]["high"].iloc[0]
            if weight > 0
            else components[instrument]["low"].iloc[0]
        ) ** weight
        expected_low *= (
            components[instrument]["low"].iloc[0]
            if weight > 0
            else components[instrument]["high"].iloc[0]
        ) ** weight

    assert np.isclose(row["open"], expected_open)
    assert np.isclose(row["close"], expected_close)
    assert np.isclose(row["high"], expected_high)
    assert np.isclose(row["low"], expected_low)
    assert row["symbol"] == "DXY"
    assert row["timeframe"] == "1h"
    assert row["tickVolume"] == 0
    assert row["volume"] == 0
    assert pd.isna(row["spread"])


def test_parse_parquet_identity_and_prepare_candle_frame(tmp_path: Path) -> None:
    module = _load_script_module("load_candles")
    path = tmp_path / "GBP_USD_H1.parquet"
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-10", periods=3, freq="1h", tz="UTC"),
            "open": [1.1, 1.2, 1.3],
            "high": [1.2, 1.3, 1.4],
            "low": [1.0, 1.1, 1.2],
            "close": [1.15, 1.25, 1.35],
            "tickVolume": [1000, 1100, 1200],
            "spread": [2, 2, 2],
        }
    )
    pq.write_table(pa.Table.from_pandas(raw), path)

    instrument, timeframe = module.parse_parquet_identity(path)
    prepared = module.prepare_candle_frame(
        path, instrument=instrument, timeframe=timeframe
    )

    assert (instrument, timeframe) == ("GBP_USD", "H1")
    assert list(prepared.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "spread",
        "instrument",
        "timeframe",
    ]
    assert prepared["instrument"].tolist() == ["GBP_USD"] * 3
    assert prepared["timeframe"].tolist() == ["H1"] * 3
    assert prepared["volume"].tolist() == [1000, 1100, 1200]


def test_select_parquet_files_filters_by_instrument(tmp_path: Path) -> None:
    module = _load_script_module("load_candles")
    for name in ("USD_JPY_D.parquet", "USD_JPY_H1.parquet", "EUR_USD_D.parquet"):
        (tmp_path / name).write_text("", encoding="utf-8")

    selected = module.select_parquet_files(data_dir=tmp_path, instruments={"USD_JPY"})

    assert [path.name for path in selected] == [
        "USD_JPY_D.parquet",
        "USD_JPY_H1.parquet",
    ]


def test_build_fetch_plan_skips_when_existing_file_is_current(tmp_path: Path) -> None:
    module = _load_script_module("fetch_data")
    path = tmp_path / "EUR_USD_H1.parquet"
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-04-10T19:00:00Z", "2026-04-10T20:00:00Z"], utc=True
            ),
            "open": [1.1, 1.2],
            "high": [1.2, 1.3],
            "low": [1.0, 1.1],
            "close": [1.15, 1.25],
            "tickVolume": [1000, 1100],
            "spread": [2, 2],
        }
    )
    pq.write_table(pa.Table.from_pandas(raw), path)

    plan = module.build_fetch_plan(raw_path=path, timeframe="1h")

    assert plan["skip"] is True
    assert plan["mode"] == "skip"
    assert not plan["existing"].empty


def test_build_fetch_plan_uses_last_local_candle_for_incremental_append(
    tmp_path: Path,
) -> None:
    module = _load_script_module("fetch_data")
    path = tmp_path / "XAU_USD_H4.parquet"
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-03-13T13:00:00Z", "2026-03-13T17:00:00Z"], utc=True
            ),
            "open": [2900.0, 2910.0],
            "high": [2910.0, 2920.0],
            "low": [2890.0, 2900.0],
            "close": [2905.0, 2915.0],
            "tickVolume": [1000, 1100],
            "spread": [25, 25],
        }
    )
    pq.write_table(pa.Table.from_pandas(raw), path)

    plan = module.build_fetch_plan(raw_path=path, timeframe="4h")

    assert plan["skip"] is False
    assert plan["mode"] == "incremental"
    assert pd.Timestamp(plan["fetch_start"]) == pd.Timestamp("2026-03-13T13:00:00Z")


def test_merge_raw_frames_appends_new_tail_without_reauditing_old_gap() -> None:
    module = _load_script_module("fetch_data")
    existing = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2021-04-15T20:00:00Z",
                    "2021-05-15T20:00:00Z",
                    "2026-03-13T20:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "tickVolume": [100, 120, 140],
            "spread": [2, 2, 2],
        }
    )
    incoming = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-13T20:00:00Z",
                    "2026-03-13T21:00:00Z",
                    "2026-04-10T20:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.2, 1.22, 1.3],
            "high": [1.3, 1.32, 1.4],
            "low": [1.1, 1.12, 1.2],
            "close": [1.25, 1.27, 1.35],
            "tickVolume": [140, 150, 160],
            "spread": [2, 2, 2],
        }
    )

    contiguous_incoming = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-04-10T18:00:00Z",
                    "2026-04-10T19:00:00Z",
                    "2026-04-10T20:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.28, 1.29, 1.3],
            "high": [1.33, 1.34, 1.4],
            "low": [1.23, 1.24, 1.2],
            "close": [1.30, 1.31, 1.35],
            "tickVolume": [145, 150, 160],
            "spread": [2, 2, 2],
        }
    )

    merged = module.merge_raw_frames(existing, incoming)
    audit = module.audit_candle_frame(
        contiguous_incoming,
        instrument="EUR_USD",
        timeframe_label="H1",
        start=pd.Timestamp("2026-04-10T18:00:00Z").to_pydatetime(),
        end=module.END,
    )

    assert len(merged) == 5
    assert merged["timestamp"].iloc[-1] == pd.Timestamp("2026-04-10T20:00:00Z")
    assert audit["status"] == "pass"


def test_build_edge_repair_plan_targets_only_missing_head_and_tail(
    tmp_path: Path,
) -> None:
    module = _load_script_module("fetch_data")
    path = tmp_path / "XAU_USD_H1.parquet"
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2014-12-26T19:00:00Z", "2026-03-13T20:00:00Z"],
                utc=True,
            ),
            "open": [1200.0, 2900.0],
            "high": [1210.0, 2910.0],
            "low": [1190.0, 2890.0],
            "close": [1205.0, 2905.0],
            "tickVolume": [1000, 1100],
            "spread": [25, 25],
        }
    )
    pq.write_table(pa.Table.from_pandas(raw), path)

    plan = module.build_edge_repair_plan(raw_path=path, timeframe="1h")

    assert plan["mode"] == "edge-repair"
    assert plan["fetch_head"] is True
    assert plan["fetch_tail"] is True
    assert plan["trimmed"] is False
    assert pd.Timestamp(plan["head_end"]) == pd.Timestamp("2014-12-26T19:00:00Z")
    assert pd.Timestamp(plan["tail_start"]) == pd.Timestamp("2026-03-13T19:00:00Z")


def test_audit_boundary_coverage_allows_preexisting_interior_gap_when_edges_are_fixed() -> (
    None
):
    module = _load_script_module("fetch_data")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2014-01-01T00:00:00Z",
                    "2014-01-01T01:00:00Z",
                    "2021-05-15T20:00:00Z",
                    "2026-04-10T20:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "tickVolume": [100, 120, 140, 160],
            "spread": [2, 2, 2, 2],
        }
    )

    audit = module.audit_boundary_coverage(
        frame,
        instrument="XAU_USD",
        timeframe_label="H1",
        start=module.START,
        end=module.END,
    )

    assert audit["status"] == "pass"
    assert any("Interior gap retained" in message for message in audit["messages"])


def test_build_edge_repair_plan_trims_pre_start_rows(tmp_path: Path) -> None:
    module = _load_script_module("fetch_data")
    path = tmp_path / "XAU_USD_D.parquet"
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2013-04-12T03:00:00Z",
                    "2014-01-01T00:00:00Z",
                    "2026-03-13T03:00:00Z",
                ],
                utc=True,
            ),
            "open": [1500.0, 1200.0, 2900.0],
            "high": [1510.0, 1210.0, 2910.0],
            "low": [1490.0, 1190.0, 2890.0],
            "close": [1505.0, 1205.0, 2905.0],
            "tickVolume": [900, 1000, 1100],
            "spread": [25, 25, 25],
        }
    )
    pq.write_table(pa.Table.from_pandas(raw), path)

    plan = module.build_edge_repair_plan(raw_path=path, timeframe="1d")

    assert plan["trimmed"] is True
    assert plan["existing"]["timestamp"].min() == pd.Timestamp("2014-01-01T00:00:00Z")
