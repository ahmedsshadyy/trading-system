from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.indicators.features.cross_asset import GLOBAL_CONTEXT_SYMBOL
from src.indicators.pipelines.build_live import (
    build_live_indicators,
    materialize_live_features,
)
from src.indicators.pipelines.build_research import (
    build_research_indicators,
    materialize_research_features,
)
from src.pipeline_runtime import load_partitioned_dataset


def _raw_frame(
    timestamps: pd.DatetimeIndex,
    close: np.ndarray,
) -> pd.DataFrame:
    close_series = pd.Series(close, dtype=float)
    open_ = close_series.shift(1).fillna(close_series.iloc[0] - 0.1)
    high = pd.concat([open_, close_series], axis=1).max(axis=1) + 0.2
    low = pd.concat([open_, close_series], axis=1).min(axis=1) - 0.2
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close_series.to_numpy(),
            "volume": np.arange(len(timestamps), dtype=float) + 1000.0,
        }
    )


def _cross_asset_universe(rows: int = 60) -> dict[str, pd.DataFrame]:
    ts = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    base = np.sin(np.linspace(0.0, 6.0, rows))
    return {
        "XAU_USD": _raw_frame(ts, 2000.0 + np.cumsum(base + 0.4)),
        "DXY": _raw_frame(ts, 100.0 + np.cumsum(-0.3 * base + 0.05)),
        "USD_JPY": _raw_frame(ts, 145.0 + np.cumsum(0.2 * base + 0.03)),
        "USOIL": _raw_frame(ts, 70.0 + np.cumsum(0.5 * base + 0.06)),
        "USD_CAD": _raw_frame(ts, 1.30 + np.cumsum(-0.02 * base + 0.002)),
        "EUR_USD": _raw_frame(ts, 1.10 + np.cumsum(0.01 * base + 0.001)),
    }


def test_live_and_research_cross_asset_columns_match() -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}

    live = build_live_indicators(
        primary,
        instrument="XAU_USD",
        include_vp=False,
        timeframe="H1",
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
    )
    research = build_research_indicators(
        primary,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
        timeframe="H1",
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
    )

    shared_xasset_columns = sorted(
        column
        for column in live.columns
        if column.startswith("xasset_") and column in research.columns
    )
    assert shared_xasset_columns
    pd.testing.assert_frame_equal(
        live[shared_xasset_columns].reset_index(drop=True),
        research[shared_xasset_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_live_materialization_persists_market_context_dataset(tmp_path: Path) -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}

    result = materialize_live_features(
        primary,
        instrument="XAU_USD",
        timeframe="H1",
        include_vp=False,
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
        features_root=str(tmp_path),
    )
    market_context = load_partitioned_dataset(
        tmp_path,
        dataset="market_context_live",
        symbol=GLOBAL_CONTEXT_SYMBOL,
        timeframe="H1",
    )

    assert result.metadata is not None
    assert result.metadata.extra["include_cross_asset"] is True
    assert not market_context.empty


def test_research_materialization_emits_cross_asset_audit_summary(
    tmp_path: Path,
) -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}

    materialize_research_features(
        primary,
        instrument="XAU_USD",
        timeframe="H1",
        include_vp=False,
        include_avwap=False,
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
        features_root=str(tmp_path),
    )

    summary_path = (
        tmp_path / "research_cross_asset_audit" / "XAU_USD" / "H1" / "summary.json"
    )
    assert summary_path.exists()
