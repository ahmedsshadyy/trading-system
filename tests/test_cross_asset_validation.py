from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.indicators.pipelines.build_research import run_research_pipeline
from src.validation.indicators.cross_asset import validate_cross_asset


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


def _cross_asset_universe(rows: int = 80) -> dict[str, pd.DataFrame]:
    ts = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    base = np.sin(np.linspace(0.0, 8.0, rows))
    return {
        "XAU_USD": _raw_frame(ts, 2000.0 + np.cumsum(base + 0.45)),
        "DXY": _raw_frame(ts, 100.0 + np.cumsum(-0.35 * base + 0.05)),
        "USD_JPY": _raw_frame(ts, 145.0 + np.cumsum(0.22 * base + 0.03)),
        "USOIL": _raw_frame(ts, 70.0 + np.cumsum(0.50 * base + 0.06)),
        "USD_CAD": _raw_frame(ts, 1.30 + np.cumsum(-0.02 * base + 0.002)),
        "EUR_USD": _raw_frame(ts, 1.10 + np.cumsum(0.01 * base + 0.001)),
        "GBP_USD": _raw_frame(ts, 1.28 + np.cumsum(0.015 * base + 0.001)),
    }


def test_validate_cross_asset_writes_html(tmp_path: Path) -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}
    runtime = run_research_pipeline(
        primary,
        instrument="XAU_USD",
        timeframe="H1",
        include_vp=False,
        include_avwap=False,
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
    )

    result = validate_cross_asset(
        runtime.frame,
        market_context=(
            runtime.market_context
            if runtime.market_context is not None
            else pd.DataFrame()
        ),
        instrument="XAU_USD",
        timeframe="H1",
        outpath=tmp_path / "cross_asset_validation.html",
        title="Cross-Asset Validation Test",
    )

    assert result["html_path"] is not None
    assert result["html_path"].exists()
    assert "summary" in result
    assert result["figures"]
