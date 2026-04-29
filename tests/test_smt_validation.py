from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_smt import _load_cached_audit_summary
from src.indicators.pipelines.build_research import run_research_pipeline
from src.indicators.research.smt_research import build_smt_research_table
from src.validation.indicators.smt import summarize_smt, validate_smt


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
        "USOIL": _raw_frame(ts, 70.0 + np.cumsum(0.5 * base + 0.06)),
        "USD_CAD": _raw_frame(ts, 1.30 + np.cumsum(-0.02 * base + 0.002)),
        "EUR_USD": _raw_frame(ts, 1.10 + np.cumsum(0.01 * base + 0.001)),
    }


def _minimal_smt_frame(rows: int = 6) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    close = np.linspace(100.0, 102.0, rows)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "xasset_smt_any_flag": np.zeros(rows, dtype=np.int8),
            "xasset_smt_best_partner": pd.Series([None] * rows, dtype="object"),
            "xasset_smt_best_score": np.full(rows, np.nan, dtype=float),
            "xasset_smt_dxy_bull_flag": np.zeros(rows, dtype=np.int8),
            "xasset_smt_dxy_bear_flag": np.zeros(rows, dtype=np.int8),
            "xasset_smt_dxy_dir": np.zeros(rows, dtype=np.int8),
            "xasset_smt_dxy_score": np.full(rows, np.nan, dtype=float),
            "xasset_smt_dxy_expected_relation": np.full(rows, -1, dtype=np.int8),
            "xasset_corr_dxy_w24": np.linspace(0.1, 0.3, rows),
            "xasset_lagcorr_gold_oil_best_w24": np.linspace(0.0, 0.2, rows),
        }
    )


def test_summarize_smt_reports_partner_breakdown() -> None:
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
    research = build_smt_research_table(runtime.frame)

    summary = summarize_smt(
        runtime.frame,
        market_context=runtime.market_context,
        research_table=research,
        instrument="XAU_USD",
        timeframe="H1",
    )

    assert "dxy" in summary["schema"]["smt_partners"]
    assert "usd_jpy" in summary["schema"]["smt_partners"]
    assert "research_summary" in summary
    assert "correlation_audit_summary" in summary


def test_summarize_smt_does_not_rebuild_audit_when_cache_inputs_missing() -> None:
    frame = _minimal_smt_frame()
    market_context = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "ret_vol_XAU_USD": np.linspace(0.0, 0.1, len(frame)),
            "ret_vol_DXY": np.linspace(0.0, -0.1, len(frame)),
        }
    )

    summary = summarize_smt(
        frame,
        market_context=market_context,
        research_table=build_smt_research_table(frame),
        instrument="XAU_USD",
        timeframe="H1",
    )

    assert summary["correlation_audit_summary"] is None


def test_load_cached_audit_summary_uses_summary_json_even_without_tables(
    tmp_path: Path,
) -> None:
    audit_dir = tmp_path / "research_cross_asset_audit" / "XAU_USD" / "H1"
    audit_dir.mkdir(parents=True)
    expected = {"pair_count": 47, "matrix_rows": 1668312}
    (audit_dir / "summary.json").write_text(json.dumps(expected), encoding="utf-8")

    summary, tables = _load_cached_audit_summary(
        audit_dir=audit_dir,
        numeric_only=False,
    )

    assert summary == expected
    assert tables is None


def test_load_cached_audit_summary_skips_cache_when_force_rebuild_enabled(
    tmp_path: Path,
) -> None:
    audit_dir = tmp_path / "research_cross_asset_audit" / "XAU_USD" / "H1"
    audit_dir.mkdir(parents=True)
    expected = {"pair_count": 47, "matrix_rows": 1668312}
    (audit_dir / "summary.json").write_text(json.dumps(expected), encoding="utf-8")

    summary, tables = _load_cached_audit_summary(
        audit_dir=audit_dir,
        numeric_only=False,
        force_rebuild_audit=True,
    )

    assert summary is None
    assert tables is None


def test_validate_smt_writes_chart(tmp_path: Path) -> None:
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

    result = validate_smt(
        runtime.frame.tail(40).copy(),
        full_df=runtime.frame,
        market_context=runtime.market_context,
        research_table=build_smt_research_table(runtime.frame),
        outpath=tmp_path / "smt_validation.html",
        title="SMT Validation Test",
        n_windows=2,
        instrument="XAU_USD",
        timeframe="H1",
    )

    assert result["html_path"].exists()
    assert "summary" in result
