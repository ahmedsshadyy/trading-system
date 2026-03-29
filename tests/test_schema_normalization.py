from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.foundation.value import add_anchored_vwap, compute_anchored_vwap
from src.indicators.foundation.volume import (
    add_volume_features,
    add_candle_delta_proxy,
    add_key_volume_flags,
    add_volume_ratio,
    add_vsa,
)
from src.indicators.foundation.volume_profile import (
    add_volume_profile,
    compute_volume_profile,
)
from src.indicators.pipelines.build_live import build_live_indicators
from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.research.displacement_research import (
    build_displacement_research_table,
)
from src.indicators.smc.displacement import add_displacement_candle
from src.validation.indicators.displacement import validate_displacement

ROOT = Path(__file__).resolve().parents[1]


def _make_tickvolume_df(n: int = 160) -> pd.DataFrame:
    steps = np.arange(n, dtype=float)
    close = 2000.0 + np.cumsum(0.6 * np.sin(steps / 4.0) + 0.25 * np.cos(steps / 7.0))
    open_ = np.r_[close[0] - 0.4, close[:-1] + 0.1 * np.sin(steps[:-1] / 3.0)]
    high = np.maximum(open_, close) + 1.0 + (steps % 5) * 0.05
    low = np.minimum(open_, close) - 1.0 - (steps % 7) * 0.04
    tick_volume = 1000.0 + steps * 13.0 + (steps % 9) * 7.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tickVolume": tick_volume,
            "spread": np.full(n, 12.0),
        }
    )


def _as_volume_only(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    tick_idx = columns.index("tickVolume")
    out = df.drop(columns=["tickVolume"]).copy()
    out.insert(tick_idx, "volume", df["tickVolume"].astype(float))
    return out


def _load_script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tests.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_candle_schema_accepts_tick_volume_only() -> None:
    raw = _make_tickvolume_df(8)
    original = raw.copy(deep=True)

    normalized = normalize_candle_schema(raw, require_volume=True)

    pd.testing.assert_frame_equal(raw, original)
    assert "volume" in normalized.columns
    assert "tickVolume" not in normalized.columns
    pd.testing.assert_series_equal(
        normalized["volume"],
        raw["tickVolume"].astype(float),
        check_names=False,
    )


def test_normalize_candle_schema_accepts_volume_only() -> None:
    raw = _as_volume_only(_make_tickvolume_df(8))

    normalized = normalize_candle_schema(raw, require_volume=True)

    assert "volume" in normalized.columns
    assert "tickVolume" not in normalized.columns
    pd.testing.assert_series_equal(
        normalized["volume"],
        raw["volume"].astype(float),
        check_names=False,
    )


def test_normalize_candle_schema_accepts_matching_volume_and_tick_volume() -> None:
    raw = _make_tickvolume_df(8)
    raw["volume"] = raw["tickVolume"].astype(float)

    normalized = normalize_candle_schema(raw, require_volume=True)

    assert "volume" in normalized.columns
    assert "tickVolume" not in normalized.columns
    pd.testing.assert_series_equal(
        normalized["volume"],
        raw["volume"].astype(float),
        check_names=False,
    )


def test_normalize_candle_schema_rejects_mismatched_volume_sources() -> None:
    raw = _make_tickvolume_df(8)
    raw["volume"] = raw["tickVolume"].astype(float)
    raw.loc[3, "volume"] += 1.0

    with pytest.raises(ValueError, match="disagree"):
        normalize_candle_schema(raw, require_volume=True)


def test_normalize_candle_schema_rejects_missing_required_volume() -> None:
    raw = _make_tickvolume_df(8).drop(columns=["tickVolume"])

    with pytest.raises(ValueError, match="missing required volume source"):
        normalize_candle_schema(raw, require_volume=True)


def test_foundation_volume_functions_match_between_volume_and_tick_volume() -> None:
    tick_df = _make_tickvolume_df()
    volume_df = _as_volume_only(tick_df)

    pd.testing.assert_frame_equal(
        add_volume_features(tick_df, include_research_only=False),
        add_volume_features(volume_df, include_research_only=False),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        add_volume_ratio(tick_df),
        add_volume_ratio(volume_df),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        add_key_volume_flags(tick_df),
        add_key_volume_flags(volume_df),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        add_candle_delta_proxy(tick_df),
        add_candle_delta_proxy(volume_df),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        add_vsa(tick_df),
        add_vsa(volume_df),
        check_dtype=False,
    )


def test_volume_profile_and_avwap_match_between_volume_and_tick_volume() -> None:
    tick_df = _make_tickvolume_df()
    volume_df = _as_volume_only(tick_df)

    tick_profile = compute_volume_profile(tick_df, lookback=80, n_bins=24)
    volume_profile = compute_volume_profile(volume_df, lookback=80, n_bins=24)
    assert np.isclose(tick_profile["poc"], volume_profile["poc"])
    assert np.isclose(tick_profile["vah"], volume_profile["vah"])
    assert np.isclose(tick_profile["val"], volume_profile["val"])
    assert np.allclose(tick_profile["profile"], volume_profile["profile"])
    assert np.allclose(tick_profile["bin_edges"], volume_profile["bin_edges"])

    pd.testing.assert_frame_equal(
        add_volume_profile(tick_df, lookback=80, n_bins=24),
        add_volume_profile(volume_df, lookback=80, n_bins=24),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        compute_anchored_vwap(tick_df, anchor_idx=40),
        compute_anchored_vwap(volume_df, anchor_idx=40),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        add_anchored_vwap(
            tick_df,
            anchor_idx=40,
            anchor_label="day_open",
            anchor_class="live_safe",
        ),
        add_anchored_vwap(
            volume_df,
            anchor_idx=40,
            anchor_label="day_open",
            anchor_class="live_safe",
        ),
        check_dtype=False,
    )


def test_indicator_pipelines_match_between_volume_and_tick_volume() -> None:
    tick_df = _make_tickvolume_df()
    volume_df = _as_volume_only(tick_df)

    live_tick = build_live_indicators(tick_df, instrument="XAU_USD", include_vp=False)
    live_volume = build_live_indicators(
        volume_df, instrument="XAU_USD", include_vp=False
    )
    pd.testing.assert_frame_equal(live_tick, live_volume, check_dtype=False)

    research_tick = build_research_indicators(
        tick_df,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
    )
    research_volume = build_research_indicators(
        volume_df,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
    )
    pd.testing.assert_frame_equal(research_tick, research_volume, check_dtype=False)


def test_displacement_research_matches_between_volume_and_tick_volume() -> None:
    n = 40
    rows = [(100.0, 101.0, 99.0, 100.2)] * n
    rows[25] = (100.0, 112.0, 99.0, 111.0)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df["volume"] = np.arange(1, n + 1, dtype=float) * 10.0
    df["atr_14"] = np.full(n, 5.0, dtype=float)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    volume_research = build_displacement_research_table(add_displacement_candle(df))

    tick_df = df.drop(columns=["volume", "vol_ratio"]).copy()
    tick_df["tickVolume"] = np.arange(1, n + 1, dtype=float) * 10.0
    tick_df = add_volume_features(tick_df, include_research_only=False)
    tick_research = build_displacement_research_table(add_displacement_candle(tick_df))

    pd.testing.assert_frame_equal(
        tick_research,
        volume_research,
        check_dtype=False,
    )


def test_validate_displacement_script_uses_shared_analysis_builder(monkeypatch) -> None:
    module = _load_script_module("validate_displacement")
    raw = _make_tickvolume_df(40)
    captured: dict[str, pd.DataFrame] = {}

    class StopAfterBaseBuilder(RuntimeError):
        pass

    def fake_build_base(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        captured["frame"] = df.copy()
        raise StopAfterBaseBuilder

    monkeypatch.setattr(module.pd, "read_parquet", lambda _: raw.copy())
    monkeypatch.setattr(
        module, "build_displacement_analysis_base_frame", fake_build_base
    )

    with pytest.raises(StopAfterBaseBuilder):
        module.main()

    assert "tickVolume" in captured["frame"].columns
    assert "volume" not in captured["frame"].columns


def test_validate_indicators_sql_loader_normalizes_tick_volume(monkeypatch) -> None:
    module = _load_script_module("validate_indicators")
    raw = _make_tickvolume_df(10)

    monkeypatch.setattr(module.pd, "read_sql", lambda query, engine: raw.copy())
    loaded = module.load_candles("XAU_USD", "H4", object())

    assert "volume" in loaded.columns
    assert "tickVolume" not in loaded.columns
    pd.testing.assert_series_equal(
        loaded["volume"],
        raw["tickVolume"].astype(float),
        check_names=False,
    )


def test_validate_displacement_uses_full_dataset_for_numeric_summary() -> None:
    raw = _make_tickvolume_df(40)
    full_df = build_research_indicators(
        raw,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
    )
    full_df = add_displacement_candle(full_df)
    research = build_displacement_research_table(full_df)
    plot_df = full_df.iloc[-10:].copy()

    result = validate_displacement(
        plot_df,
        full_df=full_df,
        research_table=research,
        outpath=None,
    )

    assert result["summary"]["rows"] == len(full_df)
    assert result["summary"]["displacement_rows"] == int(
        full_df["displacement_flag"].sum()
    )
    assert len(result["figure"].data[0]["x"]) == len(plot_df)
