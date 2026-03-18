"""
pipelines/build_research.py

Full indicator pipeline for research and backtesting.

Applies every indicator layer in dependency order, including optional
diagnostics like Volume Profile and AVWAP. This is the pipeline used
for historical signal scanning, labeling, and model training.

For live deployment, use ``build_live.py`` instead.
"""

from __future__ import annotations

import pandas as pd

# --- Foundation ---
from src.indicators.foundation.ema import add_emas
from src.indicators.foundation.adx import add_adx
from src.indicators.foundation.momentum import add_rsi, add_macd, add_rsi_divergence
from src.indicators.foundation.volatility import (
    add_atr,
    add_bb_width,
    add_rolling_atr_ratio,
    add_body_ratio,
)
from src.indicators.foundation.volume import (
    add_volume_ratio,
    add_key_volume_flags,
    add_candle_delta_proxy,
    add_vsa,
    add_wick_ratio,
)
from src.indicators.foundation.value import (
    add_avwap_from_last_swing,
    add_asian_session_hl,
    add_prev_day_hl,
    add_prev_week_hl,
    add_round_number_flag,
)
from src.indicators.foundation.volume_profile import add_volume_profile
from src.indicators.foundation.session import add_session_classifier, add_time_features
from src.indicators.foundation.regime import add_regime

# --- Structure ---
from src.indicators.structure.swings import add_swings, add_swings_causal
from src.indicators.structure.trend_state import add_trend_state
from src.indicators.structure.bos import add_bos
from src.indicators.structure.choch import add_choch

# --- SMC ---
from src.indicators.smc.fvg import add_fvg
from src.indicators.smc.fvg_fill import add_fvg_fill
from src.indicators.smc.ifvg import add_ifvg
from src.indicators.smc.ob import add_ob
from src.indicators.smc.ob_mitigation import add_ob_mitigation
from src.indicators.smc.sweeps import add_liquidity_sweep
from src.indicators.smc.equal_hl import add_equal_hl
from src.indicators.smc.displacement import add_displacement_candle
from src.indicators.smc.amd import add_amd_engine


def build_research_indicators(
    df: pd.DataFrame,
    instrument: str = "XAU_USD",
    swing_window: int = 3,
    swing_mode: str = "symmetric_causal",
    include_vp: bool = True,
    include_avwap: bool = False,
) -> pd.DataFrame:
    """Apply the full indicator stack in dependency order.

    Parameters
    ----------
    df : DataFrame
        Raw candle data with columns: timestamp, open, high, low, close, volume.
    instrument : str
        For round-number detection (XAU_USD or USOIL).
    swing_window : int
        Window for swing detection (default 3 for symmetric = 6-candle span,
        default 6 for causal = 6-bar lookback).
    swing_mode : str
        'symmetric' — look-ahead swing detection (cleaner pivots, standard for backtesting).
        'causal' — no look-ahead (same detector for training and live deployment).
        'symmetric_causal' — symmetric detection with delayed availability (default).
    include_vp : bool
        Whether to compute Volume Profile (slower — disable for quick tests).
    include_avwap : bool
        Whether to compute Anchored VWAP from last swing.

    Returns
    -------
    DataFrame with all indicator columns added.
    """
    out = df.copy()

    # Ensure float types for OHLCV (PostgreSQL returns Decimal)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype(float)
    out["volume"] = out["volume"].astype(float)

    # === Layer 1: Foundation (no dependencies) ===
    out = add_atr(out)
    out = add_emas(out)
    out = add_adx(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_bb_width(out)
    out = add_body_ratio(out)

    # === Layer 2: Structure (depends on Layer 1) ===
    if swing_mode == "causal":
        out = add_swings_causal(out, window=swing_window if swing_window != 3 else 6)
    elif swing_mode == "symmetric_causal":
        out = add_swings(out, window=swing_window, causal=True)
    else:  # "symmetric"
        out = add_swings(out, window=swing_window, causal=False)
    out = add_trend_state(out)
    out = add_bos(out)
    out = add_choch(out)

    # === Layer 3: Momentum + Divergence (depends on Layer 1 + 2) ===
    out = add_rsi_divergence(out)
    out = add_rolling_atr_ratio(out)

    # === Layer 4: Volume (no structural deps) ===
    out = add_volume_ratio(out)
    out = add_key_volume_flags(out)
    out = add_candle_delta_proxy(out)
    out = add_vsa(out)
    out = add_wick_ratio(out)

    # === Layer 5: SMC (depends on Layer 2) ===
    out = add_fvg(out)
    out = add_fvg_fill(out)
    out = add_ifvg(out)
    out = add_displacement_candle(out)
    out = add_ob(out)
    out = add_ob_mitigation(out)
    out = add_liquidity_sweep(out)
    out = add_equal_hl(out)
    out = add_amd_engine(out, add_labels=False)

    # === Layer 6: Value / Reference Levels ===
    out = add_prev_day_hl(out)
    out = add_prev_week_hl(out)
    out = add_round_number_flag(out, instrument=instrument)

    # Intraday-only features (skip for Daily timeframe)
    ts = pd.to_datetime(out["timestamp"], utc=True)
    if ts.diff().median().total_seconds() < 86400:  # sub-daily
        out = add_asian_session_hl(out)
        out = add_session_classifier(out)
        out = add_time_features(out)

    # === Layer 7: Volume Profile (expensive) ===
    if include_vp:
        out = add_volume_profile(out)

    # === Layer 8: Anchored VWAP (per-signal usually) ===
    if include_avwap:
        out = add_avwap_from_last_swing(out)

    # === Layer 9: Regime (depends on multiple layers) ===
    out = add_regime(out)

    return out


# Backward-compatible alias
build_all_indicators = build_research_indicators
