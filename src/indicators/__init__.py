"""
src/indicators/__init__.py
Indicator Library — Phase 2.

Public API re-exports only. Build orchestration lives in ``pipelines/``.

Usage
-----
    from src.indicators import build_all_indicators

    df = load_candles("XAU_USD", "H4")
    df = build_all_indicators(df, instrument="XAU_USD")
"""

# --- Foundation (pure trailing indicators) ---
from src.indicators.foundation.ema import compute_ema, add_emas
from src.indicators.foundation.adx import add_adx
from src.indicators.foundation.momentum import add_rsi, add_macd, add_rsi_divergence
from src.indicators.foundation.volatility import (
    add_atr,
    add_bb_width,
    compute_atr_ratio,
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
    compute_anchored_vwap,
    add_avwap_from_last_swing,
    add_asian_session_hl,
    add_prev_day_hl,
    add_prev_week_hl,
    add_round_number_flag,
)
from src.indicators.foundation.volume_profile import (
    compute_volume_profile,
    add_volume_profile,
)
from src.indicators.foundation.session import add_session_classifier, add_time_features
from src.indicators.foundation.regime import add_regime

# --- Structure (structural backbone) ---
from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state
from src.indicators.structure.bos import add_bos
from src.indicators.structure.choch import add_choch

# --- SMC (event detectors + active trackers) ---
from src.indicators.smc import (
    add_fvg,
    add_fvg_fill,
    add_ifvg,
    add_ob,
    add_ob_mitigation,
    add_liquidity_sweep,
    add_equal_hl,
    add_displacement_candle,
    add_amd_engine,
    add_amd_features,
    add_amd_state,
    add_amd_labels,
)

# --- Pipelines (build orchestration) ---
from src.indicators.pipelines.build_research import (
    build_research_indicators,
    build_all_indicators,
)
from src.indicators.pipelines.build_live import build_live_indicators

# SMT deferred
# from src.indicators.smt import add_smt_divergence

__all__ = [
    # Foundation
    "compute_ema",
    "add_emas",
    "add_adx",
    "add_rsi",
    "add_macd",
    "add_rsi_divergence",
    "add_atr",
    "add_bb_width",
    "compute_atr_ratio",
    "add_rolling_atr_ratio",
    "add_body_ratio",
    "add_volume_ratio",
    "add_key_volume_flags",
    "add_candle_delta_proxy",
    "add_vsa",
    "add_wick_ratio",
    "compute_anchored_vwap",
    "add_avwap_from_last_swing",
    "add_asian_session_hl",
    "add_prev_day_hl",
    "add_prev_week_hl",
    "add_round_number_flag",
    "compute_volume_profile",
    "add_volume_profile",
    "add_session_classifier",
    "add_time_features",
    "add_regime",
    # Structure
    "add_swings",
    "add_trend_state",
    "add_bos",
    "add_choch",
    # SMC
    "add_fvg",
    "add_fvg_fill",
    "add_ifvg",
    "add_ob",
    "add_ob_mitigation",
    "add_liquidity_sweep",
    "add_equal_hl",
    "add_displacement_candle",
    "add_amd_engine",
    "add_amd_features",
    "add_amd_state",
    "add_amd_labels",
    # Pipelines
    "build_research_indicators",
    "build_all_indicators",
    "build_live_indicators",
]
