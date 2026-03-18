"""
foundation/ — Pure trailing indicators.

Category 1 indicators with no event semantics and no structural dependencies.
These are the same for research and live pipelines.
"""

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
from src.indicators.foundation.session import add_session_classifier, add_time_features
from src.indicators.foundation.regime import add_regime
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

__all__ = [
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
    "add_session_classifier",
    "add_time_features",
    "add_regime",
    "compute_anchored_vwap",
    "add_avwap_from_last_swing",
    "add_asian_session_hl",
    "add_prev_day_hl",
    "add_prev_week_hl",
    "add_round_number_flag",
    "compute_volume_profile",
    "add_volume_profile",
]
