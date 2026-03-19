"""
structure/ — Structural backbone indicators.

Category 2: canonical causal swing detection, trend state, BOS, CHoCH.

These are the structural foundation that SMC detectors depend on.
"""

from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state
from src.indicators.structure.bos import add_bos
from src.indicators.structure.choch import add_choch

__all__ = [
    "add_swings",
    "add_trend_state",
    "add_bos",
    "add_choch",
]
