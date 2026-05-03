"""
smc/ — SMC (Smart Money Concepts) event detectors and active trackers.

Categories 3+4: FVG, IFVG, Order Blocks, liquidity sweeps, equal H/L,
displacement candles, and AMD phase classifier.

The canonical sweep detector lives in :mod:`src.indicators.smc.sweeps`
(``add_final_sweeps`` + ``add_unified_liquidity_sources``).
"""

from src.indicators.smc.fvg import add_fvg
from src.indicators.smc.fvg_fill import add_fvg_fill
from src.indicators.smc.ifvg import add_ifvg
from src.indicators.smc.ob import add_ob
from src.indicators.smc.ob_mitigation import add_ob_mitigation
from src.indicators.smc.sweeps import (
    add_final_sweeps,
    add_unified_liquidity_sources,
)
from src.indicators.smc.equal_hl import add_equal_hl
from src.indicators.smc.displacement import add_displacement_candle
from src.indicators.smc.amd import (
    add_amd_features,
    add_amd_state,
    add_amd_labels,
    add_amd_engine,
)

__all__ = [
    "add_fvg",
    "add_fvg_fill",
    "add_ifvg",
    "add_ob",
    "add_ob_mitigation",
    "add_final_sweeps",
    "add_unified_liquidity_sources",
    "add_equal_hl",
    "add_displacement_candle",
    "add_amd_features",
    "add_amd_state",
    "add_amd_labels",
    "add_amd_engine",
]
