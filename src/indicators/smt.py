"""
SMT (Smart Money Technique) Divergence Calculator.

DEFERRED — requires correlated instrument data (DXY, USD/CAD) not yet loaded.

Pairs to implement:
* XAU/USD vs DXY   — inverse correlation divergence
* XAU/USD vs USD/JPY — correlation divergence
* USOIL  vs USD/CAD  — inverse correlation divergence

The scanner will call these per-signal with aligned DataFrames for both
instruments at the same timeframe.
"""

from __future__ import annotations

import pandas as pd


def add_smt_divergence(
    df_primary: pd.DataFrame,
    df_correlated: pd.DataFrame,
    inverse: bool = True,
) -> pd.DataFrame:
    """Placeholder — not yet implemented.

    Will detect divergence at swing points between two instruments.
    """
    raise NotImplementedError(
        "SMT divergence deferred — load DXY and USD/CAD data first."
    )
