"""
_helpers/schema.py

Canonical candle schema normalization helpers.

Raw provider data may expose tick volume as ``tickVolume`` while the
indicator library uses canonical internal ``volume``. These helpers make
that normalization explicit and centralized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_candle_schema(
    df: pd.DataFrame,
    *,
    require_volume: bool = False,
) -> pd.DataFrame:
    """Return a copy with canonical candle columns.

    Rules
    -----
    - ``volume`` is the canonical internal column name.
    - If only ``tickVolume`` exists, it is copied into ``volume``.
    - If both exist, overlapping non-null rows must match exactly after
      float coercion. Missing ``volume`` values are backfilled from
      ``tickVolume``. The raw ``tickVolume`` alias is then dropped.
    - If ``require_volume`` is True and neither source is present, raise.
    """
    out = df.copy()

    has_volume = "volume" in out.columns
    has_tick_volume = "tickVolume" in out.columns

    if not has_volume and not has_tick_volume:
        if require_volume:
            raise ValueError(
                "normalize_candle_schema: missing required volume source; "
                "expected 'volume' or 'tickVolume'"
            )
        return out

    if has_tick_volume:
        tick_volume = pd.to_numeric(out["tickVolume"], errors="coerce").astype(float)
        if has_volume:
            volume = pd.to_numeric(out["volume"], errors="coerce").astype(float)
            overlap = volume.notna() & tick_volume.notna()
            mismatch = overlap & ~np.isclose(
                volume.to_numpy(dtype=float),
                tick_volume.to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
            if bool(mismatch.any()):
                first_bad = out.index[mismatch.argmax()]
                raise ValueError(
                    "normalize_candle_schema: 'volume' and 'tickVolume' disagree "
                    f"at row {first_bad}"
                )
            out["volume"] = volume.fillna(tick_volume).astype(float)
        else:
            insert_at = int(out.columns.get_loc("tickVolume"))
            out.insert(insert_at, "volume", tick_volume.astype(float))
        out = out.drop(columns=["tickVolume"])
    else:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype(float)

    if require_volume and "volume" not in out.columns:
        raise ValueError(
            "normalize_candle_schema: failed to produce canonical 'volume' column"
        )

    return out
