"""Build synthetic DXY parquets from existing component raw parquets.

H1 and M15 are synthesized directly (component timestamps already align).
D and H4 are resampled from DXY H1 using the same DST-aware session
assignment as resample_d_h4.py (anchor: 17:00 ET, matching forex).
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_data import (
    DXY_COMPONENT_WEIGHTS,
    synthesize_dxy_frame,
    write_raw_parquet,
)
from scripts.resample_d_h4 import _resample_from_h1

RAW_DIR = ROOT / "data" / "raw"
COMPONENTS = tuple(DXY_COMPONENT_WEIGHTS)
ANCHOR_HOUR_ET = 17  # DXY follows forex session


def _load_components(timeframe: str) -> dict[str, pd.DataFrame] | None:
    frames: dict[str, pd.DataFrame] = {}
    for instrument in COMPONENTS:
        path = RAW_DIR / f"{instrument}_{timeframe}.parquet"
        if not path.exists():
            print(f"  ✗ Missing {path.name}")
            return None
        frames[instrument] = pq.read_table(path).to_pandas()
    return frames


def main() -> None:
    # --- H1 (direct synthesis) ---
    h1_components = _load_components("H1")
    if not h1_components:
        print("✗ Cannot build DXY — missing H1 components")
        return

    dxy_h1 = synthesize_dxy_frame(h1_components, timeframe="1h")
    write_raw_parquet(dxy_h1, RAW_DIR / "DXY_H1.parquet")
    print(f"✓ DXY H1:  {len(dxy_h1):>7,} candles")

    # --- D (resample DXY H1 with DST-aware 17:00 ET session) ---
    daily = _resample_from_h1(
        dxy_h1,
        anchor_hour_et=ANCHOR_HOUR_ET,
        freq_hours=24,
        tf_raw_label="1d",
        delta=timedelta(days=1),
    )
    write_raw_parquet(daily, RAW_DIR / "DXY_D.parquet")
    print(f"✓ DXY D:   {len(daily):>7,} candles  (17:00 ET sessions, DST-aware)")

    # --- H4 (resample DXY H1 with DST-aware 17:00 ET grid) ---
    h4 = _resample_from_h1(
        dxy_h1,
        anchor_hour_et=ANCHOR_HOUR_ET,
        freq_hours=4,
        tf_raw_label="4h",
        delta=timedelta(hours=4),
    )
    write_raw_parquet(h4, RAW_DIR / "DXY_H4.parquet")
    print(f"✓ DXY H4:  {len(h4):>7,} candles  (17/21/01/05/09/13 ET grid, DST-aware)")

    # --- M15 (direct synthesis) ---
    m15_components = _load_components("M15")
    if m15_components:
        dxy_m15 = synthesize_dxy_frame(m15_components, timeframe="15m")
        write_raw_parquet(dxy_m15, RAW_DIR / "DXY_M15.parquet")
        print(f"✓ DXY M15: {len(dxy_m15):>7,} candles")
    else:
        print("✗ Cannot build DXY M15 — missing components")


if __name__ == "__main__":
    main()
