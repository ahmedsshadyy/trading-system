"""
Session & Time features.

Session classifier, London/NY open flags, Asian range detector,
day-of-week, Friday flag, macro proximity placeholder.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Session Classifier
# ---------------------------------------------------------------------------


def add_session_classifier(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each candle by trading session (UTC hours).

    Session windows (UTC):
    * Asian:          00:00 – 08:00
    * London:         08:00 – 17:00
    * NY:             13:00 – 22:00
    * London/NY overlap: 13:00 – 17:00
    * Dead zone:      22:00 – 00:00

    Columns
    ~~~~~~~
    * ``session``              – ordinal: 0=Asian, 1=London, 2=NY, 3=overlap, 4=dead
    * ``london_open``          – binary (08:00–10:00 UTC)
    * ``ny_open``              – binary (13:00–15:00 UTC)
    * ``is_dead_zone``         – binary (22:00–00:00 UTC)
    * ``hours_since_session``  – hours since current session opened
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    hour = ts.dt.hour

    session = np.full(len(out), 4, dtype=np.int8)  # default dead zone

    # Asian: 00–08
    session = np.where((hour >= 0) & (hour < 8), 0, session)
    # London (non-overlap): 08–13
    session = np.where((hour >= 8) & (hour < 13), 1, session)
    # Overlap: 13–17
    session = np.where((hour >= 13) & (hour < 17), 3, session)
    # NY (non-overlap): 17–22
    session = np.where((hour >= 17) & (hour < 22), 2, session)

    out["session"] = session
    out["london_open"] = ((hour >= 8) & (hour < 10)).astype(int)
    out["ny_open"] = ((hour >= 13) & (hour < 15)).astype(int)
    out["is_dead_zone"] = ((hour >= 22) | (hour < 0)).astype(int)

    # Hours since session open
    session_starts = {0: 0, 1: 8, 2: 17, 3: 13, 4: 22}
    hrs = np.zeros(len(out))
    for i in range(len(out)):
        s = session[i]
        start = session_starts[s]
        h = hour.iloc[i]
        hrs[i] = (h - start) % 24
    out["hours_since_session"] = hrs

    return out


# ---------------------------------------------------------------------------
# Time Features
# ---------------------------------------------------------------------------


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar / time-based features.

    Columns
    ~~~~~~~
    * ``day_of_week``    – ordinal 0=Mon … 4=Fri
    * ``is_friday``      – binary
    * ``hour_of_day``    – 0–23
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["day_of_week"] = ts.dt.dayofweek  # 0=Mon
    out["is_friday"] = (ts.dt.dayofweek == 4).astype(int)
    out["hour_of_day"] = ts.dt.hour
    return out
