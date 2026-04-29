"""Step 11B — Interaction-funnel audit for the final-sweeps detector.

This module is a *diagnostic* layer. It does not change detection logic; it
walks every (bar, ladder slot) pair on a frame that has already been through
``add_unified_liquidity_sources`` + ``add_final_sweeps`` and tabulates each
interaction's terminal funnel stage.

Funnel stages (left to right; later stages strictly imply earlier ones):

    eligible
    -> touched               (high/low entered the zone)
    -> wick_breached         (high/low pierced beyond the zone edge)
    -> close_breached        (close beyond the edge — acceptance candidate)
    -> same_bar_rejected     (wick breach + close back inside, same bar)
    -> delayed_rejected      (close-back-across within confirmation window)
    -> accepted_breakout     (close beyond + no reclaim within window)
    -> unresolved_breach     (window still open at the last bar of the slice)
    -> consumed              (cluster was swept and remains in cooldown)

The audit also reports:

* counts and conversion rates between consecutive stages
* breakdown by source family, side, penetration_atr bucket, age bucket,
  strength bucket
* which slots/families dominate the early stages (overfiring source)
* mismatch between funnel-derived expectation and the detector's actual
  ``sweep_class`` distribution

The point is to make the *cause* of overfiring visible before any rule
change lands, so we can pick a repair (penetration threshold, cooldown,
family-specific eligibility, …) for the right reason rather than tuning
blindly.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.indicators.sweeps_v2.final_sweeps import (
    DEFAULT_CONFIRMATION_WINDOW_BARS,
)
from src.indicators.sweeps_v2.unified_sources import LIQ_LADDER_DEPTH

# Funnel stage labels, in canonical order. Each interaction terminates at
# exactly one of these (the *deepest* stage it reaches).
FUNNEL_STAGES: tuple[str, ...] = (
    "eligible_only",
    "touched",
    "wick_breached",
    "close_breached_in_window",
    "same_bar_rejected",
    "delayed_rejected",
    "accepted_breakout",
    "unresolved_at_eos",
    "consumed_in_cooldown",
)

# Bucket edges
PENETRATION_ATR_EDGES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, math.inf)
AGE_BUCKET_EDGES: tuple[float, ...] = (0, 5, 20, 60, 200, math.inf)
STRENGTH_EDGES: tuple[float, ...] = (0.0, 0.40, 0.55, 0.70, 0.85, 1.0001)


def _bucket(value: float, edges: Iterable[float]) -> str:
    if not math.isfinite(value):
        return "nan"
    edges = list(edges)
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return f"[{edges[i]:.2f},{edges[i + 1]:.2f})"
    return ">=max"


@dataclass(slots=True)
class _SlotInteraction:
    bar_idx: int
    side_label: str  # "above" / "below"
    rank: int
    cluster_id: float
    family: str
    side: int
    level: float
    zone_low: float
    zone_high: float
    strength: float
    age_bars: int
    high: float
    low: float
    close: float
    atr: float
    # Will be populated after lookahead pass
    stage: str = "eligible_only"
    penetration_abs: float = 0.0
    penetration_atr: float = 0.0
    wick_prominence: float = 0.0
    close_breach: bool = False
    wick_breach: bool = False
    same_bar_close_inside: bool = False
    rejected_within_window: bool = False
    rejection_lag_bars: int = -1
    consumed: bool = False


def _read_ladder(df: pd.DataFrame, side_label: str, rank: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    base = f"liq_{side_label}_l{rank}_"
    for f in (
        "cluster_id",
        "level",
        "zone_low",
        "zone_high",
        "strength",
        "age_bars",
    ):
        col = base + f
        if col in df.columns:
            out[f] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        else:
            out[f] = np.full(len(df), np.nan, dtype=float)
    fam_col = base + "primary_family"
    if fam_col in df.columns:
        out["family"] = df[fam_col].astype(object).to_numpy()
    else:
        out["family"] = np.full(len(df), "", dtype=object)
    return out


def _compute_interaction(
    *,
    side: int,
    high: float,
    low: float,
    close: float,
    zone_low: float,
    zone_high: float,
    level: float,
) -> tuple[bool, bool, bool, bool, float, float]:
    """Return (touched, wick_breach, close_breach, same_bar_close_inside,
    penetration_abs, ratio_of_bar_range_used).
    """

    if not (math.isfinite(zone_low) and math.isfinite(zone_high)):
        return False, False, False, False, 0.0, 0.0
    if not (math.isfinite(high) and math.isfinite(low) and math.isfinite(close)):
        return False, False, False, False, 0.0, 0.0
    bar_range = max(high - low, 1e-12)
    if side == +1:
        edge = zone_high if math.isfinite(zone_high) else level
        touched = high >= zone_low - 1e-12
        wick_breach = high > edge + 1e-12
        close_breach = close > edge + 1e-12
        same_bar_close_inside = wick_breach and not close_breach
        penetration_abs = max(0.0, high - edge) if wick_breach else 0.0
    else:
        edge = zone_low if math.isfinite(zone_low) else level
        touched = low <= zone_high + 1e-12
        wick_breach = low < edge - 1e-12
        close_breach = close < edge - 1e-12
        same_bar_close_inside = wick_breach and not close_breach
        penetration_abs = max(0.0, edge - low) if wick_breach else 0.0
    prominence = penetration_abs / bar_range
    return (
        touched,
        wick_breach,
        close_breach,
        same_bar_close_inside,
        penetration_abs,
        prominence,
    )


def _check_reject_within_window(
    *,
    side: int,
    edge: float,
    closes: np.ndarray,
    breach_idx: int,
    window: int,
) -> tuple[bool, int]:
    """Look ahead up to ``window`` bars (exclusive of the breach bar) and
    report (rejected, lag_bars). The breach bar itself is handled separately
    because same-bar rejection has different semantics."""

    n = closes.shape[0]
    end = min(n - 1, breach_idx + window)
    for j in range(breach_idx + 1, end + 1):
        c = closes[j]
        if not math.isfinite(c):
            continue
        if side == +1 and c < edge - 1e-12:
            return True, j - breach_idx
        if side == -1 and c > edge + 1e-12:
            return True, j - breach_idx
    return False, -1


def _classify_interaction(
    inter: _SlotInteraction,
    *,
    is_within_window_at_eos: bool,
) -> str:
    """Resolve the deepest funnel stage this interaction reached."""

    if inter.consumed:
        return "consumed_in_cooldown"
    if not inter.wick_breach:
        if inter.close >= inter.zone_low and inter.close <= inter.zone_high:
            return "touched"
        # Did high or low at least intersect the zone?
        return (
            "touched"
            if ((inter.high >= inter.zone_low and inter.low <= inter.zone_high))
            else "eligible_only"
        )
    # Wick breach achieved
    if inter.same_bar_close_inside:
        return "same_bar_rejected"
    # Close breached
    if inter.rejected_within_window:
        return "delayed_rejected"
    # No reclaim observed within window
    if is_within_window_at_eos:
        return "unresolved_at_eos"
    return "accepted_breakout"


def build_interaction_audit(
    df: pd.DataFrame,
    *,
    confirmation_window_bars: int = DEFAULT_CONFIRMATION_WINDOW_BARS,
    cooldown_bars: int = 0,
) -> pd.DataFrame:
    """Walk every (bar, ladder slot) pair and tabulate the deepest funnel
    stage each interaction reached.

    ``cooldown_bars`` lets the audit pretend a post-sweep cooldown is in
    place — useful for *what-if* analyses before changing the detector.
    Pass ``0`` for the baseline (current detector).
    """

    if df is None or len(df) == 0:
        return pd.DataFrame()

    n = len(df)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(
        df.get("atr_14", df.get("atr", pd.Series([np.nan] * n))), errors="coerce"
    ).to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    last_sweep_bar_per_cluster: dict[tuple[str, int, float], int] = {}

    for side, side_label in ((+1, "above"), (-1, "below")):
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            arrs = _read_ladder(df, side_label, rank)
            cluster_id = arrs["cluster_id"]
            for i in range(n):
                cid = cluster_id[i]
                if not math.isfinite(cid):
                    continue
                level = arrs["level"][i]
                zone_low = arrs["zone_low"][i]
                zone_high = arrs["zone_high"][i]
                if not math.isfinite(zone_low) and math.isfinite(level):
                    zone_low = level
                if not math.isfinite(zone_high) and math.isfinite(level):
                    zone_high = level
                family = str(arrs["family"][i] or "")
                strength = arrs["strength"][i]
                age = (
                    int(arrs["age_bars"][i])
                    if math.isfinite(arrs["age_bars"][i])
                    else 0
                )

                (
                    touched,
                    wick_breach,
                    close_breach,
                    sb_close_inside,
                    pen_abs,
                    prominence,
                ) = _compute_interaction(
                    side=side,
                    high=high[i],
                    low=low[i],
                    close=close[i],
                    zone_low=zone_low,
                    zone_high=zone_high,
                    level=level,
                )
                pen_atr = (
                    pen_abs / atr[i]
                    if math.isfinite(atr[i]) and atr[i] > 0
                    else float("nan")
                )

                # Cooldown: if the same family+side+level was swept in the
                # last ``cooldown_bars`` bars, mark consumed.
                key = (
                    family,
                    side,
                    round(float(level) if math.isfinite(level) else 0.0, 4),
                )
                consumed = False
                if cooldown_bars > 0:
                    last_swept = last_sweep_bar_per_cluster.get(key, -(10**9))
                    if 0 <= i - last_swept <= cooldown_bars:
                        consumed = True

                # Lookahead reject check — only meaningful if close breached.
                edge = zone_high if side == +1 else zone_low
                if not math.isfinite(edge):
                    edge = level
                rejected, lag = (False, -1)
                is_within_window_at_eos = False
                if wick_breach and close_breach and not consumed:
                    rejected, lag = _check_reject_within_window(
                        side=side,
                        edge=edge,
                        closes=close,
                        breach_idx=i,
                        window=confirmation_window_bars,
                    )
                    if not rejected:
                        is_within_window_at_eos = i + confirmation_window_bars >= n - 1

                inter = _SlotInteraction(
                    bar_idx=i,
                    side_label=side_label,
                    rank=rank,
                    cluster_id=float(cid),
                    family=family,
                    side=side,
                    level=float(level) if math.isfinite(level) else float("nan"),
                    zone_low=(
                        float(zone_low) if math.isfinite(zone_low) else float("nan")
                    ),
                    zone_high=(
                        float(zone_high) if math.isfinite(zone_high) else float("nan")
                    ),
                    strength=(
                        float(strength) if math.isfinite(strength) else float("nan")
                    ),
                    age_bars=age,
                    high=high[i],
                    low=low[i],
                    close=close[i],
                    atr=atr[i],
                    stage="eligible_only",
                    penetration_abs=pen_abs,
                    penetration_atr=pen_atr if math.isfinite(pen_atr) else 0.0,
                    wick_prominence=prominence,
                    close_breach=close_breach,
                    wick_breach=wick_breach,
                    same_bar_close_inside=sb_close_inside,
                    rejected_within_window=rejected,
                    rejection_lag_bars=lag,
                    consumed=consumed,
                )
                stage = _classify_interaction(
                    inter, is_within_window_at_eos=is_within_window_at_eos
                )
                inter.stage = stage

                # If this interaction was a same-bar reject or a (delayed)
                # rejected sweep, mark the cluster swept for cooldown.
                if cooldown_bars > 0 and stage in (
                    "same_bar_rejected",
                    "delayed_rejected",
                ):
                    last_sweep_bar_per_cluster[key] = i

                rows.append(
                    {
                        "bar_idx": inter.bar_idx,
                        "side_label": inter.side_label,
                        "rank": inter.rank,
                        "cluster_id": inter.cluster_id,
                        "family": inter.family,
                        "side": inter.side,
                        "level": inter.level,
                        "strength": inter.strength,
                        "strength_bucket": _bucket(inter.strength, STRENGTH_EDGES),
                        "age_bars": inter.age_bars,
                        "age_bucket": _bucket(inter.age_bars, AGE_BUCKET_EDGES),
                        "stage": inter.stage,
                        "penetration_abs": inter.penetration_abs,
                        "penetration_atr": inter.penetration_atr,
                        "penetration_bucket": _bucket(
                            inter.penetration_atr, PENETRATION_ATR_EDGES
                        ),
                        "wick_prominence": inter.wick_prominence,
                        "close_breach": inter.close_breach,
                        "wick_breach": inter.wick_breach,
                        "same_bar_close_inside": inter.same_bar_close_inside,
                        "rejected_within_window": inter.rejected_within_window,
                        "rejection_lag_bars": inter.rejection_lag_bars,
                        "consumed": inter.consumed,
                    }
                )
    return pd.DataFrame(rows)


def funnel_summary(audit: pd.DataFrame) -> dict[str, object]:
    """Reduce the audit table into the printable funnel summary."""

    if audit is None or len(audit) == 0:
        return {"error": "empty_audit"}
    out: dict[str, object] = {}
    total = int(len(audit))
    out["total_interactions"] = total

    # Stage counts in canonical order
    stage_counts = Counter(audit["stage"].tolist())
    out["stage_counts"] = {s: int(stage_counts.get(s, 0)) for s in FUNNEL_STAGES}

    # Conversion rates between consecutive stages
    cumulative = []
    running = 0
    for s in FUNNEL_STAGES:
        running += int(stage_counts.get(s, 0))
        cumulative.append(running)
    conv: dict[str, float] = {}
    for i in range(1, len(FUNNEL_STAGES)):
        prev_count = cumulative[i - 1]
        if prev_count == 0:
            conv[FUNNEL_STAGES[i]] = float("nan")
        else:
            conv[FUNNEL_STAGES[i]] = round(
                int(stage_counts.get(FUNNEL_STAGES[i], 0)) / prev_count, 4
            )
    out["conversion_to_stage_from_prior_cumulative"] = conv

    # Stage distribution by family
    by_family: dict[str, dict[str, int]] = {}
    for fam, sub in audit.groupby("family"):
        by_family[str(fam) or "none"] = {
            s: int((sub["stage"] == s).sum()) for s in FUNNEL_STAGES
        }
    out["stage_by_family"] = by_family

    # Stage distribution by side
    out["stage_by_side"] = {
        side_label: {
            s: int(((audit["side_label"] == side_label) & (audit["stage"] == s)).sum())
            for s in FUNNEL_STAGES
        }
        for side_label in ("above", "below")
    }

    # Stage distribution by penetration bucket (only for wick-breached and beyond)
    breached = audit[audit["wick_breach"]]
    out["stage_by_penetration_bucket"] = {
        bucket: {
            s: int(
                (
                    (breached["penetration_bucket"] == bucket)
                    & (breached["stage"] == s)
                ).sum()
            )
            for s in FUNNEL_STAGES
            if s not in ("eligible_only", "touched")
        }
        for bucket in sorted(breached["penetration_bucket"].dropna().unique())
    }

    # Stage by source-age bucket
    out["stage_by_age_bucket"] = {
        bucket: dict(
            Counter(audit.loc[audit["age_bucket"] == bucket, "stage"].tolist())
        )
        for bucket in sorted(audit["age_bucket"].unique())
    }

    # Stage by source-strength bucket
    out["stage_by_strength_bucket"] = {
        bucket: dict(
            Counter(audit.loc[audit["strength_bucket"] == bucket, "stage"].tolist())
        )
        for bucket in sorted(audit["strength_bucket"].dropna().unique())
    }

    # Where same-bar rejections fail meaningful penetration filters
    sb = audit[audit["stage"] == "same_bar_rejected"]
    out["same_bar_penetration_distribution_atr"] = {
        "count": int(len(sb)),
        "mean": float(sb["penetration_atr"].mean()) if len(sb) else float("nan"),
        "p10": float(sb["penetration_atr"].quantile(0.10)) if len(sb) else float("nan"),
        "p50": float(sb["penetration_atr"].quantile(0.50)) if len(sb) else float("nan"),
        "p90": float(sb["penetration_atr"].quantile(0.90)) if len(sb) else float("nan"),
        "max": float(sb["penetration_atr"].max()) if len(sb) else float("nan"),
        "share_below_0_10_atr": (
            round(float((sb["penetration_atr"] < 0.10).mean()), 4)
            if len(sb)
            else float("nan")
        ),
        "share_below_0_05_atr": (
            round(float((sb["penetration_atr"] < 0.05).mean()), 4)
            if len(sb)
            else float("nan")
        ),
    }
    return out


def print_funnel_summary(summary: dict[str, object], indent: int = 0) -> None:
    prefix = " " * indent
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_funnel_summary(value, indent + 2)
        else:
            print(f"{prefix}{key}: {value}")


__all__ = [
    "FUNNEL_STAGES",
    "PENETRATION_ATR_EDGES",
    "AGE_BUCKET_EDGES",
    "STRENGTH_EDGES",
    "build_interaction_audit",
    "funnel_summary",
    "print_funnel_summary",
]
