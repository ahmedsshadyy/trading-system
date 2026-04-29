"""Causal SMT divergence detection for aligned cross-asset frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.indicators._helpers.validators import require_columns, require_ohlc

SMT_REQUIRED_SWING_COLUMNS = {
    "timestamp",
    "close",
    "swing_high_confirm_flag",
    "swing_low_confirm_flag",
    "swing_high_confirm_origin_idx",
    "swing_low_confirm_origin_idx",
    "swing_high_confirm_price",
    "swing_low_confirm_price",
}
SMT_OPTIONAL_STRENGTH_COLUMNS = (
    "swing_high_strength",
    "swing_low_strength",
    "swing_high_prominence_atr",
    "swing_low_prominence_atr",
    "atr_14",
)


@dataclass(slots=True)
class _SwingEvent:
    detect_idx: int
    origin_idx: int
    price: float
    strength: float


def _partner_token(value: str | None) -> str:
    if not value:
        return "partner"
    return value.strip().lower()


def _effective_side_columns(
    *,
    relation_sign: int,
    side: str,
) -> tuple[str, str, str]:
    if side not in {"high", "low"}:
        raise ValueError(f"Unsupported SMT side {side!r}")
    if relation_sign not in {-1, 1}:
        raise ValueError("SMT relation_sign must be +1 or -1")
    if relation_sign == 1:
        return (
            f"swing_{side}_confirm_flag",
            f"swing_{side}_confirm_origin_idx",
            f"swing_{side}_confirm_price",
        )
    flipped_side = "high" if side == "low" else "low"
    return (
        f"swing_{flipped_side}_confirm_flag",
        f"swing_{flipped_side}_confirm_origin_idx",
        f"swing_{flipped_side}_confirm_price",
    )


def _effective_price(raw_price: float, relation_sign: int) -> float:
    return float(raw_price if relation_sign == 1 else -raw_price)


def _event_strength(
    df: pd.DataFrame,
    *,
    origin_idx: int,
    side: str,
    relation_sign: int,
) -> float:
    if origin_idx < 0 or origin_idx >= len(df):
        return np.nan
    actual_side = side if relation_sign == 1 else ("high" if side == "low" else "low")
    candidates = (
        f"swing_{actual_side}_strength",
        f"swing_{actual_side}_prominence_atr",
    )
    for column in candidates:
        if column in df.columns:
            value = df[column].iloc[origin_idx]
            if pd.notna(value):
                return float(value)
    return np.nan


def _extract_effective_events(
    df: pd.DataFrame,
    *,
    relation_sign: int,
    side: str,
) -> dict[int, _SwingEvent]:
    flag_col, origin_col, price_col = _effective_side_columns(
        relation_sign=relation_sign,
        side=side,
    )
    flags = df[flag_col].to_numpy(dtype=np.int8)
    origins = df[origin_col].to_numpy(dtype=float)
    prices = df[price_col].to_numpy(dtype=float)
    out: dict[int, _SwingEvent] = {}
    for detect_idx in np.flatnonzero(flags == 1):
        origin_idx = origins[detect_idx]
        price = prices[detect_idx]
        if not np.isfinite(origin_idx) or not np.isfinite(price):
            continue
        origin_int = int(origin_idx)
        out[int(detect_idx)] = _SwingEvent(
            detect_idx=int(detect_idx),
            origin_idx=origin_int,
            price=_effective_price(float(price), relation_sign),
            strength=_event_strength(
                df,
                origin_idx=origin_int,
                side=side,
                relation_sign=relation_sign,
            ),
        )
    return out


def _relation_to_previous(
    current_event: _SwingEvent | None,
    previous_event: _SwingEvent | None,
    *,
    side: str,
    tolerance: float = 1e-9,
) -> int:
    if current_event is None or previous_event is None:
        return 0
    if side == "high":
        if current_event.price > previous_event.price + tolerance:
            return 1
        if current_event.price < previous_event.price - tolerance:
            return -1
        return 0
    if current_event.price < previous_event.price - tolerance:
        return -1
    if current_event.price > previous_event.price + tolerance:
        return 1
    return 0


def _freshness_score(gap: int, *, max_gap: int) -> float:
    if max_gap <= 0:
        return 0.0
    return float(np.clip(1.0 - (gap / max_gap), 0.0, 1.0))


def _strength_score(*values: float) -> float:
    finite = [abs(value) for value in values if np.isfinite(value)]
    if not finite:
        return 0.0
    return float(np.clip(np.nanmean(finite), 0.0, 3.0) / 3.0)


def _row_atr(df: pd.DataFrame, row_idx: int) -> float:
    if "atr_14" not in df.columns or row_idx < 0 or row_idx >= len(df):
        return np.nan
    value = df["atr_14"].iloc[row_idx]
    return float(value) if pd.notna(value) else np.nan


def add_smt_divergence(
    df_primary: pd.DataFrame,
    df_correlated: pd.DataFrame,
    *,
    inverse: bool = True,
    partner_name: str | None = None,
    confirm_window: int = 3,
) -> pd.DataFrame:
    """Append causal SMT divergence columns for one aligned partner.

    Parameters
    ----------
    df_primary, df_correlated:
        Already-aligned processed frames. Both must share the same row count and
        timestamp ordering.
    inverse:
        Whether the partner is expected to move inversely to the primary.
    partner_name:
        Partner token used in the output column names.
    confirm_window:
        Maximum detect-bar distance allowed between the primary and partner swing
        events before the comparison is considered stale.
    """
    out = df_primary.copy()
    require_ohlc(out, caller="add_smt_divergence")
    require_columns(
        out,
        SMT_REQUIRED_SWING_COLUMNS,
        caller="add_smt_divergence",
    )
    require_columns(
        df_correlated,
        SMT_REQUIRED_SWING_COLUMNS,
        caller="add_smt_divergence",
    )

    if len(out) != len(df_correlated):
        raise ValueError(
            "add_smt_divergence requires aligned frames with identical row counts"
        )

    primary_ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    partner_ts = pd.to_datetime(df_correlated["timestamp"], utc=True, errors="coerce")
    if not primary_ts.equals(partner_ts):
        raise ValueError(
            "add_smt_divergence requires identical timestamps on aligned frames"
        )

    partner = _partner_token(partner_name)
    relation_sign = -1 if inverse else 1
    n = len(out)

    bull_flag = np.zeros(n, dtype=np.int8)
    bear_flag = np.zeros(n, dtype=np.int8)
    direction = np.zeros(n, dtype=np.int8)
    score = np.full(n, np.nan, dtype=float)
    event_gap = np.full(n, np.nan, dtype=float)
    primary_side_relation = np.zeros(n, dtype=np.int8)
    partner_side_relation = np.zeros(n, dtype=np.int8)
    primary_side_strength = np.full(n, np.nan, dtype=float)
    partner_side_strength = np.full(n, np.nan, dtype=float)
    primary_side_atr = np.full(n, np.nan, dtype=float)
    partner_side_atr = np.full(n, np.nan, dtype=float)

    primary_low_events = _extract_effective_events(out, relation_sign=1, side="low")
    primary_high_events = _extract_effective_events(out, relation_sign=1, side="high")
    partner_low_events = _extract_effective_events(
        df_correlated,
        relation_sign=relation_sign,
        side="low",
    )
    partner_high_events = _extract_effective_events(
        df_correlated,
        relation_sign=relation_sign,
        side="high",
    )

    last_primary_low: _SwingEvent | None = None
    prev_primary_low: _SwingEvent | None = None
    last_partner_low: _SwingEvent | None = None
    prev_partner_low: _SwingEvent | None = None
    last_primary_high: _SwingEvent | None = None
    prev_primary_high: _SwingEvent | None = None
    last_partner_high: _SwingEvent | None = None
    prev_partner_high: _SwingEvent | None = None

    # Track which swing-pair has already fired to prevent the same structural
    # divergence from being flagged on multiple consecutive bars.
    last_bull_pair: tuple[int, int, int, int] | None = None
    last_bear_pair: tuple[int, int, int, int] | None = None

    for i in range(n):
        if i in primary_low_events:
            prev_primary_low = last_primary_low
            last_primary_low = primary_low_events[i]
        if i in partner_low_events:
            prev_partner_low = last_partner_low
            last_partner_low = partner_low_events[i]
        if i in primary_high_events:
            prev_primary_high = last_primary_high
            last_primary_high = primary_high_events[i]
        if i in partner_high_events:
            prev_partner_high = last_partner_high
            last_partner_high = partner_high_events[i]

        low_rel_primary = _relation_to_previous(
            last_primary_low,
            prev_primary_low,
            side="low",
        )
        low_rel_partner = _relation_to_previous(
            last_partner_low,
            prev_partner_low,
            side="low",
        )
        high_rel_primary = _relation_to_previous(
            last_primary_high,
            prev_primary_high,
            side="high",
        )
        high_rel_partner = _relation_to_previous(
            last_partner_high,
            prev_partner_high,
            side="high",
        )

        bullish_ready = (
            last_primary_low is not None
            and prev_primary_low is not None
            and last_partner_low is not None
            and prev_partner_low is not None
            and max(
                abs(last_primary_low.detect_idx - i),
                abs(last_partner_low.detect_idx - i),
                abs(last_primary_low.detect_idx - last_partner_low.detect_idx),
            )
            <= confirm_window
            and (low_rel_primary == -1) != (low_rel_partner == -1)
        )
        bearish_ready = (
            last_primary_high is not None
            and prev_primary_high is not None
            and last_partner_high is not None
            and prev_partner_high is not None
            and max(
                abs(last_primary_high.detect_idx - i),
                abs(last_partner_high.detect_idx - i),
                abs(last_primary_high.detect_idx - last_partner_high.detect_idx),
            )
            <= confirm_window
            and (high_rel_primary == 1) != (high_rel_partner == 1)
        )

        # Deduplicate: only fire on the first bar where a new swing-pair
        # divergence is detected.  The tuple identifies the four origin
        # indices that define the structural comparison.
        if bullish_ready:
            bull_pair = (
                prev_primary_low.origin_idx,
                last_primary_low.origin_idx,
                prev_partner_low.origin_idx,
                last_partner_low.origin_idx,
            )
            if bull_pair == last_bull_pair:
                bullish_ready = False
            else:
                last_bull_pair = bull_pair
        if bearish_ready:
            bear_pair = (
                prev_primary_high.origin_idx,
                last_primary_high.origin_idx,
                prev_partner_high.origin_idx,
                last_partner_high.origin_idx,
            )
            if bear_pair == last_bear_pair:
                bearish_ready = False
            else:
                last_bear_pair = bear_pair

        if bullish_ready and not bearish_ready:
            gap = abs(last_primary_low.detect_idx - last_partner_low.detect_idx)
            bull_flag[i] = 1
            direction[i] = 1
            event_gap[i] = float(gap)
            primary_side_relation[i] = low_rel_primary
            partner_side_relation[i] = low_rel_partner
            primary_side_strength[i] = last_primary_low.strength
            partner_side_strength[i] = last_partner_low.strength
            primary_side_atr[i] = _row_atr(out, last_primary_low.origin_idx)
            partner_side_atr[i] = _row_atr(df_correlated, last_partner_low.origin_idx)
            score[i] = float(
                np.clip(
                    0.20
                    + 0.40 * _freshness_score(gap, max_gap=confirm_window)
                    + 0.40
                    * _strength_score(
                        last_primary_low.strength,
                        last_partner_low.strength,
                    ),
                    0.0,
                    1.0,
                )
            )
        elif bearish_ready and not bullish_ready:
            gap = abs(last_primary_high.detect_idx - last_partner_high.detect_idx)
            bear_flag[i] = 1
            direction[i] = -1
            event_gap[i] = float(gap)
            primary_side_relation[i] = high_rel_primary
            partner_side_relation[i] = high_rel_partner
            primary_side_strength[i] = last_primary_high.strength
            partner_side_strength[i] = last_partner_high.strength
            primary_side_atr[i] = _row_atr(out, last_primary_high.origin_idx)
            partner_side_atr[i] = _row_atr(df_correlated, last_partner_high.origin_idx)
            score[i] = float(
                np.clip(
                    0.20
                    + 0.40 * _freshness_score(gap, max_gap=confirm_window)
                    + 0.40
                    * _strength_score(
                        last_primary_high.strength,
                        last_partner_high.strength,
                    ),
                    0.0,
                    1.0,
                )
            )

    out[f"xasset_smt_{partner}_bull_flag"] = bull_flag
    out[f"xasset_smt_{partner}_bear_flag"] = bear_flag
    out[f"xasset_smt_{partner}_dir"] = direction
    out[f"xasset_smt_{partner}_score"] = score
    out[f"xasset_smt_{partner}_expected_relation"] = np.full(
        n,
        relation_sign,
        dtype=np.int8,
    )
    out[f"xasset_smt_{partner}_event_gap"] = event_gap
    out[f"xasset_smt_{partner}_primary_relation"] = primary_side_relation
    out[f"xasset_smt_{partner}_partner_relation"] = partner_side_relation
    out[f"xasset_smt_{partner}_primary_strength"] = primary_side_strength
    out[f"xasset_smt_{partner}_partner_strength"] = partner_side_strength
    out[f"xasset_smt_{partner}_primary_atr"] = primary_side_atr
    out[f"xasset_smt_{partner}_partner_atr"] = partner_side_atr
    return out
