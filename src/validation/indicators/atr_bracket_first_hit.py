"""Step 11T — ATR bracket first-hit audit.

Compares confirmed swings vs confirmed sweeps under a symmetric
``±0.5 * ATR`` TP/SL bracket. Research-only: nothing in this module
mutates indicator outputs; all forward-looking metrics are anchored
strictly at ``confirm_idx`` and read only bars in
``confirm_idx + 1 .. confirm_idx + horizon``.

Outcome categories per (event, horizon):

* ``tp_first`` — TP hit on a bar where SL did not also hit, before any SL hit
* ``sl_first`` — SL hit on a bar where TP did not also hit, before any TP hit
* ``ambiguous_same_bar`` — TP and SL touched on the same candle (NEVER guess)
* ``neither`` — neither touched within the horizon
* ``insufficient_future`` — ``confirm_idx + horizon`` is past the frame end,
  or the entry/ATR reference is unusable
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

ATR_BRACKET_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5)
ATR_BRACKET_TP_MULT: float = 0.5
ATR_BRACKET_SL_MULT: float = 0.5

_BULLISH = "bullish_reversal"
_BEARISH = "bearish_reversal"

_OUTCOME_CATEGORIES: tuple[str, ...] = (
    "tp_first",
    "sl_first",
    "ambiguous_same_bar",
    "neither",
    "insufficient_future",
)


# ---------------------------------------------------------------------------
# Per-event scan
# ---------------------------------------------------------------------------


def _scan_first_hits(
    *,
    side: str,
    entry_close: float,
    atr_ref: float,
    high: np.ndarray,
    low: np.ndarray,
    confirm_idx: int,
    max_horizon: int,
) -> tuple[float | None, float | None, float | None, float, float]:
    """Forward-scan up to ``max_horizon`` bars and return the first TP-only,
    SL-only, and same-bar collision bar offsets (1-indexed). Stops at the
    earliest event because the bracket position closes there.
    """

    if side == _BULLISH:
        tp_price = entry_close + ATR_BRACKET_TP_MULT * atr_ref
        sl_price = entry_close - ATR_BRACKET_SL_MULT * atr_ref
    else:
        tp_price = entry_close - ATR_BRACKET_TP_MULT * atr_ref
        sl_price = entry_close + ATR_BRACKET_SL_MULT * atr_ref

    first_tp_only: float | None = None
    first_sl_only: float | None = None
    first_ambiguous: float | None = None

    n = len(high)
    end_idx = min(confirm_idx + max_horizon, n - 1)
    for delay in range(1, max_horizon + 1):
        j = confirm_idx + delay
        if j > end_idx:
            break
        bar_high = high[j]
        bar_low = low[j]
        if not (np.isfinite(bar_high) and np.isfinite(bar_low)):
            continue
        if side == _BULLISH:
            tp_hit = bar_high >= tp_price - 1e-12
            sl_hit = bar_low <= sl_price + 1e-12
        else:
            tp_hit = bar_low <= tp_price + 1e-12
            sl_hit = bar_high >= sl_price - 1e-12
        if tp_hit and sl_hit:
            first_ambiguous = float(delay)
            break
        if tp_hit:
            first_tp_only = float(delay)
            break
        if sl_hit:
            first_sl_only = float(delay)
            break
    return first_tp_only, first_sl_only, first_ambiguous, tp_price, sl_price


def _outcome_for_horizon(
    *,
    horizon: int,
    first_tp_only: float | None,
    first_sl_only: float | None,
    first_ambiguous: float | None,
    available_bars: int,
) -> tuple[str, float | None, float | None]:
    """Resolve the per-horizon outcome label. Returns
    ``(outcome, bars_to_tp, bars_to_sl)``.

    The bracket position closes at the first TP/SL/collision bar, so if a
    hit was already observed within the available future bars we lock in
    that outcome regardless of horizon. ``insufficient_future`` is reserved
    for the case where no hit was seen AND we did not have ``horizon`` bars
    to look at — i.e. we genuinely cannot decide.
    """

    candidates: list[tuple[float, str]] = []
    if first_tp_only is not None and first_tp_only <= horizon:
        candidates.append((first_tp_only, "tp_first"))
    if first_sl_only is not None and first_sl_only <= horizon:
        candidates.append((first_sl_only, "sl_first"))
    if first_ambiguous is not None and first_ambiguous <= horizon:
        candidates.append((first_ambiguous, "ambiguous_same_bar"))

    if candidates:
        candidates.sort(key=lambda x: (x[0],))
        bar, outcome = candidates[0]
        if outcome == "tp_first":
            return outcome, float(bar), None
        if outcome == "sl_first":
            return outcome, None, float(bar)
        return outcome, float(bar), float(bar)

    if available_bars < horizon:
        return "insufficient_future", None, None
    return "neither", None, None


# ---------------------------------------------------------------------------
# Event collection
# ---------------------------------------------------------------------------


def _swing_direction(side_value: str) -> str:
    return _BULLISH if side_value == "swing_low" else _BEARISH


def _sweep_direction(sweep_side_numeric: float) -> str:
    if not np.isfinite(sweep_side_numeric):
        return ""
    val = int(sweep_side_numeric)
    if val == -1:
        return _BULLISH  # below_sell_side
    if val == 1:
        return _BEARISH  # above_buy_side
    return ""


def _sweep_side_label(sweep_side_numeric: float) -> str:
    if not np.isfinite(sweep_side_numeric):
        return ""
    val = int(sweep_side_numeric)
    if val == -1:
        return "below_sell_side"
    if val == 1:
        return "above_buy_side"
    return ""


def build_atr_bracket_events(
    df: pd.DataFrame,
    *,
    horizons: Iterable[int] = ATR_BRACKET_HORIZONS,
) -> pd.DataFrame:
    """Build the per-event Step 11T table for both confirmed swings and
    confirmed sweeps.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame()

    horizons_t = tuple(int(h) for h in horizons)
    if not horizons_t:
        return pd.DataFrame()
    max_horizon = max(horizons_t)

    required = {"high", "low", "close", "atr_14"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    # Ensure 0..N-1 RangeIndex for safe positional arithmetic.
    if (
        not df.index.is_monotonic_increasing
        or df.index[0] != 0
        or df.index[-1] != len(df) - 1
    ):
        df = df.reset_index(drop=True)

    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(df["atr_14"], errors="coerce").to_numpy(dtype=float)
    n = len(df)

    timestamps = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if "timestamp" in df.columns
        else pd.Series([pd.NaT] * n)
    )

    regime_label = (
        df["regime_label"].astype(str).fillna("").to_numpy(dtype=object)
        if "regime_label" in df.columns
        else np.array([""] * n, dtype=object)
    )
    session_phase = (
        df["session_name"].astype(str).fillna("").to_numpy(dtype=object)
        if "session_name" in df.columns
        else np.array([""] * n, dtype=object)
    )

    rows: list[dict[str, object]] = []

    # ---- swings ----
    swing_high_flag = (
        pd.to_numeric(df["swing_high_confirm_flag"], errors="coerce")
        .fillna(0.0)
        .to_numpy()
        if "swing_high_confirm_flag" in df.columns
        else np.zeros(n)
    )
    swing_low_flag = (
        pd.to_numeric(df["swing_low_confirm_flag"], errors="coerce")
        .fillna(0.0)
        .to_numpy()
        if "swing_low_confirm_flag" in df.columns
        else np.zeros(n)
    )

    event_id = 1
    for confirm_idx in range(n):
        for side, flag_arr in (
            ("swing_high", swing_high_flag),
            ("swing_low", swing_low_flag),
        ):
            if flag_arr[confirm_idx] <= 0:
                continue
            entry_close = close[confirm_idx]
            atr_ref = atr[confirm_idx]
            direction = _swing_direction(side)
            row = _build_event_row(
                entity_type="swing",
                signal_side=direction,
                signal_subtype=side,
                signal_confirm_idx=confirm_idx,
                signal_confirm_timestamp=timestamps.iat[confirm_idx],
                entry_close=entry_close,
                atr_ref=atr_ref,
                high=high,
                low=low,
                n=n,
                horizons=horizons_t,
                max_horizon=max_horizon,
                regime_label=str(regime_label[confirm_idx] or ""),
                session_phase=str(session_phase[confirm_idx] or ""),
                event_id=event_id,
                extra={
                    "swing_side": side,
                    "sweep_side_label": "",
                    "sweep_selectivity_class": "",
                    "sweep_primary_family": "",
                    "sweep_is_displacement_confirmed": np.nan,
                    "sweep_is_tradeable_candidate": np.nan,
                    "sweep_class": np.nan,
                },
            )
            rows.append(row)
            event_id += 1

    # ---- sweeps ----
    if "sweep_flag" in df.columns:
        sweep_flag = (
            pd.to_numeric(df["sweep_flag"], errors="coerce").fillna(0.0).to_numpy()
        )
        sweep_side = (
            pd.to_numeric(df["sweep_side"], errors="coerce").to_numpy()
            if "sweep_side" in df.columns
            else np.full(n, np.nan)
        )
        sel_class = (
            df["sweep_selectivity_class"].astype(str).fillna("").to_numpy(dtype=object)
            if "sweep_selectivity_class" in df.columns
            else np.array([""] * n, dtype=object)
        )
        primary_family = (
            df["sweep_primary_family"].astype(str).fillna("").to_numpy(dtype=object)
            if "sweep_primary_family" in df.columns
            else np.array([""] * n, dtype=object)
        )
        disp_conf = (
            pd.to_numeric(df["sweep_is_displacement_confirmed"], errors="coerce")
            .fillna(0.0)
            .to_numpy()
            if "sweep_is_displacement_confirmed" in df.columns
            else np.zeros(n)
        )
        tradeable_flag = (
            pd.to_numeric(df["sweep_is_tradeable_candidate"], errors="coerce")
            .fillna(0.0)
            .to_numpy()
            if "sweep_is_tradeable_candidate" in df.columns
            else np.zeros(n)
        )
        sweep_class_arr = (
            pd.to_numeric(df["sweep_class"], errors="coerce").to_numpy()
            if "sweep_class" in df.columns
            else np.full(n, np.nan)
        )

        for confirm_idx in range(n):
            if sweep_flag[confirm_idx] <= 0:
                continue
            direction = _sweep_direction(sweep_side[confirm_idx])
            if direction == "":
                continue
            entry_close = close[confirm_idx]
            atr_ref = atr[confirm_idx]
            row = _build_event_row(
                entity_type="sweep",
                signal_side=direction,
                signal_subtype=_sweep_side_label(sweep_side[confirm_idx]),
                signal_confirm_idx=confirm_idx,
                signal_confirm_timestamp=timestamps.iat[confirm_idx],
                entry_close=entry_close,
                atr_ref=atr_ref,
                high=high,
                low=low,
                n=n,
                horizons=horizons_t,
                max_horizon=max_horizon,
                regime_label=str(regime_label[confirm_idx] or ""),
                session_phase=str(session_phase[confirm_idx] or ""),
                event_id=event_id,
                extra={
                    "swing_side": "",
                    "sweep_side_label": _sweep_side_label(sweep_side[confirm_idx]),
                    "sweep_selectivity_class": str(sel_class[confirm_idx] or ""),
                    "sweep_primary_family": str(primary_family[confirm_idx] or ""),
                    "sweep_is_displacement_confirmed": float(
                        disp_conf[confirm_idx] > 0
                    ),
                    "sweep_is_tradeable_candidate": float(
                        tradeable_flag[confirm_idx] > 0
                    ),
                    "sweep_class": (
                        float(sweep_class_arr[confirm_idx])
                        if np.isfinite(sweep_class_arr[confirm_idx])
                        else np.nan
                    ),
                },
            )
            rows.append(row)
            event_id += 1

    if not rows:
        return pd.DataFrame()

    events = (
        pd.DataFrame(rows)
        .sort_values(["signal_confirm_idx", "signal_entity_type", "event_id"])
        .reset_index(drop=True)
    )
    return events


def _build_event_row(
    *,
    entity_type: str,
    signal_side: str,
    signal_subtype: str,
    signal_confirm_idx: int,
    signal_confirm_timestamp: pd.Timestamp,
    entry_close: float,
    atr_ref: float,
    high: np.ndarray,
    low: np.ndarray,
    n: int,
    horizons: tuple[int, ...],
    max_horizon: int,
    regime_label: str,
    session_phase: str,
    event_id: int,
    extra: dict[str, object],
) -> dict[str, object]:
    available_bars = max(0, n - 1 - signal_confirm_idx)
    valid_inputs = np.isfinite(entry_close) and np.isfinite(atr_ref) and atr_ref > 0

    if valid_inputs:
        first_tp_only, first_sl_only, first_ambig, tp_price, sl_price = (
            _scan_first_hits(
                side=signal_side,
                entry_close=float(entry_close),
                atr_ref=float(atr_ref),
                high=high,
                low=low,
                confirm_idx=signal_confirm_idx,
                max_horizon=max_horizon,
            )
        )
    else:
        first_tp_only = first_sl_only = first_ambig = None
        if signal_side == _BULLISH:
            tp_price = (
                float(entry_close) + ATR_BRACKET_TP_MULT * float(atr_ref)
                if np.isfinite(entry_close) and np.isfinite(atr_ref)
                else np.nan
            )
            sl_price = (
                float(entry_close) - ATR_BRACKET_SL_MULT * float(atr_ref)
                if np.isfinite(entry_close) and np.isfinite(atr_ref)
                else np.nan
            )
        else:
            tp_price = (
                float(entry_close) - ATR_BRACKET_TP_MULT * float(atr_ref)
                if np.isfinite(entry_close) and np.isfinite(atr_ref)
                else np.nan
            )
            sl_price = (
                float(entry_close) + ATR_BRACKET_SL_MULT * float(atr_ref)
                if np.isfinite(entry_close) and np.isfinite(atr_ref)
                else np.nan
            )

    row: dict[str, object] = {
        "event_id": int(event_id),
        "signal_entity_type": entity_type,
        "signal_subtype": signal_subtype,
        "signal_side": signal_side,
        "signal_confirm_idx": int(signal_confirm_idx),
        "signal_confirm_timestamp": signal_confirm_timestamp,
        "entry_close": float(entry_close) if np.isfinite(entry_close) else np.nan,
        "atr_ref": float(atr_ref) if np.isfinite(atr_ref) else np.nan,
        "tp_price": float(tp_price) if np.isfinite(tp_price) else np.nan,
        "sl_price": float(sl_price) if np.isfinite(sl_price) else np.nan,
        "regime_label": regime_label,
        "session_phase": session_phase,
        **extra,
    }

    for horizon in horizons:
        if not valid_inputs:
            outcome, bars_to_tp, bars_to_sl = "insufficient_future", None, None
        else:
            outcome, bars_to_tp, bars_to_sl = _outcome_for_horizon(
                horizon=horizon,
                first_tp_only=first_tp_only,
                first_sl_only=first_sl_only,
                first_ambiguous=first_ambig,
                available_bars=available_bars,
            )
        first_tp_bar = (
            float(bars_to_tp)
            if outcome == "tp_first"
            else (float(first_ambig) if outcome == "ambiguous_same_bar" else np.nan)
        )
        first_sl_bar = (
            float(bars_to_sl)
            if outcome == "sl_first"
            else (float(first_ambig) if outcome == "ambiguous_same_bar" else np.nan)
        )
        row[f"first_tp_hit_bar_{horizon}"] = first_tp_bar
        row[f"first_sl_hit_bar_{horizon}"] = first_sl_bar
        row[f"bracket_outcome_{horizon}"] = outcome
        row[f"bars_to_tp_{horizon}"] = (
            float(bars_to_tp) if bars_to_tp is not None else np.nan
        )
        row[f"bars_to_sl_{horizon}"] = (
            float(bars_to_sl) if bars_to_sl is not None else np.nan
        )
        row[f"tp_before_sl_flag_{horizon}"] = 1.0 if outcome == "tp_first" else 0.0
        row[f"sl_before_tp_flag_{horizon}"] = 1.0 if outcome == "sl_first" else 0.0
        row[f"ambiguous_same_bar_flag_{horizon}"] = (
            1.0 if outcome == "ambiguous_same_bar" else 0.0
        )
    return row


# ---------------------------------------------------------------------------
# Group rollups
# ---------------------------------------------------------------------------


def _group_metrics(group: pd.DataFrame, horizons: Iterable[int]) -> dict[str, object]:
    metrics: dict[str, object] = {"count": int(len(group))}
    for horizon in horizons:
        col = f"bracket_outcome_{horizon}"
        labels = (
            group[col].astype(str) if col in group.columns else pd.Series([], dtype=str)
        )
        tp = int((labels == "tp_first").sum())
        sl = int((labels == "sl_first").sum())
        amb = int((labels == "ambiguous_same_bar").sum())
        none = int((labels == "neither").sum())
        ins = int((labels == "insufficient_future").sum())
        n = tp + sl + amb + none + ins
        resolved = tp + sl
        metrics[f"tp_first_count_{horizon}"] = tp
        metrics[f"sl_first_count_{horizon}"] = sl
        metrics[f"ambiguous_count_{horizon}"] = amb
        metrics[f"neither_count_{horizon}"] = none
        metrics[f"insufficient_count_{horizon}"] = ins
        metrics[f"resolved_count_{horizon}"] = resolved
        metrics[f"tp_first_rate_{horizon}"] = float(tp / n) if n else float("nan")
        metrics[f"sl_first_rate_{horizon}"] = float(sl / n) if n else float("nan")
        metrics[f"ambiguous_rate_{horizon}"] = float(amb / n) if n else float("nan")
        metrics[f"neither_rate_{horizon}"] = float(none / n) if n else float("nan")
        metrics[f"win_rate_ex_ambiguous_{horizon}"] = (
            float(tp / resolved) if resolved > 0 else float("nan")
        )
        denom_half = resolved + amb
        metrics[f"win_rate_with_ambiguous_half_credit_{horizon}"] = (
            float((tp + 0.5 * amb) / denom_half) if denom_half > 0 else float("nan")
        )
    return metrics


def _group_table(
    events: pd.DataFrame,
    *,
    group_col: str,
    out_col: str,
    horizons: Iterable[int],
    ordered_values: list[str] | None = None,
    sort_by_count: bool = True,
) -> pd.DataFrame:
    if events.empty or group_col not in events.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    if ordered_values is not None:
        for value in ordered_values:
            sub = events[events[group_col].astype(str) == value]
            rows.append({out_col: value, **_group_metrics(sub, horizons)})
    else:
        for value, sub in events.groupby(group_col, dropna=False, observed=False):
            label = "" if pd.isna(value) else str(value)
            if label == "" or label == "nan":
                continue
            rows.append({out_col: label, **_group_metrics(sub, horizons)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if sort_by_count:
        return table.sort_values(
            ["count", out_col], ascending=[False, True]
        ).reset_index(drop=True)
    return table.reset_index(drop=True)


def _best_group_label(
    table: pd.DataFrame,
    *,
    group_col: str,
    horizon: int,
    min_resolved: int = 5,
) -> str:
    if table.empty or group_col not in table.columns:
        return ""
    resolved_col = f"resolved_count_{horizon}"
    win_col = f"win_rate_ex_ambiguous_{horizon}"
    if resolved_col not in table.columns or win_col not in table.columns:
        return ""
    candidates = table.copy()
    candidates = candidates[
        pd.to_numeric(candidates[resolved_col], errors="coerce").fillna(0)
        >= min_resolved
    ]
    candidates = candidates.dropna(subset=[win_col])
    if candidates.empty:
        return ""
    return str(
        candidates.sort_values([win_col, resolved_col], ascending=[False, False]).iloc[
            0
        ][group_col]
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def build_atr_bracket_first_hit_diagnostics(
    df: pd.DataFrame,
    *,
    horizons: Iterable[int] = ATR_BRACKET_HORIZONS,
) -> dict[str, object]:
    """Produce the full Step 11T diagnostic bundle.

    Returns a dict of pandas tables and a headline summary suitable for
    pretty-printing in :mod:`scripts.validate_final_sweeps`.
    """

    horizons_t = tuple(int(h) for h in horizons)
    events = build_atr_bracket_events(df, horizons=horizons_t)
    if events.empty:
        empty = pd.DataFrame()
        return {
            "events": empty,
            "by_entity": empty,
            "by_sweep_class": empty,
            "by_sweep_family": empty,
            "by_swing_side": empty,
            "by_regime": empty,
            "by_session": empty,
            "summary": _empty_summary(horizons_t),
        }

    swings = events[events["signal_entity_type"] == "swing"].copy()
    sweeps = events[events["signal_entity_type"] == "sweep"].copy()

    by_entity = pd.DataFrame(
        [
            {"signal_entity_type": "swing", **_group_metrics(swings, horizons_t)},
            {"signal_entity_type": "sweep", **_group_metrics(sweeps, horizons_t)},
        ]
    )

    by_sweep_class = _group_table(
        sweeps,
        group_col="sweep_selectivity_class",
        out_col="sweep_selectivity_class",
        horizons=horizons_t,
    )
    by_sweep_family = _group_table(
        sweeps,
        group_col="sweep_primary_family",
        out_col="sweep_primary_family",
        horizons=horizons_t,
    )
    by_swing_side = _group_table(
        swings,
        group_col="swing_side",
        out_col="swing_side",
        horizons=horizons_t,
        ordered_values=["swing_high", "swing_low"],
        sort_by_count=False,
    )
    by_regime = _group_table(
        events,
        group_col="regime_label",
        out_col="regime_label",
        horizons=horizons_t,
    )
    by_session = _group_table(
        events,
        group_col="session_phase",
        out_col="session_phase",
        horizons=horizons_t,
    )

    summary = _build_headline_summary(
        swings=swings,
        sweeps=sweeps,
        by_sweep_class=by_sweep_class,
        by_sweep_family=by_sweep_family,
        by_swing_side=by_swing_side,
        horizons=horizons_t,
    )

    return {
        "events": events,
        "by_entity": by_entity,
        "by_sweep_class": by_sweep_class,
        "by_sweep_family": by_sweep_family,
        "by_swing_side": by_swing_side,
        "by_regime": by_regime,
        "by_session": by_session,
        "summary": summary,
    }


def _empty_summary(horizons: tuple[int, ...]) -> dict[str, object]:
    base: dict[str, object] = {
        "swings_total": 0,
        "sweeps_total": 0,
        "best_sweep_class_by_win_rate_5": "",
        "best_sweep_family_by_win_rate_5": "",
        "best_swing_side_by_win_rate_5": "",
        "displacement_confirmed_sweep_win_rate_5": float("nan"),
        "tradeable_sweep_candidate_win_rate_5": float("nan"),
    }
    for h in horizons:
        base[f"swings_tp_first_rate_{h}"] = float("nan")
        base[f"swings_sl_first_rate_{h}"] = float("nan")
        base[f"swings_win_rate_ex_ambiguous_{h}"] = float("nan")
        base[f"sweeps_tp_first_rate_{h}"] = float("nan")
        base[f"sweeps_sl_first_rate_{h}"] = float("nan")
        base[f"sweeps_win_rate_ex_ambiguous_{h}"] = float("nan")
    return base


def _build_headline_summary(
    *,
    swings: pd.DataFrame,
    sweeps: pd.DataFrame,
    by_sweep_class: pd.DataFrame,
    by_sweep_family: pd.DataFrame,
    by_swing_side: pd.DataFrame,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    swing_metrics = _group_metrics(swings, horizons)
    sweep_metrics = _group_metrics(sweeps, horizons)
    summary: dict[str, object] = {
        "swings_total": int(len(swings)),
        "sweeps_total": int(len(sweeps)),
    }
    for h in horizons:
        summary[f"swings_tp_first_rate_{h}"] = swing_metrics[f"tp_first_rate_{h}"]
        summary[f"swings_sl_first_rate_{h}"] = swing_metrics[f"sl_first_rate_{h}"]
        summary[f"swings_win_rate_ex_ambiguous_{h}"] = swing_metrics[
            f"win_rate_ex_ambiguous_{h}"
        ]
        summary[f"sweeps_tp_first_rate_{h}"] = sweep_metrics[f"tp_first_rate_{h}"]
        summary[f"sweeps_sl_first_rate_{h}"] = sweep_metrics[f"sl_first_rate_{h}"]
        summary[f"sweeps_win_rate_ex_ambiguous_{h}"] = sweep_metrics[
            f"win_rate_ex_ambiguous_{h}"
        ]

    summary["displacement_confirmed_sweep_win_rate_5"] = _lookup_class_metric(
        by_sweep_class,
        class_name="displacement_confirmed_sweep",
        metric="win_rate_ex_ambiguous_5",
    )
    summary["tradeable_sweep_candidate_win_rate_5"] = _lookup_class_metric(
        by_sweep_class,
        class_name="tradeable_sweep_candidate",
        metric="win_rate_ex_ambiguous_5",
    )
    summary["best_sweep_class_by_win_rate_5"] = _best_group_label(
        by_sweep_class,
        group_col="sweep_selectivity_class",
        horizon=5,
    )
    summary["best_sweep_family_by_win_rate_5"] = _best_group_label(
        by_sweep_family,
        group_col="sweep_primary_family",
        horizon=5,
    )
    summary["best_swing_side_by_win_rate_5"] = _best_group_label(
        by_swing_side,
        group_col="swing_side",
        horizon=5,
        min_resolved=1,
    )
    return summary


def _lookup_class_metric(
    table: pd.DataFrame,
    *,
    class_name: str,
    metric: str,
) -> float:
    if table.empty or "sweep_selectivity_class" not in table.columns:
        return float("nan")
    match = table[table["sweep_selectivity_class"].astype(str) == class_name]
    if match.empty or metric not in match.columns:
        return float("nan")
    val = pd.to_numeric(match.iloc[0][metric], errors="coerce")
    return float(val) if pd.notna(val) else float("nan")


def print_atr_bracket_summary(summary: dict[str, object], *, indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}=== STEP 11T ATR BRACKET FIRST-HIT AUDIT ===")
    keys_int = ("swings_total", "sweeps_total")
    keys_str = (
        "best_sweep_class_by_win_rate_5",
        "best_sweep_family_by_win_rate_5",
        "best_swing_side_by_win_rate_5",
    )
    for key in keys_int:
        print(f"{prefix}{key}: {int(summary.get(key, 0))}")
    for key in (
        "swings_tp_first_rate_5",
        "swings_sl_first_rate_5",
        "swings_win_rate_ex_ambiguous_5",
        "sweeps_tp_first_rate_5",
        "sweeps_sl_first_rate_5",
        "sweeps_win_rate_ex_ambiguous_5",
        "displacement_confirmed_sweep_win_rate_5",
        "tradeable_sweep_candidate_win_rate_5",
    ):
        val = summary.get(key, float("nan"))
        try:
            num = float(val)
        except (TypeError, ValueError):
            num = float("nan")
        formatted = "" if not np.isfinite(num) else f"{num:.4f}"
        print(f"{prefix}{key}: {formatted}")
    for key in keys_str:
        print(f"{prefix}{key}: {summary.get(key, '')}")


__all__ = [
    "ATR_BRACKET_HORIZONS",
    "ATR_BRACKET_TP_MULT",
    "ATR_BRACKET_SL_MULT",
    "build_atr_bracket_events",
    "build_atr_bracket_first_hit_diagnostics",
    "print_atr_bracket_summary",
]
