from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SWING_POST_CONFIRM_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5)
SWING_POST_CONFIRM_THRESHOLDS_ATR: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)
_LATENCY_BUCKET_LABELS: tuple[str, ...] = (
    "[0,1]",
    "[2,3]",
    "[4,5]",
    "[6,10]",
    ">10",
)
_SWING_STRENGTH_BUCKETS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, np.inf)
_VOLATILITY_BUCKETS: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, np.inf)
_DISTANCE_BUCKETS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)


def _bucket_labels(edges: tuple[float, ...]) -> list[str]:
    labels: list[str] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if np.isinf(right):
            labels.append(f"[{left:.2f}, inf)")
        else:
            labels.append(f"[{left:.2f}, {right:.2f})")
    return labels


def _bucketize(values: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    return pd.cut(values, bins=list(edges), right=False, labels=_bucket_labels(edges))


def swing_confirmation_contract() -> dict[str, str]:
    return {
        "swing_candidate_definition": (
            "A bar becomes a swing-high candidate if its high exceeds the prior "
            "lookback-window highs; likewise a swing-low candidate if its low "
            "undercuts the prior lookback-window lows."
        ),
        "confirmation_rule": (
            "A candidate is confirmed only after price retraces at least "
            "min_retrace_atr × ATR away from the candidate extreme."
        ),
        "confirmation_delay_logic": (
            "Confirmation is delayed until both the minimum confirm-bar spacing "
            "and the retrace threshold are satisfied."
        ),
        "swing_idx_columns": (
            "Origin bar uses swing_high_idx / swing_low_idx, with confirm-bar "
            "origin references in swing_high_confirm_origin_idx / "
            "swing_low_confirm_origin_idx."
        ),
        "confirm_idx_columns": (
            "Confirm bars are identified by swing_high_confirm_flag / "
            "swing_low_confirm_flag. Backward-compatible aliases are "
            "swing_high_detect_flag / swing_low_detect_flag."
        ),
        "live_activation_timing": (
            "Confirmed running state last_swing_high / last_swing_low updates on "
            "the confirm bar, and swing_high_age / swing_low_age are zero there."
        ),
        "active_from_confirm_idx_only": (
            "Yes. Swing levels become live only at confirm_idx close. This audit "
            "anchors all forward behavior from confirm_idx, not swing_idx."
        ),
    }


def _latency_bucket(latency: float) -> str:
    if not np.isfinite(latency):
        return ""
    latency_i = int(latency)
    if latency_i <= 1:
        return "[0,1]"
    if latency_i <= 3:
        return "[2,3]"
    if latency_i <= 5:
        return "[4,5]"
    if latency_i <= 10:
        return "[6,10]"
    return ">10"


def _speed_bucket(first_bar: float | None) -> str:
    if first_bar is None or not np.isfinite(first_bar):
        return "none"
    if first_bar <= 1.0:
        return "immediate"
    if first_bar <= 2.0:
        return "fast"
    if first_bar <= 4.0:
        return "normal"
    if first_bar <= 5.0:
        return "slow"
    return "none"


def _touches_threshold(
    *,
    side: str,
    ref_close: float,
    atr_value: float,
    threshold_atr: float,
    bar_high: float,
    bar_low: float,
) -> tuple[bool, bool]:
    if side == "swing_high":
        favorable_level = ref_close - threshold_atr * atr_value
        adverse_level = ref_close + threshold_atr * atr_value
        favorable = np.isfinite(bar_low) and bar_low <= favorable_level + 1e-12
        adverse = np.isfinite(bar_high) and bar_high >= adverse_level - 1e-12
    else:
        favorable_level = ref_close + threshold_atr * atr_value
        adverse_level = ref_close - threshold_atr * atr_value
        favorable = np.isfinite(bar_high) and bar_high >= favorable_level - 1e-12
        adverse = np.isfinite(bar_low) and bar_low <= adverse_level + 1e-12
    return favorable, adverse


def _first_hit_bar(
    *,
    side: str,
    ref_close: float,
    atr_value: float,
    threshold_atr: float,
    high: np.ndarray,
    low: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> tuple[float | None, float | None]:
    favorable_bar: float | None = None
    adverse_bar: float | None = None
    for delay, j in enumerate(range(start_idx, end_idx + 1), start=1):
        favorable_hit, adverse_hit = _touches_threshold(
            side=side,
            ref_close=ref_close,
            atr_value=atr_value,
            threshold_atr=threshold_atr,
            bar_high=high[j],
            bar_low=low[j],
        )
        if favorable_bar is None and favorable_hit:
            favorable_bar = float(delay)
        if adverse_bar is None and adverse_hit:
            adverse_bar = float(delay)
        if favorable_bar is not None and adverse_bar is not None:
            break
    return favorable_bar, adverse_bar


def _path_label(
    *,
    horizon: int,
    favorable_1p0: float | None,
    adverse_1p0: float | None,
    favorable_0p5: float | None,
    adverse_0p5: float | None,
) -> str:
    favorable_1p0_in = favorable_1p0 is not None and favorable_1p0 <= float(horizon)
    adverse_1p0_in = adverse_1p0 is not None and adverse_1p0 <= float(horizon)
    adverse_0p5_in = adverse_0p5 is not None and adverse_0p5 <= float(horizon)

    if favorable_1p0_in and adverse_1p0_in:
        if abs(float(favorable_1p0) - float(adverse_1p0)) <= 1e-12:
            return "two_sided_volatile"
        if float(favorable_1p0) < float(adverse_1p0):
            if (not adverse_0p5_in) or float(favorable_1p0) < float(adverse_0p5):
                return "clean_reversal"
            return "dirty_reversal"
        return "continuation"
    if favorable_1p0_in:
        if (not adverse_0p5_in) or float(favorable_1p0) < float(adverse_0p5):
            return "clean_reversal"
        return "dirty_reversal"
    if adverse_1p0_in:
        return "continuation"
    return "chop_no_resolution"


def build_swing_post_confirm_events(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "high",
        "low",
        "close",
        "atr_14",
        "swing_high_confirm_flag",
        "swing_low_confirm_flag",
        "swing_high_confirm_origin_idx",
        "swing_low_confirm_origin_idx",
        "swing_high_confirm_price",
        "swing_low_confirm_price",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing swing columns: {sorted(missing)}")

    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(df["atr_14"], errors="coerce").to_numpy(dtype=float)
    regime_label = (
        df["regime_label"].astype(str).fillna("").to_numpy(dtype=object)
        if "regime_label" in df.columns
        else np.array([""] * len(df), dtype=object)
    )
    session_phase = (
        df["session_name"].astype(str).fillna("").to_numpy(dtype=object)
        if "session_name" in df.columns
        else np.array([""] * len(df), dtype=object)
    )

    rows: list[dict[str, object]] = []
    event_id = 1
    threshold_labels = {0.25: "0p25", 0.5: "0p5", 1.0: "1p0", 1.5: "1p5"}

    for confirm_idx in range(len(df)):
        for side in ("swing_high", "swing_low"):
            flag_col = f"{side}_confirm_flag"
            if (
                int(pd.to_numeric(df.at[confirm_idx, flag_col], errors="coerce") or 0)
                != 1
            ):
                continue

            origin_col = f"{side}_confirm_origin_idx"
            price_col = f"{side}_confirm_price"
            strength_col = f"{side}_strength"
            swing_idx_num = pd.to_numeric(
                df.at[confirm_idx, origin_col], errors="coerce"
            )
            swing_price = float(
                pd.to_numeric(df.at[confirm_idx, price_col], errors="coerce")
            )
            if pd.isna(swing_idx_num) or not np.isfinite(swing_price):
                continue

            swing_idx = int(swing_idx_num)
            if swing_idx < 0 or swing_idx >= len(df):
                continue

            confirm_close = close[confirm_idx]
            confirm_high = high[confirm_idx]
            confirm_low = low[confirm_idx]
            atr_ref = atr[confirm_idx]
            latency = confirm_idx - swing_idx
            strength = (
                float(pd.to_numeric(df.at[swing_idx, strength_col], errors="coerce"))
                if strength_col in df.columns
                else float("nan")
            )
            volatility_ratio = (
                float(abs(atr_ref) / abs(confirm_close))
                if np.isfinite(atr_ref)
                and atr_ref > 0
                and np.isfinite(confirm_close)
                and abs(confirm_close) > 1e-12
                else float("nan")
            )
            distance_atr = (
                float(abs(confirm_close - swing_price) / atr_ref)
                if np.isfinite(atr_ref) and atr_ref > 0 and np.isfinite(confirm_close)
                else float("nan")
            )

            row: dict[str, object] = {
                "event_id": int(event_id),
                "swing_side": side,
                "swing_idx": int(swing_idx),
                "confirm_idx": int(confirm_idx),
                "swing_price": float(swing_price),
                "confirm_close": float(confirm_close),
                "confirm_high": float(confirm_high),
                "confirm_low": float(confirm_low),
                "atr_at_confirm": float(atr_ref),
                "confirmation_latency_bars": float(latency),
                "confirmation_latency_bucket": _latency_bucket(float(latency)),
                "swing_strength": strength,
                "regime_label": str(regime_label[confirm_idx] or ""),
                "session_phase": str(session_phase[confirm_idx] or ""),
                "volatility_ratio_at_confirm": volatility_ratio,
                "distance_confirm_to_swing_atr": distance_atr,
            }
            event_id += 1

            enough_first_hit_bars = (
                confirm_idx + max(SWING_POST_CONFIRM_HORIZONS)
            ) < len(df)
            first_hits: dict[float, tuple[float | None, float | None]] = {}
            if np.isfinite(atr_ref) and atr_ref > 0 and enough_first_hit_bars:
                for threshold in SWING_POST_CONFIRM_THRESHOLDS_ATR:
                    favorable_bar, adverse_bar = _first_hit_bar(
                        side=side,
                        ref_close=confirm_close,
                        atr_value=atr_ref,
                        threshold_atr=threshold,
                        high=high,
                        low=low,
                        start_idx=confirm_idx + 1,
                        end_idx=confirm_idx + max(SWING_POST_CONFIRM_HORIZONS),
                    )
                    first_hits[threshold] = (favorable_bar, adverse_bar)
                    label = threshold_labels[threshold]
                    row[f"first_favorable_{label}_bar"] = favorable_bar
                    row[f"first_adverse_{label}_bar"] = adverse_bar
                favorable_1p0, adverse_1p0 = first_hits[1.0]
                row["swing_reversed_by_5"] = (
                    1.0 if favorable_1p0 is not None and favorable_1p0 <= 5.0 else 0.0
                )
                row["swing_continued_by_5"] = (
                    1.0 if adverse_1p0 is not None and adverse_1p0 <= 5.0 else 0.0
                )
                row["swing_reversal_speed_bucket"] = _speed_bucket(favorable_1p0)
                row["swing_continuation_speed_bucket"] = _speed_bucket(adverse_1p0)
            else:
                for threshold in SWING_POST_CONFIRM_THRESHOLDS_ATR:
                    label = threshold_labels[threshold]
                    row[f"first_favorable_{label}_bar"] = np.nan
                    row[f"first_adverse_{label}_bar"] = np.nan
                row["swing_reversed_by_5"] = np.nan
                row["swing_continued_by_5"] = np.nan
                row["swing_reversal_speed_bucket"] = "none"
                row["swing_continuation_speed_bucket"] = "none"

            for horizon in SWING_POST_CONFIRM_HORIZONS:
                future_idx = confirm_idx + horizon
                if not (np.isfinite(atr_ref) and atr_ref > 0) or future_idx >= len(df):
                    row[f"fwd_close_ret_atr_{horizon}"] = np.nan
                    row[f"fwd_mfe_atr_{horizon}"] = np.nan
                    row[f"fwd_mae_atr_{horizon}"] = np.nan
                    row[f"fwd_net_edge_atr_{horizon}"] = np.nan
                    row[f"fwd_path_label_{horizon}"] = np.nan
                    continue

                future_close = close[future_idx]
                future_high = high[confirm_idx + 1 : future_idx + 1]
                future_low = low[confirm_idx + 1 : future_idx + 1]
                if (
                    not np.isfinite(future_close)
                    or future_high.size == 0
                    or future_low.size == 0
                ):
                    row[f"fwd_close_ret_atr_{horizon}"] = np.nan
                    row[f"fwd_mfe_atr_{horizon}"] = np.nan
                    row[f"fwd_mae_atr_{horizon}"] = np.nan
                    row[f"fwd_net_edge_atr_{horizon}"] = np.nan
                    row[f"fwd_path_label_{horizon}"] = np.nan
                    continue

                if side == "swing_high":
                    fwd_close_ret_atr = (confirm_close - future_close) / atr_ref
                    mfe = confirm_close - float(np.nanmin(future_low))
                    mae = float(np.nanmax(future_high)) - confirm_close
                else:
                    fwd_close_ret_atr = (future_close - confirm_close) / atr_ref
                    mfe = float(np.nanmax(future_high)) - confirm_close
                    mae = confirm_close - float(np.nanmin(future_low))

                mfe_atr = float(max(mfe, 0.0) / atr_ref)
                mae_atr = float(max(mae, 0.0) / atr_ref)
                row[f"fwd_close_ret_atr_{horizon}"] = float(fwd_close_ret_atr)
                row[f"fwd_mfe_atr_{horizon}"] = mfe_atr
                row[f"fwd_mae_atr_{horizon}"] = mae_atr
                row[f"fwd_net_edge_atr_{horizon}"] = float(mfe_atr - mae_atr)

                favorable_1p0: float | None
                adverse_1p0: float | None
                favorable_0p5: float | None
                adverse_0p5: float | None
                if enough_first_hit_bars and first_hits:
                    favorable_1p0, adverse_1p0 = first_hits[1.0]
                    favorable_0p5, adverse_0p5 = first_hits[0.5]
                else:
                    favorable_1p0, adverse_1p0 = _first_hit_bar(
                        side=side,
                        ref_close=confirm_close,
                        atr_value=atr_ref,
                        threshold_atr=1.0,
                        high=high,
                        low=low,
                        start_idx=confirm_idx + 1,
                        end_idx=future_idx,
                    )
                    favorable_0p5, adverse_0p5 = _first_hit_bar(
                        side=side,
                        ref_close=confirm_close,
                        atr_value=atr_ref,
                        threshold_atr=0.5,
                        high=high,
                        low=low,
                        start_idx=confirm_idx + 1,
                        end_idx=future_idx,
                    )
                row[f"fwd_path_label_{horizon}"] = _path_label(
                    horizon=horizon,
                    favorable_1p0=favorable_1p0,
                    adverse_1p0=adverse_1p0,
                    favorable_0p5=favorable_0p5,
                    adverse_0p5=adverse_0p5,
                )

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    events = (
        pd.DataFrame(rows)
        .sort_values(["confirm_idx", "event_id"])
        .reset_index(drop=True)
    )
    events["swing_strength_bucket"] = _bucketize(
        pd.to_numeric(events["swing_strength"], errors="coerce"),
        _SWING_STRENGTH_BUCKETS,
    ).astype(str)
    events["volatility_bucket"] = _bucketize(
        pd.to_numeric(events["volatility_ratio_at_confirm"], errors="coerce"),
        _VOLATILITY_BUCKETS,
    ).astype(str)
    events["distance_bucket"] = _bucketize(
        pd.to_numeric(events["distance_confirm_to_swing_atr"], errors="coerce"),
        _DISTANCE_BUCKETS,
    ).astype(str)
    return events


def _path_metrics_for_group(group: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, object] = {"count": int(len(group))}
    for horizon in SWING_POST_CONFIRM_HORIZONS:
        ret_col = f"fwd_close_ret_atr_{horizon}"
        mfe_col = f"fwd_mfe_atr_{horizon}"
        mae_col = f"fwd_mae_atr_{horizon}"
        net_col = f"fwd_net_edge_atr_{horizon}"
        path_col = f"fwd_path_label_{horizon}"
        available = group[pd.to_numeric(group[ret_col], errors="coerce").notna()].copy()
        metrics[f"mean_fwd_close_ret_atr_{horizon}"] = float(
            pd.to_numeric(available[ret_col], errors="coerce").mean()
        )
        metrics[f"median_fwd_close_ret_atr_{horizon}"] = float(
            pd.to_numeric(available[ret_col], errors="coerce").median()
        )
        metrics[f"mean_mfe_atr_{horizon}"] = float(
            pd.to_numeric(available[mfe_col], errors="coerce").mean()
        )
        metrics[f"mean_mae_atr_{horizon}"] = float(
            pd.to_numeric(available[mae_col], errors="coerce").mean()
        )
        metrics[f"mean_net_edge_atr_{horizon}"] = float(
            pd.to_numeric(available[net_col], errors="coerce").mean()
        )

        labels = available[path_col].dropna().astype(str)
        labels = labels[labels != ""]
        metrics[f"clean_reversal_rate_{horizon}"] = (
            float(labels.eq("clean_reversal").mean())
            if not labels.empty
            else float("nan")
        )
        metrics[f"dirty_reversal_rate_{horizon}"] = (
            float(labels.eq("dirty_reversal").mean())
            if not labels.empty
            else float("nan")
        )
        metrics[f"continuation_rate_{horizon}"] = (
            float(labels.eq("continuation").mean())
            if not labels.empty
            else float("nan")
        )
        metrics[f"chop_rate_{horizon}"] = (
            float(labels.eq("chop_no_resolution").mean())
            if not labels.empty
            else float("nan")
        )
        metrics[f"two_sided_volatile_rate_{horizon}"] = (
            float(labels.eq("two_sided_volatile").mean())
            if not labels.empty
            else float("nan")
        )
        clean = metrics[f"clean_reversal_rate_{horizon}"]
        dirty = metrics[f"dirty_reversal_rate_{horizon}"]
        metrics[f"reversal_rate_{horizon}"] = (
            float(clean + dirty)
            if np.isfinite(clean) and np.isfinite(dirty)
            else float("nan")
        )

        fav_bar = pd.to_numeric(available["first_favorable_1p0_bar"], errors="coerce")
        adv_bar = pd.to_numeric(available["first_adverse_1p0_bar"], errors="coerce")
        favorable_first = (
            (fav_bar.notna())
            & (adv_bar.isna() | (fav_bar < adv_bar))
            & (fav_bar <= float(horizon))
        )
        adverse_first = (
            (adv_bar.notna())
            & (fav_bar.isna() | (adv_bar < fav_bar))
            & (adv_bar <= float(horizon))
        )
        metrics[f"favorable_first_1p0_rate_{horizon}"] = (
            float(favorable_first.mean()) if len(available) else float("nan")
        )
        metrics[f"adverse_first_1p0_rate_{horizon}"] = (
            float(adverse_first.mean()) if len(available) else float("nan")
        )
    return metrics


def _group_table(
    events: pd.DataFrame,
    *,
    group_col: str,
    out_col: str,
    ordered_values: list[str] | None = None,
    sort_by_count: bool = True,
) -> pd.DataFrame:
    if events.empty or group_col not in events.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    if ordered_values is not None:
        for value in ordered_values:
            group = events[events[group_col].astype(str) == value]
            rows.append({out_col: value, **_path_metrics_for_group(group)})
    else:
        for value, group in events.groupby(group_col, dropna=False, observed=False):
            if pd.isna(value) or value in ("", "nan"):
                continue
            rows.append({out_col: str(value), **_path_metrics_for_group(group)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if sort_by_count:
        return table.sort_values(
            ["count", out_col], ascending=[False, True]
        ).reset_index(drop=True)
    return table.reset_index(drop=True)


def _extract_sweep_comparison_rows(
    swing_events: pd.DataFrame,
    sweep_diagnostics: dict[str, Any] | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _append_from_events(label: str, events: pd.DataFrame) -> None:
        if events.empty:
            rows.append({"entity": label})
            return
        metrics = _path_metrics_for_group(events)
        rows.append(
            {
                "entity": label,
                "count": int(len(events)),
                "reversal_rate_5": metrics["reversal_rate_5"],
                "continuation_rate_5": metrics["continuation_rate_5"],
                "clean_reversal_rate_5": metrics["clean_reversal_rate_5"],
                "mean_fwd_close_ret_atr_5": metrics["mean_fwd_close_ret_atr_5"],
                "mean_net_edge_atr_5": metrics["mean_net_edge_atr_5"],
            }
        )

    _append_from_events("all_confirmed_swings", swing_events)
    _append_from_events(
        "swing_high_confirmed",
        swing_events[swing_events["swing_side"] == "swing_high"],
    )
    _append_from_events(
        "swing_low_confirmed",
        swing_events[swing_events["swing_side"] == "swing_low"],
    )

    if not sweep_diagnostics:
        return pd.DataFrame(rows)

    sweep_summary = sweep_diagnostics.get("post_path_summary", {})
    if sweep_summary:
        rows.append(
            {
                "entity": "all_confirmed_sweeps",
                "count": int(sweep_summary.get("total_confirmed_sweeps", 0)),
                "reversal_rate_5": sweep_summary.get("reversal_rate_5"),
                "continuation_rate_5": sweep_summary.get("continuation_rate_5"),
                "clean_reversal_rate_5": sweep_summary.get("clean_reversal_rate_5"),
                "mean_fwd_close_ret_atr_5": sweep_summary.get(
                    "mean_fwd_close_ret_atr_5"
                ),
                "mean_net_edge_atr_5": sweep_summary.get("mean_net_edge_5"),
            }
        )

    class_table = sweep_diagnostics.get("outcome_by_class_table", pd.DataFrame())
    if isinstance(class_table, pd.DataFrame) and not class_table.empty:
        for cls in ("displacement_confirmed_sweep", "tradeable_sweep_candidate"):
            match = class_table[class_table["selectivity_class"] == cls]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "entity": cls,
                    "count": int(row["count"]),
                    "reversal_rate_5": float(row["reversal_rate_5"]),
                    "continuation_rate_5": float(row["continuation_rate_5"]),
                    "clean_reversal_rate_5": float(row["clean_reversal_rate_5"]),
                    "mean_fwd_close_ret_atr_5": float(row["mean_fwd_close_ret_atr_5"]),
                    "mean_net_edge_atr_5": float(row["mean_net_edge_atr_5"]),
                }
            )

    family_table = sweep_diagnostics.get("outcome_by_family_table", pd.DataFrame())
    if isinstance(family_table, pd.DataFrame) and not family_table.empty:
        for family in ("swing_high", "swing_low"):
            match = family_table[family_table["family"] == family]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "entity": f"sweep_family_{family}",
                    "count": int(row["count"]),
                    "reversal_rate_5": float(row["reversal_rate_5"]),
                    "continuation_rate_5": float(row["continuation_rate_5"]),
                    "clean_reversal_rate_5": float(row["clean_reversal_rate_5"]),
                    "mean_fwd_close_ret_atr_5": float(row["mean_fwd_close_ret_atr_5"]),
                    "mean_net_edge_atr_5": float(row["mean_net_edge_atr_5"]),
                }
            )

    return pd.DataFrame(rows)


def _best_group_name(table: pd.DataFrame, group_col: str) -> str:
    if table.empty or "reversal_rate_5" not in table.columns:
        return ""
    ranked = table.dropna(subset=["reversal_rate_5"])
    if ranked.empty:
        return ""
    return str(
        ranked.sort_values(
            ["reversal_rate_5", "mean_net_edge_atr_5"], ascending=[False, False]
        ).iloc[0][group_col]
    )


def build_swing_post_confirm_diagnostics(
    df: pd.DataFrame,
    *,
    sweep_diagnostics: dict[str, Any] | None = None,
) -> dict[str, object]:
    events = build_swing_post_confirm_events(df)
    if events.empty:
        return {
            "contract": swing_confirmation_contract(),
            "events": pd.DataFrame(),
            "summary": {},
            "by_side": pd.DataFrame(),
            "by_latency": pd.DataFrame(),
            "by_strength": pd.DataFrame(),
            "by_regime": pd.DataFrame(),
            "by_session": pd.DataFrame(),
            "by_volatility": pd.DataFrame(),
            "by_distance": pd.DataFrame(),
            "comparison_to_sweeps": pd.DataFrame(),
            "best_swing_side": "",
            "best_latency_bucket": "",
            "best_regime": "",
            "best_session_phase": "",
            "best_distance_bucket": "",
            "memo_markdown": "",
        }

    summary_metrics = _path_metrics_for_group(events)
    latency_quantiles = pd.to_numeric(
        events["confirmation_latency_bars"], errors="coerce"
    )
    summary = {
        "total_confirmed_swings": int(len(events)),
        "confirmed_swing_high_count": int((events["swing_side"] == "swing_high").sum()),
        "confirmed_swing_low_count": int((events["swing_side"] == "swing_low").sum()),
        "median_confirmation_latency_bars": float(latency_quantiles.median()),
        "p90_confirmation_latency_bars": float(latency_quantiles.quantile(0.90)),
    }
    for horizon in SWING_POST_CONFIRM_HORIZONS:
        for key in (
            "reversal_rate",
            "continuation_rate",
            "chop_rate",
            "mean_mfe_atr",
            "mean_mae_atr",
            "mean_net_edge_atr",
        ):
            summary[f"{key}_{horizon}"] = summary_metrics[f"{key}_{horizon}"]

    by_side = _group_table(
        events,
        group_col="swing_side",
        out_col="swing_side",
        ordered_values=["swing_high", "swing_low"],
    )
    by_latency = _group_table(
        events,
        group_col="confirmation_latency_bucket",
        out_col="confirmation_latency_bucket",
        ordered_values=list(_LATENCY_BUCKET_LABELS),
        sort_by_count=False,
    )
    by_strength = _group_table(
        events,
        group_col="swing_strength_bucket",
        out_col="swing_strength_bucket",
        sort_by_count=False,
    )
    by_regime = _group_table(events, group_col="regime_label", out_col="regime_label")
    by_session = _group_table(
        events, group_col="session_phase", out_col="session_phase"
    )
    by_volatility = _group_table(
        events,
        group_col="volatility_bucket",
        out_col="volatility_bucket",
        sort_by_count=False,
    )
    by_distance = _group_table(
        events,
        group_col="distance_bucket",
        out_col="distance_bucket",
        sort_by_count=False,
    )
    comparison = _extract_sweep_comparison_rows(events, sweep_diagnostics)

    best_swing_side = _best_group_name(by_side, "swing_side")
    best_latency_bucket = _best_group_name(by_latency, "confirmation_latency_bucket")
    best_regime = _best_group_name(by_regime, "regime_label")
    best_session_phase = _best_group_name(by_session, "session_phase")
    best_distance_bucket = _best_group_name(by_distance, "distance_bucket")

    contract = swing_confirmation_contract()
    high_events = events[events["swing_side"] == "swing_high"]
    low_events = events[events["swing_side"] == "swing_low"]
    high_metrics = _path_metrics_for_group(high_events) if not high_events.empty else {}
    low_metrics = _path_metrics_for_group(low_events) if not low_events.empty else {}
    comp_map = (
        comparison.set_index("entity")
        if not comparison.empty and "entity" in comparison.columns
        else pd.DataFrame()
    )
    tradeable_better = ""
    displacement_better = ""
    if not comparison.empty and "tradeable_sweep_candidate" in comp_map.index:
        trade_row = comp_map.loc["tradeable_sweep_candidate"]
        tradeable_better = (
            "Yes"
            if summary["reversal_rate_5"] > float(trade_row["reversal_rate_5"])
            else "No"
        )
    if not comparison.empty and "displacement_confirmed_sweep" in comp_map.index:
        disp_row = comp_map.loc["displacement_confirmed_sweep"]
        displacement_better = (
            "No"
            if summary["reversal_rate_5"] < float(disp_row["reversal_rate_5"])
            else "Yes"
        )
    memo_lines = [
        "# Step 11S Swing Post-Confirmation Edge Audit",
        "",
        "## Swing Confirmation Contract",
        f"- Candidate definition: {contract['swing_candidate_definition']}",
        f"- Confirmation rule: {contract['confirmation_rule']}",
        f"- Delay logic: {contract['confirmation_delay_logic']}",
        f"- swing_idx columns: {contract['swing_idx_columns']}",
        f"- confirm_idx columns: {contract['confirm_idx_columns']}",
        f"- Live activation timing: {contract['live_activation_timing']}",
        f"- Live from confirm_idx only: {contract['active_from_confirm_idx_only']}",
        "",
        "## Headline Read",
        f"- Total confirmed swings: {summary['total_confirmed_swings']}",
        f"- Swing highs: {summary['confirmed_swing_high_count']}",
        f"- Swing lows: {summary['confirmed_swing_low_count']}",
        f"- Median confirmation latency: {summary['median_confirmation_latency_bars']:.2f} bars",
        f"- P90 confirmation latency: {summary['p90_confirmation_latency_bars']:.2f} bars",
    ]
    for horizon in SWING_POST_CONFIRM_HORIZONS:
        memo_lines.append(
            f"- H{horizon}: reversal={summary[f'reversal_rate_{horizon}']:.4f}, "
            f"continuation={summary[f'continuation_rate_{horizon}']:.4f}, "
            f"chop={summary[f'chop_rate_{horizon}']:.4f}, "
            f"net_edge={summary[f'mean_net_edge_atr_{horizon}']:.4f}"
        )
    memo_lines.extend(
        [
            "",
            "## Best Groups",
            f"- Best swing side: {best_swing_side}",
            f"- Best latency bucket: {best_latency_bucket}",
            f"- Best regime: {best_regime}",
            f"- Best session phase: {best_session_phase}",
            f"- Best distance bucket: {best_distance_bucket}",
            "",
            "## Interpretation",
            f"- Are confirmed swings themselves reversal signals? {'Yes' if summary['reversal_rate_5'] > summary['continuation_rate_5'] else 'No'}",
            f"- Are swing highs and swing lows symmetric? {'No' if abs(float(high_metrics.get('reversal_rate_5', np.nan)) - float(low_metrics.get('reversal_rate_5', np.nan))) > 0.03 else 'Mostly'}",
            f"- Does confirmation latency damage edge? Best observed latency bucket by H5 reversal is {best_latency_bucket}.",
            f"- Are older/delayed swings still useful after confirmation? {'Only selectively' if summary['reversal_rate_5'] <= summary['continuation_rate_5'] else 'Potentially yes'}",
            f"- Should swing liquidity sources be promoted in unified liquidity ranking? {'Potentially yes' if summary['reversal_rate_5'] > summary['continuation_rate_5'] and summary['mean_net_edge_atr_5'] > 0 else 'Not yet'}",
            f"- Should sweeps treat swing-based liquidity differently? {'Potentially yes' if not comparison.empty else 'Undetermined'}",
            f"- Does standalone swing behavior outperform tradeable_sweep_candidate? {tradeable_better or 'Undetermined'}",
            f"- Does standalone swing behavior outperform displacement_confirmed_sweep? {displacement_better or 'Undetermined'}",
            "- Should Step 11I be delayed until swing precedence is re-audited? This audit is intended to answer that before tuning.",
        ]
    )
    memo = "\n".join(memo_lines) + "\n"

    return {
        "contract": contract,
        "events": events,
        "summary": summary,
        "by_side": by_side,
        "by_latency": by_latency,
        "by_strength": by_strength,
        "by_regime": by_regime,
        "by_session": by_session,
        "by_volatility": by_volatility,
        "by_distance": by_distance,
        "comparison_to_sweeps": comparison,
        "best_swing_side": best_swing_side,
        "best_latency_bucket": best_latency_bucket,
        "best_regime": best_regime,
        "best_session_phase": best_session_phase,
        "best_distance_bucket": best_distance_bucket,
        "memo_markdown": memo,
    }


__all__ = [
    "SWING_POST_CONFIRM_HORIZONS",
    "SWING_POST_CONFIRM_THRESHOLDS_ATR",
    "swing_confirmation_contract",
    "build_swing_post_confirm_events",
    "build_swing_post_confirm_diagnostics",
]
