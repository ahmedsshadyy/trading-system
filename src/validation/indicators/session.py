from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.validation.common.chart_core import (
    create_candlestick_figure,
    save_figure_html,
)

SESSION_BACKGROUND = {
    0: "rgba(64, 145, 108, 0.08)",
    1: "rgba(0, 95, 115, 0.08)",
    2: "rgba(238, 155, 0, 0.08)",
    3: "rgba(174, 32, 18, 0.08)",
    4: "rgba(108, 117, 125, 0.08)",
}

REQUIRED_SESSION_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "session_code",
    "session_name",
    "session_asia_flag",
    "session_london_flag",
    "session_overlap_flag",
    "session_ny_flag",
    "session_dead_flag",
    "is_london_open_window",
    "is_ny_open_window",
    "is_london_active_window",
    "is_ny_active_window",
    "is_dead_zone",
    "bars_since_session_open",
    "bars_remaining_in_session",
    "session_progress_frac",
    "session_high_so_far",
    "session_low_so_far",
    "is_last_bar_of_session",
    "prev_asia_high",
    "prev_asia_low",
    "prev_london_high",
    "prev_london_low",
    "prev_ny_high",
    "prev_ny_low",
    "asia_range_active_flag",
    "asia_range_complete_flag",
    "asia_range_high_final",
    "asia_range_low_final",
    "asia_range_width_final",
}

COMBO_SPECS = (
    ("r_asia_london_direction_combo_final", (0, 1)),
    ("r_london_ny_direction_combo_final", (1, 3)),
    ("r_asia_ny_direction_combo_final", (0, 3)),
)
VALID_DIRECTION_VALUES = (-1, 0, 1)
TRIPLE_SPECS = (
    (
        "r_asia_london_ny_direction_triple_final",
        "r_asia_london_ny_direction_triple_label",
        "direction_triple_distribution",
    ),
    (
        "r_asia_london_active_ny_direction_triple_final",
        "r_asia_london_active_ny_direction_triple_label",
        "direction_triple_active_london_distribution",
    ),
)


def _infer_interval_minutes(ts: pd.Series) -> int:
    diffs = ts.diff().dt.total_seconds().div(60.0)
    valid = diffs[(diffs > 0) & diffs.notna()]
    rounded = valid.round().astype(int)
    return int(rounded.value_counts().idxmax())


def _build_session_group_ids(
    session_code: np.ndarray,
    *,
    ts: pd.Series,
    interval_minutes: int,
) -> np.ndarray:
    if len(session_code) == 0:
        return np.array([], dtype=int)
    day_keys = ts.dt.floor("D").astype("int64").to_numpy()
    starts = np.ones(len(session_code), dtype=bool)
    starts[1:] = (session_code[1:] != session_code[:-1]) | (
        day_keys[1:] != day_keys[:-1]
    )
    return np.cumsum(starts) - 1


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    out: dict[str, int] = {}
    for key, value in counts.items():
        label = "NaN" if pd.isna(key) else str(key)
        out[label] = int(value)
    return out


def _session_group_rows(
    df: pd.DataFrame, interval_minutes: int
) -> list[tuple[int, int, int]]:
    codes = df["session_code"].to_numpy(dtype=int)
    groups = _build_session_group_ids(
        codes,
        ts=pd.to_datetime(df["timestamp"], utc=True),
        interval_minutes=interval_minutes,
    )
    rows: list[tuple[int, int, int]] = []
    for gid in np.unique(groups):
        positions = np.flatnonzero(groups == gid)
        rows.append((int(gid), int(positions[0]), int(positions[-1])))
    return rows


def _bars_since_reset_ok(df: pd.DataFrame) -> bool:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    interval_minutes = _infer_interval_minutes(ts)
    groups = _build_session_group_ids(
        df["session_code"].to_numpy(dtype=int),
        ts=ts,
        interval_minutes=interval_minutes,
    )
    expected = pd.Series(groups).groupby(groups).cumcount().to_numpy(dtype=int)
    actual = pd.to_numeric(df["bars_since_session_open"], errors="coerce").to_numpy(
        dtype=int
    )
    return bool(np.array_equal(expected, actual))


def _session_monotonicity_ok(df: pd.DataFrame) -> tuple[bool, bool]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    interval_minutes = _infer_interval_minutes(ts)
    groups = _build_session_group_ids(
        df["session_code"].to_numpy(dtype=int),
        ts=ts,
        interval_minutes=interval_minutes,
    )
    high = pd.to_numeric(df["session_high_so_far"], errors="coerce")
    low = pd.to_numeric(df["session_low_so_far"], errors="coerce")
    high_ok = bool(
        high.groupby(groups).apply(lambda s: s.is_monotonic_increasing).all()
    )
    low_ok = bool(low.groupby(groups).apply(lambda s: s.is_monotonic_decreasing).all())
    return high_ok, low_ok


def _previous_session_boundary_check(
    df: pd.DataFrame,
    *,
    ended_code: int,
    prefix: str,
    interval_minutes: int,
) -> dict[str, object]:
    checks = 0
    passed = 0
    first_failure: str | None = None

    groups = _session_group_rows(df, interval_minutes)
    for _, start, end in groups:
        if int(df.iloc[start]["session_code"]) != ended_code:
            continue
        if end + 1 >= len(df):
            continue

        checks += 1
        next_row = df.iloc[end + 1]
        expected_high = float(
            pd.to_numeric(df.iloc[start : end + 1]["high"], errors="coerce").max()
        )
        expected_low = float(
            pd.to_numeric(df.iloc[start : end + 1]["low"], errors="coerce").min()
        )
        actual_high = next_row[f"prev_{prefix}_high"]
        actual_low = next_row[f"prev_{prefix}_low"]
        ok = bool(np.isclose(actual_high, expected_high, equal_nan=False)) and bool(
            np.isclose(actual_low, expected_low, equal_nan=False)
        )
        if ok:
            passed += 1
        elif first_failure is None:
            first_failure = str(pd.to_datetime(next_row["timestamp"], utc=True))

    return {
        "checks": checks,
        "passed": passed,
        "all_passed": passed == checks if checks > 0 else True,
        "first_failure": first_failure,
    }


def _asia_final_leakage_check(df: pd.DataFrame) -> dict[str, object]:
    active_asia = df["session_code"] == 0
    no_final_fields = bool(
        df.loc[
            active_asia,
            ["asia_range_high_final", "asia_range_low_final", "asia_range_width_final"],
        ]
        .isna()
        .all()
        .all()
    )
    return {
        "no_final_fields_during_active_asia": no_final_fields,
        "active_flag_consistent": bool(
            (df.loc[active_asia, "asia_range_active_flag"] == 1).all()
        ),
        "complete_flag_suppressed": bool(
            (df.loc[active_asia, "asia_range_complete_flag"] == 0).all()
        ),
    }


def _window_semantics_audit(
    df: pd.DataFrame, interval_minutes: int
) -> dict[str, object]:
    start_minutes = pd.to_datetime(df["timestamp"], utc=True).dt.hour.to_numpy(
        dtype=int
    ) * 60 + pd.to_datetime(df["timestamp"], utc=True).dt.minute.to_numpy(dtype=int)
    london_open_start = ((start_minutes >= 8 * 60) & (start_minutes < 10 * 60)).sum()
    ny_open_start = ((start_minutes >= 13 * 60) & (start_minutes < 15 * 60)).sum()
    london_active_start = ((start_minutes >= 8 * 60) & (start_minutes < 17 * 60)).sum()
    ny_active_start = ((start_minutes >= 13 * 60) & (start_minutes < 22 * 60)).sum()

    return {
        "interval_minutes": interval_minutes,
        "session_identity_semantics": "bar_start",
        "open_window_semantics": "bar_overlap",
        "active_window_semantics": "bar_overlap",
        "bar_start_reference_counts": {
            "london_open": int(london_open_start),
            "ny_open": int(ny_open_start),
            "london_active": int(london_active_start),
            "ny_active": int(ny_active_start),
        },
        "overlap_counts": {
            "london_open": int(df["is_london_open_window"].sum()),
            "ny_open": int(df["is_ny_open_window"].sum()),
            "london_active": int(df["is_london_active_window"].sum()),
            "ny_active": int(df["is_ny_active_window"].sum()),
        },
        "overlap_differs_from_bar_start": {
            "london_open": int(df["is_london_open_window"].sum())
            != int(london_open_start),
            "ny_open": int(df["is_ny_open_window"].sum()) != int(ny_open_start),
            "london_active": int(df["is_london_active_window"].sum())
            != int(london_active_start),
            "ny_active": int(df["is_ny_active_window"].sum()) != int(ny_active_start),
        },
    }


def _research_combo_summary(df: pd.DataFrame) -> dict[str, object] | None:
    if "r_session_direction_final" not in df.columns:
        return None

    ts = pd.to_datetime(df["timestamp"], utc=True)
    interval_minutes = _infer_interval_minutes(ts)
    groups = _session_group_rows(df, interval_minutes)
    group_ids = _build_session_group_ids(
        df["session_code"].to_numpy(dtype=int),
        ts=ts,
        interval_minutes=interval_minutes,
    )
    completed = {0: False, 1: False, 3: False}
    expected_available = {
        name: np.zeros(len(df), dtype=bool) for name, _ in COMBO_SPECS
    }

    group_end_to_code = {
        end: int(df.iloc[start]["session_code"]) for _, start, end in groups
    }
    for pos in range(len(df)):
        prev_end = pos - 1
        if prev_end in group_end_to_code:
            prev_code = group_end_to_code[prev_end]
            if prev_code in completed:
                completed[prev_code] = True
        for name, (left_code, right_code) in COMBO_SPECS:
            expected_available[name][pos] = (
                completed[left_code] and completed[right_code]
            )

    output: dict[str, object] = {
        "session_direction_final_counts": _value_counts(
            df["r_session_direction_final"]
        ),
    }
    for name, _ in COMBO_SPECS:
        actual_non_null = df[name].notna().to_numpy()
        output[name] = {
            "value_counts": _value_counts(df[name]),
            "no_premature_values": bool(
                (~actual_non_null | expected_available[name]).all()
            ),
            "non_null_count": int(actual_non_null.sum()),
        }

    expected_non_null = np.zeros(len(df), dtype=bool)
    completed_by_day: dict[pd.Timestamp, set[int]] = {}
    day_keys = pd.to_datetime(df["timestamp"], utc=True).dt.floor("D")
    group_end_to_meta = {
        end: (int(df.iloc[start]["session_code"]), pd.Timestamp(day_keys.iloc[start]))
        for _, start, end in groups
    }
    for pos in range(len(df)):
        prev_end = pos - 1
        if prev_end in group_end_to_meta:
            prev_code, prev_day = group_end_to_meta[prev_end]
            if prev_code in (0, 1, 3):
                completed_by_day.setdefault(prev_day, set()).add(prev_code)
                if prev_code == 3 and completed_by_day[prev_day] >= {0, 1, 3}:
                    expected_non_null[pos] = True

    valid_state_space = {
        f"{a}_{b}_{c}"
        for a in VALID_DIRECTION_VALUES
        for b in VALID_DIRECTION_VALUES
        for c in VALID_DIRECTION_VALUES
    }
    for triple_col, triple_label_col, output_key in TRIPLE_SPECS:
        if triple_col not in df.columns:
            continue
        actual_non_null = df[triple_col].notna().to_numpy()
        observed_values = {
            str(value) for value in df.loc[df[triple_col].notna(), triple_col].unique()
        }
        counts = (
            df.loc[df[triple_col].notna(), triple_col]
            .astype(str)
            .value_counts()
            .sort_index()
        )
        total = int(counts.sum())
        output[output_key] = {
            "value_counts": {str(key): int(value) for key, value in counts.items()},
            "percentages": (
                {str(key): float(value / total) for key, value in counts.items()}
                if total > 0
                else {}
            ),
            "total_completed_day_records": total,
            "no_premature_values": bool((~actual_non_null | expected_non_null).all()),
            "all_values_in_valid_27_state_space": observed_values <= valid_state_space,
            "label_value_counts": (
                _value_counts(df[triple_label_col])
                if triple_label_col in df.columns
                else {}
            ),
        }
    return output


def summarize_session_features(
    df: pd.DataFrame,
    *,
    live_df: pd.DataFrame | None = None,
    parity_ok: bool | None = None,
) -> dict[str, object]:
    missing = REQUIRED_SESSION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing session validation columns: {sorted(missing)}")

    ts = pd.to_datetime(df["timestamp"], utc=True)
    interval_minutes = _infer_interval_minutes(ts)
    session_flags = df[
        [
            "session_asia_flag",
            "session_london_flag",
            "session_overlap_flag",
            "session_ny_flag",
            "session_dead_flag",
        ]
    ]
    exclusivity_ok = bool((session_flags.sum(axis=1) == 1).all())
    high_mono_ok, low_mono_ok = _session_monotonicity_ok(df)
    boundary_asia = _previous_session_boundary_check(
        df,
        ended_code=0,
        prefix="asia",
        interval_minutes=interval_minutes,
    )
    boundary_london = _previous_session_boundary_check(
        df,
        ended_code=1,
        prefix="london",
        interval_minutes=interval_minutes,
    )
    boundary_ny = _previous_session_boundary_check(
        df,
        ended_code=3,
        prefix="ny",
        interval_minutes=interval_minutes,
    )
    asia_leakage = _asia_final_leakage_check(df)
    research_summary = _research_combo_summary(df)
    no_research_cols_in_live = (
        not any(col.startswith("r_") for col in live_df.columns)
        if live_df is not None
        else None
    )

    checks = {
        "session_exclusivity_ok": exclusivity_ok,
        "dead_zone_matches_session_dead": bool(
            (df["is_dead_zone"] == df["session_dead_flag"]).all()
        ),
        "bars_since_session_open_reset_ok": _bars_since_reset_ok(df),
        "session_high_so_far_monotonic": high_mono_ok,
        "session_low_so_far_monotonic": low_mono_ok,
        "session_progress_frac_valid": bool(
            df["session_progress_frac"].between(0.0, 1.0, inclusive="left").all()
        ),
        "bars_remaining_non_negative": bool(
            (df["bars_remaining_in_session"] >= 0).all()
        ),
        "no_asia_final_leakage": all(asia_leakage.values()),
        "no_research_cols_in_live": no_research_cols_in_live,
        "parity_ok": parity_ok,
    }
    availability_all = (
        boundary_asia["all_passed"]
        and boundary_london["all_passed"]
        and boundary_ny["all_passed"]
    )
    checks["previous_session_boundary_checks_ok"] = availability_all

    active_safe = all(
        bool(checks[key])
        for key in (
            "session_exclusivity_ok",
            "dead_zone_matches_session_dead",
            "bars_since_session_open_reset_ok",
            "session_high_so_far_monotonic",
            "session_low_so_far_monotonic",
            "session_progress_frac_valid",
            "bars_remaining_non_negative",
            "no_asia_final_leakage",
            "previous_session_boundary_checks_ok",
        )
    )
    model_safe = active_safe and bool(parity_ok) and bool(no_research_cols_in_live)

    return {
        "window": {
            "start": str(ts.min()),
            "end": str(ts.max()),
            "rows": int(len(df)),
        },
        "session_counts": {
            str(key): int(value)
            for key, value in df["session_name"].value_counts().sort_index().items()
        },
        "open_window_counts": {
            "london_open": int(df["is_london_open_window"].sum()),
            "ny_open": int(df["is_ny_open_window"].sum()),
            "london_active": int(df["is_london_active_window"].sum()),
            "ny_active": int(df["is_ny_active_window"].sum()),
            "dead_zone": int(df["is_dead_zone"].sum()),
        },
        "checks": checks,
        "previous_session_boundary_checks": {
            "asia": boundary_asia,
            "london": boundary_london,
            "ny": boundary_ny,
        },
        "asia_leakage_checks": asia_leakage,
        "window_semantics_audit": _window_semantics_audit(df, interval_minutes),
        "research_direction_summary": research_summary,
        "audit_classification": {
            "annotation_safe": bool(exclusivity_ok),
            "detect_safe": bool(availability_all),
            "confirm_safe": bool(availability_all),
            "active_safe": bool(active_safe),
            "model_safe": bool(model_safe),
            "research_only_model_safe": False,
        },
    }


def _add_session_background(fig: go.Figure, df: pd.DataFrame) -> None:
    if df.empty:
        return

    codes = df["session_code"].to_numpy(dtype=int)
    x = pd.to_datetime(df["timestamp"], utc=True).reset_index(drop=True)

    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or codes[i] != codes[start]:
            fig.add_vrect(
                x0=x.iloc[start],
                x1=x.iloc[i - 1],
                fillcolor=SESSION_BACKGROUND.get(int(codes[start]), "rgba(0,0,0,0.03)"),
                line_width=0,
                layer="below",
            )
            start = i


def _add_level_line(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    column: str,
    name: str,
    color: str,
    dash: str = "dash",
) -> None:
    if column not in df.columns:
        return

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[column],
            mode="lines",
            name=name,
            line=dict(color=color, dash=dash, width=1.5),
        )
    )


def _add_open_window_markers(fig: go.Figure, df: pd.DataFrame) -> None:
    for col, name, color, symbol in (
        ("is_london_open_window", "London Open Window", "#005f73", "triangle-up"),
        ("is_ny_open_window", "NY Open Window", "#ae2012", "triangle-down"),
    ):
        sub = df[df[col] == 1]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp"],
                y=sub["close"],
                mode="markers",
                name=name,
                marker=dict(color=color, symbol=symbol, size=9),
            )
        )


def validate_session(
    plot_df: pd.DataFrame,
    *,
    summary_df: pd.DataFrame | None = None,
    live_df: pd.DataFrame | None = None,
    outpath: str | Path,
    title: str = "Session Validation",
    parity_ok: bool | None = None,
) -> dict[str, object]:
    summary_source = summary_df if summary_df is not None else plot_df
    summary = summarize_session_features(
        summary_source,
        live_df=live_df,
        parity_ok=parity_ok,
    )

    fig = create_candlestick_figure(plot_df, title=title)
    _add_session_background(fig, plot_df)
    _add_level_line(
        fig, plot_df, column="prev_asia_high", name="Prev Asia High", color="#2a9d8f"
    )
    _add_level_line(
        fig, plot_df, column="prev_asia_low", name="Prev Asia Low", color="#2a9d8f"
    )
    _add_level_line(
        fig,
        plot_df,
        column="prev_london_high",
        name="Prev London High",
        color="#005f73",
    )
    _add_level_line(
        fig, plot_df, column="prev_london_low", name="Prev London Low", color="#005f73"
    )
    _add_level_line(
        fig, plot_df, column="prev_ny_high", name="Prev NY High", color="#ae2012"
    )
    _add_level_line(
        fig, plot_df, column="prev_ny_low", name="Prev NY Low", color="#ae2012"
    )
    _add_open_window_markers(fig, plot_df)

    fig.update_layout(legend=dict(orientation="h"))
    html_path = save_figure_html(fig, outpath)
    return {
        "summary": summary,
        "html_path": html_path,
    }
