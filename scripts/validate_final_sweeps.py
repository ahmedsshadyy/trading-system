"""Validation runner for final sweeps v1 (Step 11).

Usage::

    poetry run python scripts/validate_final_sweeps.py
    poetry run python scripts/validate_final_sweeps.py --instrument USOIL --timeframe H4
    poetry run python scripts/validate_final_sweeps.py --runs XAU_USD:H4,XAU_USD:H1

The validator runs the full research pipeline (which now includes the
``unified_liquidity_sources`` and ``final_sweeps`` stages) and prints the
canonical summary defined in :mod:`src.validation.indicators.final_sweeps`.

Outputs per (instrument, timeframe):
* stdout summary
* CSV: ``notebooks/sweeps_v2/final_sweeps_events_{instrument}_{timeframe}.csv``
* CSV: ``notebooks/sweeps_v2/final_sweeps_breaches_{instrument}_{timeframe}.csv``
* CSV: ``notebooks/sweeps_v2/final_sweeps_unified_sources_audit_{instrument}_{timeframe}.csv``
* HTML: ``notebooks/sweeps_v2/final_sweeps_chart_{instrument}_{timeframe}.html``
* HTML: refresh any existing dated ``final_sweeps_chart_{instrument}_{timeframe}_YYYY-MM-DD_to_end.html``
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.smc.sweeps.final_sweeps import (
    FINAL_SWEEPS_COLUMNS,
    add_final_sweeps,
    step11d_default_kwargs,
    step11e_default_kwargs,
    step11e_profile_kwargs,
)
from src.indicators.smc.sweeps.mtf_policy import mtf_policy_summary
from src.indicators.smc.sweeps.unified_sources import (
    build_unified_liquidity_clusters_audit,
)
from src.validation.indicators.final_sweeps import (
    build_final_sweeps_diagnostics,
    print_final_sweeps_summary,
    summarize_final_sweeps,
)
from src.validation.indicators.swing_post_confirm import (
    build_swing_post_confirm_diagnostics,
)
from src.validation.indicators.atr_bracket_first_hit import (
    build_atr_bracket_first_hit_diagnostics,
    print_atr_bracket_summary,
)
from src.validation.indicators.bracket_matrix_confluence import (
    build_step11u_diagnostics,
    print_step11u_summary,
)
from src.validation.indicators.final_sweeps_chart import render_final_sweeps_chart
from src.validation.indicators.unified_sources import (
    print_unified_sources_summary,
    summarize_unified_sources,
)

DEFAULT_OUT_DIR = Path("notebooks/sweeps_v2")
DEFAULT_RUNS = (
    ("XAU_USD", "H4"),
    ("XAU_USD", "H1"),
    ("USOIL", "H4"),
    ("EUR_USD", "H4"),
)
_DATED_CHART_RE = re.compile(
    r"^final_sweeps_chart_(?P<instrument>.+)_(?P<timeframe>[^_]+)_(?P<start>\d{4}-\d{2}-\d{2})_to_end\.html$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instrument", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument(
        "--runs",
        default=None,
        help=(
            "Comma-separated INSTRUMENT:TIMEFRAME tuples, e.g. "
            "XAU_USD:H4,XAU_USD:H1,USOIL:H4. Overrides --instrument/--timeframe."
        ),
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for fast local runs.",
    )
    p.add_argument(
        "--frame-cache",
        action="store_true",
        help=(
            "Cache the fully-built research frame at "
            "notebooks/sweeps_v2/_frame_cache_<INSTRUMENT>_<TIMEFRAME>.parquet "
            "and reuse it when the raw parquet's mtime/size hasn't changed. "
            "Skips the ~30s of pre-existing non-materialized Class B stages "
            "(fvg_stack, trend_state, equal_hl, regime, ...) on warm re-runs."
        ),
    )
    p.add_argument(
        "--no-frame-cache",
        dest="frame_cache",
        action="store_false",
        help="Force a full pipeline rebuild even if the frame cache is valid.",
    )
    p.set_defaults(frame_cache=True)
    return p.parse_args()


def _resolve_runs(args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    if args.runs:
        out: list[tuple[str, str]] = []
        for chunk in args.runs.split(","):
            parts = chunk.strip().split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"--runs entry must be INSTRUMENT:TIMEFRAME, got {chunk!r}"
                )
            out.append((parts[0], parts[1]))
        return out
    if args.instrument and args.timeframe:
        return [(args.instrument, args.timeframe)]
    return list(DEFAULT_RUNS)


def _frame_cache_path(out_dir: Path, instrument: str, timeframe: str) -> Path:
    return out_dir / f"_frame_cache_{instrument}_{timeframe}.parquet"


def _frame_cache_meta_path(out_dir: Path, instrument: str, timeframe: str) -> Path:
    return out_dir / f"_frame_cache_{instrument}_{timeframe}.meta.json"


def _try_load_cached_frame(
    *,
    data_file: Path,
    out_dir: Path,
    instrument: str,
    timeframe: str,
    max_rows: int | None,
) -> pd.DataFrame | None:
    """Return a cached frame iff the raw parquet's identity (size + mtime
    + max_rows) matches what was last cached. Otherwise returns None."""

    import json

    frame_path = _frame_cache_path(out_dir, instrument, timeframe)
    meta_path = _frame_cache_meta_path(out_dir, instrument, timeframe)
    if not frame_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    stat = data_file.stat()
    expected = {
        "raw_size": stat.st_size,
        "raw_mtime_ns": stat.st_mtime_ns,
        "max_rows": max_rows,
        "feature_contract_version": meta.get("feature_contract_version"),
    }
    if any(
        meta.get(k) != v for k, v in expected.items() if k != "feature_contract_version"
    ):
        return None
    # Also re-check feature contract version: if it has changed since the
    # cache was written, force rebuild.
    from src.indicators.pipelines.build_research import (
        RESEARCH_FEATURE_CONTRACT_VERSION,
    )

    if meta.get("feature_contract_version") != RESEARCH_FEATURE_CONTRACT_VERSION:
        return None
    return pd.read_parquet(frame_path)


def _save_cached_frame(
    df: pd.DataFrame,
    *,
    data_file: Path,
    out_dir: Path,
    instrument: str,
    timeframe: str,
    max_rows: int | None,
) -> None:
    import json
    from src.indicators.pipelines.build_research import (
        RESEARCH_FEATURE_CONTRACT_VERSION,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stat = data_file.stat()
    df.to_parquet(_frame_cache_path(out_dir, instrument, timeframe), index=False)
    meta = {
        "raw_size": stat.st_size,
        "raw_mtime_ns": stat.st_mtime_ns,
        "max_rows": max_rows,
        "feature_contract_version": RESEARCH_FEATURE_CONTRACT_VERSION,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }
    _frame_cache_meta_path(out_dir, instrument, timeframe).write_text(
        json.dumps(meta, indent=2)
    )


def _profile_metric_row(
    summary: dict[str, object],
    frame: pd.DataFrame,
    label: str,
) -> dict[str, object]:
    follow = summary.get("followthrough_overlap_count", {})
    classes = summary.get("sweeps_by_class", {})
    selectivity = (
        frame.loc[frame["sweep_flag"].fillna(0) > 0, "sweep_selectivity_class"]
        .astype(str)
        .value_counts()
        .to_dict()
        if "sweep_selectivity_class" in frame.columns
        else {}
    )
    return {
        "profile": label,
        "confirmed_sweeps_count": int(summary.get("confirmed_sweeps_count", 0)),
        "accepted_breakouts_count": int(summary.get("accepted_breakouts_count", 0)),
        "unresolved_breaches_count": int(summary.get("unresolved_breaches_count", 0)),
        "same_bar_sweeps": int(summary.get("same_bar_sweeps", 0)),
        "delayed_rejection_sweeps": int(summary.get("delayed_rejection_sweeps", 0)),
        "micro_interaction_sweeps": int(selectivity.get("micro_interaction_sweep", 0)),
        "standard_liquidity_sweeps": int(
            selectivity.get("standard_liquidity_sweep", 0)
        ),
        "displacement_confirmed_sweeps": int(
            selectivity.get("displacement_confirmed_sweep", 0)
        ),
        "tradeable_sweep_candidate_class": int(
            selectivity.get("tradeable_sweep_candidate", 0)
        ),
        "tradeable_candidates": int(summary.get("tradeable_candidates", 0)),
        "tradeable_rate": float(summary.get("tradeable_rate", 0.0)),
        "followed_by_bos": int(follow.get("sweep_followed_by_bos", 0)),
        "followed_by_choch": int(follow.get("sweep_followed_by_choch", 0)),
        "followed_by_displacement": int(
            follow.get("sweep_followed_by_displacement", 0)
        ),
        "failed_breakout_reclaim": int(classes.get("failed_breakout_reclaim", 0)),
        "causality_violations": int(summary.get("causality_violations", 0)),
    }


def _print_table(name: str, table: pd.DataFrame, indent: int = 2) -> None:
    prefix = " " * indent
    print(f"\n{name}:")
    if table is None or table.empty:
        print(f"{prefix}<empty>")
        return
    printable = table.copy()
    for col in printable.columns:
        if pd.api.types.is_float_dtype(printable[col]):
            printable[col] = printable[col].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.4f}"
            )
    print(
        printable.to_string(
            index=False,
            max_colwidth=48,
        ).replace("\n", f"\n{prefix}")
    )


def _write_table(
    out_dir: Path, stem: str, instrument: str, timeframe: str, table: pd.DataFrame
) -> Path:
    path = out_dir / f"{stem}_{instrument}_{timeframe}.csv"
    table.to_csv(path, index=False)
    return path


def _render_final_sweeps_charts(
    *,
    df: pd.DataFrame,
    out_dir: Path,
    instrument: str,
    timeframe: str,
) -> list[Path]:
    """Refresh final-sweeps charts after a validator run.

    Behavior:
    * always render a canonical full-range chart
    * also refresh any existing dated ``*_YYYY-MM-DD_to_end.html`` charts for
      the same instrument/timeframe, preserving their slice semantics
    """

    if df is None or df.empty or "timestamp" not in df.columns:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    base_chart = out_dir / f"final_sweeps_chart_{instrument}_{timeframe}.html"
    rendered.append(
        render_final_sweeps_chart(
            df,
            out_path=base_chart,
            title=f"Final Sweeps — {instrument} {timeframe} — full range",
        )
    )

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    candidates = sorted(
        out_dir.glob(f"final_sweeps_chart_{instrument}_{timeframe}_*_to_end.html")
    )
    for path in candidates:
        match = _DATED_CHART_RE.match(path.name)
        if match is None:
            continue
        start = pd.Timestamp(match.group("start"), tz="UTC")
        sliced = df.loc[ts >= start].copy()
        if sliced.empty:
            continue
        rendered.append(
            render_final_sweeps_chart(
                sliced,
                out_path=path,
                title=f"Final Sweeps — {instrument} {timeframe} — {match.group('start')} to end",
            )
        )
    return rendered


def _run_single(
    instrument: str,
    timeframe: str,
    *,
    out_dir: Path,
    max_rows: int | None,
    frame_cache: bool,
) -> int:
    data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
    if not data_file.exists():
        print(f"  SKIP: input data not found: {data_file}")
        return 0

    df = None
    if frame_cache:
        df = _try_load_cached_frame(
            data_file=data_file,
            out_dir=out_dir,
            instrument=instrument,
            timeframe=timeframe,
            max_rows=max_rows,
        )
        if df is not None:
            print(
                f"\n[frame-cache HIT] reused cached frame for {instrument} {timeframe}"
            )

    if df is None:
        raw = pd.read_parquet(data_file)
        if max_rows is not None:
            raw = raw.tail(int(max_rows)).reset_index(drop=True)
        df = build_research_indicators(
            raw,
            instrument=instrument,
            include_vp=False,
            include_avwap=False,
            timeframe=timeframe,
        )
        if frame_cache:
            _save_cached_frame(
                df,
                data_file=data_file,
                out_dir=out_dir,
                instrument=instrument,
                timeframe=timeframe,
                max_rows=max_rows,
            )

    print(
        f"\n=== final sweeps validation — {instrument} {timeframe} (rows={len(df)}) ==="
    )
    print("\nMTF policy stamp:")
    for k, v in mtf_policy_summary(timeframe).items():
        print(f"  {k}: {v}")

    print("\n--- unified liquidity sources ---")
    us_summary = summarize_unified_sources(df, scan_timeframe=timeframe)
    print_unified_sources_summary(us_summary, indent=2)

    print("\n--- final sweeps ---")
    step11d_df = add_final_sweeps(df, **step11d_default_kwargs(cooldown_bars=10))
    tested_profiles: list[tuple[str, pd.DataFrame]] = [
        ("step11d_current", step11d_df),
        (
            "step11e_dist0.25_exc0.50_rej0.60",
            add_final_sweeps(
                df,
                **step11e_profile_kwargs(
                    cooldown_bars=10,
                    standard_min_pre_breach_distance_atr=0.25,
                    standard_exceptional_penetration_atr=0.50,
                    standard_exceptional_rejection_component=0.60,
                    tradeable_min_rejection_component=0.60,
                ),
            ),
        ),
        (
            "step11e_dist0.25_exc0.50_rej0.65",
            add_final_sweeps(
                df,
                **step11e_profile_kwargs(
                    cooldown_bars=10,
                    standard_min_pre_breach_distance_atr=0.25,
                    standard_exceptional_penetration_atr=0.50,
                    standard_exceptional_rejection_component=0.65,
                    tradeable_min_rejection_component=0.60,
                ),
            ),
        ),
        (
            "step11e_dist0.25_exc0.60_rej0.65",
            add_final_sweeps(
                df,
                **step11e_profile_kwargs(
                    cooldown_bars=10,
                    standard_min_pre_breach_distance_atr=0.25,
                    standard_exceptional_penetration_atr=0.60,
                    standard_exceptional_rejection_component=0.65,
                    tradeable_min_rejection_component=0.65,
                    tradeable_max_active_sources=7,
                ),
            ),
        ),
        (
            "step11e_default_chosen",
            add_final_sweeps(df, **step11e_default_kwargs(cooldown_bars=10)),
        ),
    ]
    profile_summaries = [
        (label, frame, summarize_final_sweeps(frame, scan_timeframe=timeframe))
        for label, frame in tested_profiles
    ]
    fs_df = next(
        frame
        for label, frame, _ in profile_summaries
        if label == "step11e_default_chosen"
    )
    fs_summary = next(
        summary
        for label, _, summary in profile_summaries
        if label == "step11e_default_chosen"
    )
    print_final_sweeps_summary(fs_summary, indent=2)

    comparison = pd.DataFrame(
        [
            _profile_metric_row(summary, frame, label)
            for label, frame, summary in profile_summaries
        ]
    )

    diagnostics = build_final_sweeps_diagnostics(fs_df)
    swing_diagnostics = build_swing_post_confirm_diagnostics(
        df,
        sweep_diagnostics=diagnostics,
    )
    family_funnel = diagnostics["family_funnel"]
    age_buckets = diagnostics["age_buckets"]
    strength_buckets = diagnostics["strength_buckets"]
    distance_buckets = diagnostics["distance_buckets"]
    selectivity_class_table = diagnostics["selectivity_class_table"]
    selectivity_family_table = diagnostics["selectivity_family_table"]
    selectivity_distance_table = diagnostics["selectivity_distance_table"]
    selectivity_penetration_table = diagnostics["selectivity_penetration_table"]
    selectivity_quality_table = diagnostics["selectivity_quality_table"]
    session_funnel = diagnostics["session_funnel"]
    session_age_buckets = diagnostics["session_age_buckets"]
    session_distance_buckets = diagnostics["session_distance_buckets"]
    session_penetration_buckets = diagnostics["session_penetration_buckets"]
    session_phase_table = diagnostics["session_phase_table"]
    session_source_type_table = diagnostics["session_source_type_table"]
    repeated_sweeps = diagnostics["repeated_sweeps"]
    family_group_rates = diagnostics["family_group_rates"]
    source_events = diagnostics["source_events"]
    post_path_summary = diagnostics["post_path_summary"]
    outcome_by_class_table = diagnostics["outcome_by_class_table"]
    outcome_by_family_table = diagnostics["outcome_by_family_table"]
    outcome_by_side_table = diagnostics["outcome_by_side_table"]
    outcome_by_regime_table = diagnostics["outcome_by_regime_table"]
    outcome_by_session_phase_table = diagnostics["outcome_by_session_phase_table"]
    outcome_by_quality_bucket_table = diagnostics["outcome_by_quality_bucket_table"]
    outcome_by_source_age_bucket_table = diagnostics[
        "outcome_by_source_age_bucket_table"
    ]
    outcome_by_distance_bucket_table = diagnostics["outcome_by_distance_bucket_table"]
    best_reversal_group_by_class = diagnostics["best_reversal_group_by_class"]
    worst_continuation_group_by_class = diagnostics["worst_continuation_group_by_class"]
    best_reversal_family = diagnostics["best_reversal_family"]
    worst_continuation_family = diagnostics["worst_continuation_family"]
    swing_contract = swing_diagnostics["contract"]
    swing_summary = swing_diagnostics["summary"]
    swing_events = swing_diagnostics["events"]
    swing_by_side = swing_diagnostics["by_side"]
    swing_by_latency = swing_diagnostics["by_latency"]
    swing_by_strength = swing_diagnostics["by_strength"]
    swing_by_regime = swing_diagnostics["by_regime"]
    swing_by_session = swing_diagnostics["by_session"]
    swing_by_volatility = swing_diagnostics["by_volatility"]
    swing_by_distance = swing_diagnostics["by_distance"]
    swing_comparison = swing_diagnostics["comparison_to_sweeps"]
    best_swing_side = swing_diagnostics["best_swing_side"]
    best_latency_bucket = swing_diagnostics["best_latency_bucket"]
    best_regime = swing_diagnostics["best_regime"]
    best_session_phase = swing_diagnostics["best_session_phase"]
    best_distance_bucket = swing_diagnostics["best_distance_bucket"]
    swing_memo_markdown = swing_diagnostics["memo_markdown"]

    _print_table("before_after", comparison)
    _print_table("family_conversion", family_funnel)
    _print_table("session_funnel", session_funnel)
    _print_table("source_age_buckets", age_buckets)
    _print_table("source_strength_buckets", strength_buckets)
    _print_table("nearest_distance_buckets", distance_buckets)
    _print_table("selectivity_class_table", selectivity_class_table)
    _print_table("selectivity_family_table", selectivity_family_table)
    _print_table("selectivity_distance_table", selectivity_distance_table)
    _print_table("selectivity_penetration_table", selectivity_penetration_table)
    _print_table("selectivity_quality_table", selectivity_quality_table)
    _print_table("session_age_buckets", session_age_buckets)
    _print_table("session_distance_buckets", session_distance_buckets)
    _print_table("session_penetration_buckets", session_penetration_buckets)
    _print_table("session_phase_table", session_phase_table)
    _print_table("session_source_type_table", session_source_type_table)
    _print_table("repeated_source_cooldown_impact", repeated_sweeps)
    _print_table("family_group_sweep_rates", family_group_rates)
    print("\n=== STEP 11H POST-SWEEP PATH BEHAVIOR ===")
    for key, value in post_path_summary.items():
        print(f"{key}: {value}")
    print(f"best_reversal_group_by_class: {best_reversal_group_by_class}")
    print(f"worst_continuation_group_by_class: {worst_continuation_group_by_class}")
    print(f"best_reversal_family: {best_reversal_family}")
    print(f"worst_continuation_family: {worst_continuation_family}")
    _print_table("step11h_outcome_by_class", outcome_by_class_table)
    _print_table("step11h_outcome_by_family", outcome_by_family_table)
    _print_table("step11h_outcome_by_side", outcome_by_side_table)
    _print_table("step11h_outcome_by_regime", outcome_by_regime_table)
    _print_table("step11h_outcome_by_session_phase", outcome_by_session_phase_table)
    _print_table("step11h_outcome_by_quality_bucket", outcome_by_quality_bucket_table)
    _print_table(
        "step11h_outcome_by_source_age_bucket", outcome_by_source_age_bucket_table
    )
    _print_table("step11h_outcome_by_distance_bucket", outcome_by_distance_bucket_table)
    print("\n=== STEP 11S SWING POST-CONFIRMATION EDGE AUDIT ===")
    print("\nswing_confirmation_contract:")
    for key, value in swing_contract.items():
        print(f"  {key}: {value}")
    for key, value in swing_summary.items():
        print(f"{key}: {value}")
    print(f"best_swing_side: {best_swing_side}")
    print(f"best_latency_bucket: {best_latency_bucket}")
    print(f"best_regime: {best_regime}")
    print(f"best_session_phase: {best_session_phase}")
    print(f"best_distance_bucket: {best_distance_bucket}")
    _print_table("step11s_by_side", swing_by_side)
    _print_table("step11s_by_latency", swing_by_latency)
    _print_table("step11s_by_strength", swing_by_strength)
    _print_table("step11s_by_regime", swing_by_regime)
    _print_table("step11s_by_session", swing_by_session)
    _print_table("step11s_by_volatility", swing_by_volatility)
    _print_table("step11s_by_distance", swing_by_distance)
    _print_table("step11s_comparison_to_sweeps", swing_comparison)
    print("\nnegative_nearest_distance_explanation:")
    print(f"  {diagnostics['negative_distance_explanation']}")

    # Step 11T — ATR bracket first-hit audit. Reads only confirm-anchored
    # forward bars; no production indicator mutation.
    bracket_diagnostics = build_atr_bracket_first_hit_diagnostics(fs_df)
    bracket_events = bracket_diagnostics["events"]
    bracket_summary = bracket_diagnostics["summary"]
    print()
    print_atr_bracket_summary(bracket_summary)
    _print_table("step11t_by_entity", bracket_diagnostics["by_entity"])
    _print_table("step11t_by_sweep_class", bracket_diagnostics["by_sweep_class"])
    _print_table("step11t_by_sweep_family", bracket_diagnostics["by_sweep_family"])
    _print_table("step11t_by_swing_side", bracket_diagnostics["by_swing_side"])
    _print_table("step11t_by_regime", bracket_diagnostics["by_regime"])
    _print_table("step11t_by_session", bracket_diagnostics["by_session"])

    # Step 11U — bracket matrix + confluence edge audit. Research-only;
    # all confluence checks use confirmed-before-signal swings.
    step11u = build_step11u_diagnostics(fs_df)
    step11u_events = step11u["events"]
    step11u_summary = step11u["summary"]
    print()
    print_step11u_summary(step11u_summary)
    _print_table("step11u_by_confluence_type", step11u["by_confluence_type"])
    _print_table("step11u_by_entity_type", step11u["by_entity_type"])
    _print_table("step11u_by_sweep_class", step11u["by_sweep_class"])
    _print_table("step11u_by_sweep_family", step11u["by_sweep_family"])
    _print_table("step11u_by_swing_side", step11u["by_swing_side"])
    _print_table("step11u_by_regime", step11u["by_regime"])
    _print_table("step11u_by_session", step11u["by_session"])
    _print_table("step11u_by_volume_confirmed", step11u["by_volume_confirmed"])
    _print_table(
        "step11u_by_displacement_confirmed", step11u["by_displacement_confirmed"]
    )
    _print_table("step11u_by_swing_confluent", step11u["by_swing_confluent"])
    _print_table("step11u_by_confluence_regime", step11u["by_confluence_regime"])
    _print_table("step11u_by_confluence_session", step11u["by_confluence_session"])
    _print_table("step11u_by_confluence_volume", step11u["by_confluence_volume"])

    out_dir.mkdir(parents=True, exist_ok=True)
    events_cols = [c for c in FINAL_SWEEPS_COLUMNS if c in fs_df.columns]
    keep_cols = ["timestamp", "open", "high", "low", "close", "atr_14", *events_cols]
    keep_cols = [c for c in keep_cols if c in fs_df.columns]
    events_path = out_dir / f"final_sweeps_events_{instrument}_{timeframe}.csv"
    fs_df.loc[fs_df["sweep_flag"].fillna(0) > 0, keep_cols].to_csv(
        events_path, index=False
    )
    print(f"\nWrote events table → {events_path}")

    breaches_path = out_dir / f"final_sweeps_breaches_{instrument}_{timeframe}.csv"
    fs_df.loc[fs_df["sweep_breach_flag"].fillna(0) > 0, keep_cols].to_csv(
        breaches_path, index=False
    )
    print(f"Wrote breaches table → {breaches_path}")

    audit = build_unified_liquidity_clusters_audit(df)
    audit_path = (
        out_dir / f"final_sweeps_unified_sources_audit_{instrument}_{timeframe}.csv"
    )
    audit.to_csv(audit_path, index=False)
    print(f"Wrote unified-sources audit → {audit_path} ({len(audit)} rows)")

    compare_path = _write_table(
        out_dir, "final_sweeps_before_after", instrument, timeframe, comparison
    )
    family_path = _write_table(
        out_dir, "final_sweeps_family_conversion", instrument, timeframe, family_funnel
    )
    age_path = _write_table(
        out_dir, "final_sweeps_age_buckets", instrument, timeframe, age_buckets
    )
    strength_path = _write_table(
        out_dir,
        "final_sweeps_strength_buckets",
        instrument,
        timeframe,
        strength_buckets,
    )
    distance_path = _write_table(
        out_dir,
        "final_sweeps_distance_buckets",
        instrument,
        timeframe,
        distance_buckets,
    )
    selectivity_class_path = _write_table(
        out_dir,
        "final_sweeps_selectivity_class",
        instrument,
        timeframe,
        selectivity_class_table,
    )
    selectivity_family_path = _write_table(
        out_dir,
        "final_sweeps_selectivity_family",
        instrument,
        timeframe,
        selectivity_family_table,
    )
    selectivity_distance_path = _write_table(
        out_dir,
        "final_sweeps_selectivity_distance",
        instrument,
        timeframe,
        selectivity_distance_table,
    )
    selectivity_penetration_path = _write_table(
        out_dir,
        "final_sweeps_selectivity_penetration",
        instrument,
        timeframe,
        selectivity_penetration_table,
    )
    selectivity_quality_path = _write_table(
        out_dir,
        "final_sweeps_selectivity_quality",
        instrument,
        timeframe,
        selectivity_quality_table,
    )
    repeated_path = _write_table(
        out_dir,
        "final_sweeps_repeated_source_reuse",
        instrument,
        timeframe,
        repeated_sweeps,
    )
    session_funnel_path = _write_table(
        out_dir, "final_sweeps_session_funnel", instrument, timeframe, session_funnel
    )
    session_age_path = _write_table(
        out_dir,
        "final_sweeps_session_age_buckets",
        instrument,
        timeframe,
        session_age_buckets,
    )
    session_distance_path = _write_table(
        out_dir,
        "final_sweeps_session_distance_buckets",
        instrument,
        timeframe,
        session_distance_buckets,
    )
    session_pen_path = _write_table(
        out_dir,
        "final_sweeps_session_penetration_buckets",
        instrument,
        timeframe,
        session_penetration_buckets,
    )
    session_phase_path = _write_table(
        out_dir,
        "final_sweeps_session_phase",
        instrument,
        timeframe,
        session_phase_table,
    )
    session_type_path = _write_table(
        out_dir,
        "final_sweeps_session_source_type",
        instrument,
        timeframe,
        session_source_type_table,
    )
    group_rates_path = _write_table(
        out_dir,
        "final_sweeps_family_group_rates",
        instrument,
        timeframe,
        family_group_rates,
    )
    source_events_path = _write_table(
        out_dir,
        "final_sweeps_source_events_enriched",
        instrument,
        timeframe,
        source_events,
    )
    outcome_events_path = _write_table(
        out_dir,
        "final_sweeps_post_path_events",
        instrument,
        timeframe,
        source_events,
    )
    outcome_by_class_path = _write_table(
        out_dir,
        "final_sweeps_outcome_by_class",
        instrument,
        timeframe,
        outcome_by_class_table,
    )
    outcome_by_family_path = _write_table(
        out_dir,
        "final_sweeps_outcome_by_family",
        instrument,
        timeframe,
        outcome_by_family_table,
    )
    outcome_by_quality_path = _write_table(
        out_dir,
        "final_sweeps_outcome_by_quality_bucket",
        instrument,
        timeframe,
        outcome_by_quality_bucket_table,
    )
    outcome_by_side_path = _write_table(
        out_dir,
        "final_sweeps_outcome_by_side",
        instrument,
        timeframe,
        outcome_by_side_table,
    )
    outcome_by_regime_path = _write_table(
        out_dir,
        "final_sweeps_outcome_by_regime",
        instrument,
        timeframe,
        outcome_by_regime_table,
    )
    outcome_by_session_phase_path = _write_table(
        out_dir,
        "final_sweeps_outcome_by_session_phase",
        instrument,
        timeframe,
        outcome_by_session_phase_table,
    )
    outcome_by_source_age_path = _write_table(
        out_dir,
        "final_sweeps_outcome_by_source_age_bucket",
        instrument,
        timeframe,
        outcome_by_source_age_bucket_table,
    )
    outcome_by_distance_path = _write_table(
        out_dir,
        "final_sweeps_outcome_by_distance_bucket",
        instrument,
        timeframe,
        outcome_by_distance_bucket_table,
    )
    swing_events_path = _write_table(
        out_dir,
        "swing_post_confirm_events",
        instrument,
        timeframe,
        swing_events,
    )
    swing_by_side_path = _write_table(
        out_dir,
        "swing_post_confirm_by_side",
        instrument,
        timeframe,
        swing_by_side,
    )
    swing_by_latency_path = _write_table(
        out_dir,
        "swing_post_confirm_by_latency",
        instrument,
        timeframe,
        swing_by_latency,
    )
    swing_by_strength_path = _write_table(
        out_dir,
        "swing_post_confirm_by_strength",
        instrument,
        timeframe,
        swing_by_strength,
    )
    swing_by_regime_path = _write_table(
        out_dir,
        "swing_post_confirm_by_regime",
        instrument,
        timeframe,
        swing_by_regime,
    )
    swing_by_session_path = _write_table(
        out_dir,
        "swing_post_confirm_by_session",
        instrument,
        timeframe,
        swing_by_session,
    )
    swing_by_volatility_path = _write_table(
        out_dir,
        "swing_post_confirm_by_volatility",
        instrument,
        timeframe,
        swing_by_volatility,
    )
    swing_by_distance_path = _write_table(
        out_dir,
        "swing_post_confirm_by_distance",
        instrument,
        timeframe,
        swing_by_distance,
    )
    swing_comparison_path = _write_table(
        out_dir,
        "swing_post_confirm_comparison_to_sweeps",
        instrument,
        timeframe,
        swing_comparison,
    )
    swing_memo_path = (
        out_dir / f"swing_post_confirm_audit_memo_{instrument}_{timeframe}.md"
    )
    swing_memo_path.write_text(swing_memo_markdown)
    bracket_events_path = _write_table(
        out_dir,
        "atr_bracket_first_hit_events",
        instrument,
        timeframe,
        bracket_events,
    )
    bracket_by_entity_path = _write_table(
        out_dir,
        "atr_bracket_first_hit_by_entity",
        instrument,
        timeframe,
        bracket_diagnostics["by_entity"],
    )
    bracket_by_sweep_class_path = _write_table(
        out_dir,
        "atr_bracket_first_hit_by_sweep_class",
        instrument,
        timeframe,
        bracket_diagnostics["by_sweep_class"],
    )
    bracket_by_sweep_family_path = _write_table(
        out_dir,
        "atr_bracket_first_hit_by_sweep_family",
        instrument,
        timeframe,
        bracket_diagnostics["by_sweep_family"],
    )
    bracket_by_swing_side_path = _write_table(
        out_dir,
        "atr_bracket_first_hit_by_swing_side",
        instrument,
        timeframe,
        bracket_diagnostics["by_swing_side"],
    )
    bracket_by_regime_path = _write_table(
        out_dir,
        "atr_bracket_first_hit_by_regime",
        instrument,
        timeframe,
        bracket_diagnostics["by_regime"],
    )
    bracket_by_session_path = _write_table(
        out_dir,
        "atr_bracket_first_hit_by_session",
        instrument,
        timeframe,
        bracket_diagnostics["by_session"],
    )
    step11u_paths = {
        "events": _write_table(
            out_dir,
            "final_sweeps_step11u_events",
            instrument,
            timeframe,
            step11u_events,
        ),
        "by_confluence_type": _write_table(
            out_dir,
            "final_sweeps_step11u_by_confluence_type",
            instrument,
            timeframe,
            step11u["by_confluence_type"],
        ),
        "by_entity_type": _write_table(
            out_dir,
            "final_sweeps_step11u_by_entity_type",
            instrument,
            timeframe,
            step11u["by_entity_type"],
        ),
        "by_sweep_class": _write_table(
            out_dir,
            "final_sweeps_step11u_by_sweep_class",
            instrument,
            timeframe,
            step11u["by_sweep_class"],
        ),
        "by_sweep_family": _write_table(
            out_dir,
            "final_sweeps_step11u_by_sweep_family",
            instrument,
            timeframe,
            step11u["by_sweep_family"],
        ),
        "by_swing_side": _write_table(
            out_dir,
            "final_sweeps_step11u_by_swing_side",
            instrument,
            timeframe,
            step11u["by_swing_side"],
        ),
        "by_regime": _write_table(
            out_dir,
            "final_sweeps_step11u_by_regime",
            instrument,
            timeframe,
            step11u["by_regime"],
        ),
        "by_session": _write_table(
            out_dir,
            "final_sweeps_step11u_by_session",
            instrument,
            timeframe,
            step11u["by_session"],
        ),
        "by_volume_confirmed": _write_table(
            out_dir,
            "final_sweeps_step11u_by_volume_confirmed",
            instrument,
            timeframe,
            step11u["by_volume_confirmed"],
        ),
        "by_displacement_confirmed": _write_table(
            out_dir,
            "final_sweeps_step11u_by_displacement_confirmed",
            instrument,
            timeframe,
            step11u["by_displacement_confirmed"],
        ),
        "by_swing_confluent": _write_table(
            out_dir,
            "final_sweeps_step11u_by_swing_confluent",
            instrument,
            timeframe,
            step11u["by_swing_confluent"],
        ),
        "by_confluence_regime": _write_table(
            out_dir,
            "final_sweeps_step11u_by_confluence_regime",
            instrument,
            timeframe,
            step11u["by_confluence_regime"],
        ),
        "by_confluence_session": _write_table(
            out_dir,
            "final_sweeps_step11u_by_confluence_session",
            instrument,
            timeframe,
            step11u["by_confluence_session"],
        ),
        "by_confluence_volume": _write_table(
            out_dir,
            "final_sweeps_step11u_by_confluence_volume",
            instrument,
            timeframe,
            step11u["by_confluence_volume"],
        ),
    }
    chart_paths = _render_final_sweeps_charts(
        df=fs_df,
        out_dir=out_dir,
        instrument=instrument,
        timeframe=timeframe,
    )
    print(f"Wrote before/after table → {compare_path}")
    print(f"Wrote family conversion table → {family_path}")
    print(f"Wrote age bucket table → {age_path}")
    print(f"Wrote strength bucket table → {strength_path}")
    print(f"Wrote distance bucket table → {distance_path}")
    print(f"Wrote selectivity-class table → {selectivity_class_path}")
    print(f"Wrote selectivity-family table → {selectivity_family_path}")
    print(f"Wrote selectivity-distance table → {selectivity_distance_path}")
    print(f"Wrote selectivity-penetration table → {selectivity_penetration_path}")
    print(f"Wrote selectivity-quality table → {selectivity_quality_path}")
    print(f"Wrote repeated-source table → {repeated_path}")
    print(f"Wrote session funnel table → {session_funnel_path}")
    print(f"Wrote session age bucket table → {session_age_path}")
    print(f"Wrote session distance bucket table → {session_distance_path}")
    print(f"Wrote session penetration bucket table → {session_pen_path}")
    print(f"Wrote session phase table → {session_phase_path}")
    print(f"Wrote session source-type table → {session_type_path}")
    print(f"Wrote family-group rate table → {group_rates_path}")
    print(f"Wrote enriched source-events table → {source_events_path}")
    print(f"Wrote post-path events table → {outcome_events_path}")
    print(f"Wrote outcome-by-class table → {outcome_by_class_path}")
    print(f"Wrote outcome-by-family table → {outcome_by_family_path}")
    print(f"Wrote outcome-by-side table → {outcome_by_side_path}")
    print(f"Wrote outcome-by-regime table → {outcome_by_regime_path}")
    print(f"Wrote outcome-by-session-phase table → {outcome_by_session_phase_path}")
    print(f"Wrote outcome-by-quality-bucket table → {outcome_by_quality_path}")
    print(f"Wrote outcome-by-source-age-bucket table → {outcome_by_source_age_path}")
    print(f"Wrote outcome-by-distance-bucket table → {outcome_by_distance_path}")
    print(f"Wrote swing post-confirm events table → {swing_events_path}")
    print(f"Wrote swing post-confirm by-side table → {swing_by_side_path}")
    print(f"Wrote swing post-confirm by-latency table → {swing_by_latency_path}")
    print(f"Wrote swing post-confirm by-strength table → {swing_by_strength_path}")
    print(f"Wrote swing post-confirm by-regime table → {swing_by_regime_path}")
    print(f"Wrote swing post-confirm by-session table → {swing_by_session_path}")
    print(f"Wrote swing post-confirm by-volatility table → {swing_by_volatility_path}")
    print(f"Wrote swing post-confirm by-distance table → {swing_by_distance_path}")
    print(f"Wrote swing post-confirm comparison table → {swing_comparison_path}")
    print(f"Wrote swing audit memo → {swing_memo_path}")
    print(f"Wrote ATR bracket first-hit events table → {bracket_events_path}")
    print(f"Wrote ATR bracket first-hit by-entity table → {bracket_by_entity_path}")
    print(
        f"Wrote ATR bracket first-hit by-sweep-class table → "
        f"{bracket_by_sweep_class_path}"
    )
    print(
        f"Wrote ATR bracket first-hit by-sweep-family table → "
        f"{bracket_by_sweep_family_path}"
    )
    print(
        f"Wrote ATR bracket first-hit by-swing-side table → "
        f"{bracket_by_swing_side_path}"
    )
    print(f"Wrote ATR bracket first-hit by-regime table → {bracket_by_regime_path}")
    print(f"Wrote ATR bracket first-hit by-session table → {bracket_by_session_path}")
    for label, path in step11u_paths.items():
        print(f"Wrote step 11U {label} table → {path}")
    for chart_path in chart_paths:
        print(f"Wrote chart → {chart_path}")

    failures = 0
    failures += int(fs_summary.get("causality_violations", 0))
    failures += int(us_summary.get("causality_violations", 0))
    if fs_summary.get("deprecated_families_in_attribution"):
        failures += 1
    if us_summary.get("deprecated_families_present"):
        failures += 1
    if fs_summary.get("forward_outcomes_in_production_schema"):
        failures += 1
    if fs_summary.get("forward_outcome_prefix_violations"):
        failures += 1
    return failures


def main() -> int:
    args = parse_args()
    runs = _resolve_runs(args)
    out_dir = Path(args.out_dir)
    total_failures = 0
    for instrument, timeframe in runs:
        total_failures += _run_single(
            instrument,
            timeframe,
            out_dir=out_dir,
            max_rows=args.max_rows,
            frame_cache=args.frame_cache,
        )
    if total_failures > 0:
        print(f"\nVALIDATION FAILED: {total_failures} acceptance-gate violations")
        return 1
    print("\nVALIDATION OK: all acceptance gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
