from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.dag_runtime import GraphRunContext, execute_graph
from src.dag_runtime.builtin_graphs import get_builtin_graph
from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.foundation.adx import add_adx
from src.indicators.foundation.ema import add_emas
from src.indicators.foundation.regime import add_regime
from src.indicators.foundation.session import add_session_features
from src.indicators.foundation.volatility import add_atr, add_bb_width
from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state
from src.pipeline_runtime import (
    dataframe_fingerprint,
    load_partitioned_dataset,
)
from src.validation.common import (
    cleanup_validation_artifacts,
)

OUT_DIR = Path("notebooks/structure")
CACHE_ROOT = Path("data/validation_cache")
FEATURES_ROOT = Path("data/features")
VALIDATOR_NAME = "validate_trend_state"
PLOT_ROWS = 300
RUNS = (
    ("XAU_USD", "H1"),
    ("XAU_USD", "H4"),
)
SWING_WINDOW = 6


def _print_summary(value: object, *, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                print(f"{prefix}{key}:")
                _print_summary(child, indent=indent + 2)
            elif hasattr(child, "to_string"):
                print(f"{prefix}{key}:")
                print(
                    child.to_string(index=False)
                    if isinstance(child, pd.DataFrame)
                    else child.to_string()
                )
            else:
                print(f"{prefix}{key}: {child}")
        return
    print(f"{prefix}{value}")


def _print_compact_summary(
    summary: dict[str, object],
    *,
    indent: int,
    include_keys: tuple[str, ...],
) -> None:
    for key in include_keys:
        if key in summary:
            print(f"{' ' * indent}{key}:")
            _print_summary(summary[key], indent=indent + 2)


def _build_context(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_candle_schema(df, require_volume=False)
    out = out.sort_values("timestamp").reset_index(drop=True)
    out = add_atr(out)
    out = add_emas(out)
    out = add_adx(out)
    out = add_bb_width(out)
    out = add_swings(out, window=SWING_WINDOW)
    out = add_trend_state(out)

    ts = pd.to_datetime(out["timestamp"], utc=True)
    if ts.diff().median().total_seconds() < 86400:
        out = add_session_features(out, include_research_only=False)

    out = add_regime(out, include_research_only=False)
    return out


def _build_trend_state_context(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_candle_schema(df, require_volume=False)
    out = out.sort_values("timestamp").reset_index(drop=True)
    out = add_atr(out)
    out = add_emas(out)
    out = add_adx(out)
    out = add_bb_width(out)
    out = add_swings(out, window=SWING_WINDOW)
    out = add_trend_state(out)
    return out


def _load_canonical_live_context(
    raw: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
) -> pd.DataFrame | None:
    canonical = load_partitioned_dataset(
        FEATURES_ROOT,
        dataset="live",
        symbol=instrument,
        timeframe=timeframe,
    )
    if canonical.empty:
        return None
    raw_view = raw.loc[:, ["timestamp", "open", "high", "low", "close"]].reset_index(
        drop=True
    )
    canonical_view = canonical.loc[
        :, ["timestamp", "open", "high", "low", "close"]
    ].reset_index(drop=True)
    if len(raw_view) != len(canonical_view):
        return None
    if dataframe_fingerprint(raw_view, strategy="content") != dataframe_fingerprint(
        canonical_view, strategy="content"
    ):
        return None
    return canonical.reset_index(drop=True)


def _print_validation_sections(summary: dict[str, object]) -> None:
    ordered_keys = [
        "current_trend_snapshot",
        "strict_state_counts",
        "strict_state_pct",
        "bias_state_counts",
        "bias_state_pct",
        "transition_count",
        "transition_matrix",
        "bias_transition_matrix",
        "dwell_diagnostics",
        "confidence_by_state",
        "confidence_ordering_check",
        "confidence_separation_check",
        "neutral_confidence_cap_check",
        "strength_by_state",
        "commitment_by_state",
        "bias_interaction",
        "regime_interaction",
        "neutral_overuse_audit",
        "neutral_confidence_audit",
        "neutral_in_trend_audit",
        "directional_in_range_audit",
        "old_neutral_strong_env_audit",
        "stale_neutral_neutral_env_split",
        "stale_neutral_candidate_forward_audit",
        "stale_neutral_gap_age_grid",
        "candidate_vs_range_decay_comparison",
        "stale_neutral_commit_structure_audit",
        "stale_neutral_commit_component_audit",
        "stale_neutral_commit_mass_vs_resolution_audit",
        "stale_neutral_weak_side_survival_audit",
        "stale_neutral_strength_vs_resolution_audit",
        "stale_neutral_conflict_signature_audit",
        "clean_asymmetry_candidate_audit",
        "clean_asymmetry_age_audit",
        "clean_asymmetry_env_audit",
        "clean_asymmetry_transition_risk_proxy",
        "clean_asymmetry_vs_bias_rows_audit",
        "promotion_input_live_safety_audit",
        "confirmed_input_promotion_prototype_sweep",
        "stale_neutral_event_recency_audit",
        "stale_neutral_contradiction_audit",
        "stale_neutral_dual_commit_grid",
        "strict_neutral_anomaly_bucket",
        "stale_neutral_promotion_candidate_audit",
        "mature_directional_in_range_decay_audit",
        "neutral_with_directional_bias_audit",
        "commit_gap_audit",
        "neutral_age_audit",
        "semantic_buckets",
    ]
    print(f"row_count: {summary['row_count']}")
    for key in ordered_keys:
        if key in summary:
            print(f"{key}:")
            if key == "confirmed_input_promotion_prototype_sweep":
                _print_compact_summary(
                    summary[key],
                    indent=2,
                    include_keys=(
                        "config_count",
                        "configs_with_rows",
                        "selected_config_id",
                        "selected_config",
                        "selected_config_evaluation",
                    ),
                )
                continue
            _print_summary(summary[key], indent=2)


def _state_distribution(series: pd.Series) -> dict[str, int]:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    return {
        "bearish": int(values.eq(-1).sum()),
        "neutral": int(values.eq(0).sum()),
        "bullish": int(values.eq(1).sum()),
    }


def _print_minimal_overlay_check(
    df: pd.DataFrame, *, instrument: str, timeframe: str
) -> None:
    """Print the frozen stale-neutral overlay contract check only.

    Accepted checks:
    - promoted_row_count > 0
    - effective_state_diff_count == promoted_row_count
    - nonneutral_diff_count == 0
    - invalid_promo_on_nondirectional_count == 0
    - canonical_trend_state_unchanged == True
    """
    promo = (
        pd.to_numeric(df["stale_neutral_promo_side"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    trend = pd.to_numeric(df["trend_state"], errors="coerce").fillna(0).astype(int)
    effective = (
        pd.to_numeric(df["effective_trend_state"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    print(f"\n=== STALE NEUTRAL LIVE OVERLAY CHECK: {instrument} {timeframe} ===")
    print(f"promoted_row_count: {int(promo.ne(0).sum())}")
    print(f"promoted_bull_count: {int(promo.eq(1).sum())}")
    print(f"promoted_bear_count: {int(promo.eq(-1).sum())}")
    print(f"effective_state_diff_count: {int(effective.ne(trend).sum())}")
    print(f"nonneutral_diff_count: {int((trend.ne(0) & effective.ne(trend)).sum())}")
    print(
        f"invalid_promo_on_nondirectional_count: {int((trend.ne(0) & promo.ne(0)).sum())}"
    )
    print("effective_trend_state_distribution:")
    for key, value in _state_distribution(effective).items():
        print(f"  {key}: {value}")
    print(
        "canonical_trend_state_unchanged: "
        f"{bool(df.attrs.get('canonical_trend_state_unchanged', False))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Print only the stale-neutral live overlay check.",
    )
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--tail-rows", type=int, default=PLOT_ROWS)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--invalidate-cache", action="store_true")
    parser.add_argument("--cleanup-stale", action="store_true")
    parser.add_argument("--max-artifact-age-days", type=int, default=30)
    args = parser.parse_args()
    if args.cleanup_stale:
        removed = cleanup_validation_artifacts(
            cache_root=CACHE_ROOT,
            max_age_days=args.max_artifact_age_days,
            report_roots=[OUT_DIR],
        )
        print(f"cleanup_removed: {len(removed)}")

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        raw = pd.read_parquet(data_file)
        if args.minimal:
            graph = get_builtin_graph(
                "validate_trend_state",
                instrument=instrument,
                timeframe=timeframe,
            )
            context = GraphRunContext(
                graph_name=graph.graph_name,
                symbol=instrument,
                timeframe=timeframe,
                inputs={"raw_input": raw},
                config={"html": False, "out_dir": str(OUT_DIR)},
                cache_root=CACHE_ROOT,
                features_root=FEATURES_ROOT,
                force=args.force,
                invalidate_cache=args.invalidate_cache,
            )
            graph_result = execute_graph(
                graph,
                context=context,
                target="trend_state_minimal_overlay_context",
            )
            trend_df = graph_result.primary_frame()
            _print_minimal_overlay_check(
                trend_df, instrument=instrument, timeframe=timeframe
            )
            profile_path = (
                CACHE_ROOT
                / VALIDATOR_NAME
                / instrument
                / timeframe
                / "run-summary.json"
            )
            graph_result.profiler.write_json(profile_path)
            print(f"Profiler summary: {profile_path}")
            continue

        graph = get_builtin_graph(
            "validate_trend_state",
            instrument=instrument,
            timeframe=timeframe,
        )
        context = GraphRunContext(
            graph_name=graph.graph_name,
            symbol=instrument,
            timeframe=timeframe,
            inputs={"raw_input": raw},
            config={
                "plot_rows": args.tail_rows,
                "full": args.full,
                "html": args.html,
                "out_dir": str(OUT_DIR),
            },
            cache_root=CACHE_ROOT,
            features_root=FEATURES_ROOT,
            force=args.force,
            invalidate_cache=args.invalidate_cache,
        )
        graph_result = execute_graph(
            graph, context=context, target="trend_state_validation_bundle"
        )
        result = graph_result.output().payload

        print(f"\n=== TREND STATE SUMMARY: {instrument} {timeframe} ===")
        _print_validation_sections(result["summary"])
        if args.html and result["html_path"] is not None:
            print(f"Wrote chart to: {result['html_path']}")
        else:
            print("HTML output skipped. Pass --html to generate charts.")
        profile_path = (
            CACHE_ROOT / VALIDATOR_NAME / instrument / timeframe / "run-summary.json"
        )
        graph_result.profiler.write_json(profile_path)
        print(f"Profiler summary: {profile_path}")


if __name__ == "__main__":
    main()
