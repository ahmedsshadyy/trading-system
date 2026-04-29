from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.dag_runtime import GraphRunContext, execute_graph
from src.dag_runtime.builtin_graphs import get_builtin_graph
from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.foundation.volatility import add_atr
from src.indicators.smc.displacement import add_displacement_candle
from src.indicators.smc.ob import add_ob
from src.indicators.smc.ob_mitigation import add_ob_mitigation
from src.indicators.structure.bos import add_bos
from src.indicators.structure.choch import add_choch
from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/smc")
CACHE_ROOT = Path("data/dag_cache")
VALIDATOR_NAME = "validate_ob"

SWING_WINDOW = 4
SWING_RETRACE = 0.7
SWING_CONFIRM_BARS = 2
PLOT_START = pd.Timestamp("2026-01-01", tz="UTC")


def _print_summary(value: object, *, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                print(f"{prefix}{key}:")
                _print_summary(child, indent=indent + 2)
            else:
                print(f"{prefix}{key}: {child}")
        return
    print(f"{prefix}{value}")


def _freeze_recommendation(
    summary: dict[str, object],
    inventory: dict[str, object],
    execution: dict[str, object],
) -> str:
    sanity = summary.get("sanity_checks", {}) if isinstance(summary, dict) else {}
    if not bool(sanity.get("activation_equals_parent_confirmation", False)):
        return "DO NOT FREEZE"
    if not bool(sanity.get("geometry_full_range_consistency", False)):
        return "DO NOT FREEZE"
    if not bool(sanity.get("one_raw_ob_per_confirmed_bos_or_pathological", False)):
        return "DO NOT FREEZE"
    if pd.isna(execution.get("bos_expectancy")) or pd.isna(
        execution.get("ob_first_touch_expectancy")
    ):
        return "DO NOT FREEZE"
    return "FREEZE"


def _freeze_blockers(
    summary: dict[str, object],
    inventory: dict[str, object],
    execution: dict[str, object],
) -> list[str]:
    blockers: list[str] = []
    sanity = summary.get("sanity_checks", {}) if isinstance(summary, dict) else {}
    mitigation = (
        summary.get("mitigation_checks", {}) if isinstance(summary, dict) else {}
    )
    if not bool(sanity.get("activation_equals_parent_confirmation", False)):
        blockers.append(
            "activation does not equal parent BOS confirmation for every canonical OB"
        )
    if not bool(sanity.get("geometry_full_range_consistency", False)):
        blockers.append(
            "canonical geometry does not match the full source candle range"
        )
    if not bool(sanity.get("one_raw_ob_per_confirmed_bos_or_pathological", False)):
        blockers.append(
            "raw canonical OB coverage is not structurally aligned with confirmed BOS count"
        )
    if not bool(mitigation.get("no_touch_before_activation", False)):
        blockers.append("mitigation lifecycle allows touch before activation")
    if pd.isna(execution.get("bos_expectancy")) or pd.isna(
        execution.get("ob_first_touch_expectancy")
    ):
        blockers.append(
            "BOS vs OB execution harness did not produce stable expectancy metrics"
        )
    if (
        pd.notna(inventory.get("fraction_with_top_inventory_within_1atr"))
        and float(inventory["fraction_with_top_inventory_within_1atr"]) < 0.10
    ):
        blockers.append("top inventory remains too far from price too often")
    return blockers


def _build_context(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_candle_schema(raw, require_volume=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = add_atr(df)
    df = add_swings(
        df,
        window=SWING_WINDOW,
        min_retrace_atr=SWING_RETRACE,
        min_confirm_bars=SWING_CONFIRM_BARS,
    )
    df = add_trend_state(df)
    df = add_bos(
        df,
        min_source_age_bars=1,
        min_break_distance_atr=0.05,
        min_body_atr=0.10,
    )
    df = add_choch(
        df,
        min_source_age_bars=1,
        min_break_distance_atr=0.05,
        min_body_atr=0.10,
    )
    df = add_displacement_candle(df)
    df = add_ob(df)
    df = add_ob_mitigation(df)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", action="store_true", default=True)
    parser.add_argument("--no-html", dest="html", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--invalidate-cache", action="store_true")
    parser.add_argument("--plot-start", default=str(PLOT_START))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = pd.read_parquet(DATA_FILE)
    graph = get_builtin_graph(
        "validate_ob",
        instrument="XAU_USD",
        timeframe="H4",
    )
    context = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={
            "html": args.html,
            "out_dir": str(OUT_DIR),
            "plot_start": args.plot_start,
        },
        cache_root=CACHE_ROOT,
        force=args.force,
        invalidate_cache=args.invalidate_cache,
    )
    graph_result = execute_graph(graph, context=context, target="ob_validation_bundle")
    output = graph_result.output()
    result = output.payload
    frames = output.frames

    print("\n=== OB SUMMARY ===")
    _print_summary(result["summary"])

    print("\n=== OB BOS COVERAGE ===")
    _print_summary(result["coverage_summary"])

    print("\n=== OB INVENTORY USABILITY ===")
    _print_summary(result["inventory_summary"])

    print("\n=== OB LIVE VS NONLIVE EQUIVALENCE ===")
    _print_summary(result["equivalence_summary"])
    casebook = frames["live_vs_nonlive_casebook"]
    if casebook.empty:
        print("No equivalence case rows.")
    else:
        print(
            casebook[
                [
                    "reference_case_id",
                    "match_class",
                    "source_lag_bars",
                    "activation_lag_bars",
                    "geometry_drift_atr",
                    "failure_primary_reason",
                ]
            ].to_string(index=False)
        )

    print("\n=== BOS VS OB EXECUTION ===")
    _print_summary(result["execution_summary"])
    print(
        frames["execution_comparison"][
            [
                "scope_type",
                "scope_value",
                "policy",
                "sample_count",
                "fill_rate",
                "expectancy_r",
                "win_rate_1r",
                "win_rate_2r",
                "win_rate_3r",
                "entry_improvement_atr",
                "stop_compression_improvement_atr",
            ]
        ].to_string(index=False)
    )

    print("\n=== OB REDUNDANCY ===")
    _print_summary(result["redundancy_summary"])

    summary = result["summary"]
    coverage = result["coverage_summary"]
    inventory = result["inventory_summary"]
    execution = result["execution_summary"]
    doctrine_status = (
        "PASS"
        if summary["sanity_checks"]["activation_equals_parent_confirmation"]
        and summary["sanity_checks"]["geometry_full_range_consistency"]
        and summary["sanity_checks"]["source_before_or_at_activation"]
        else "FAIL"
    )
    live_safety = (
        "PASS"
        if summary["sanity_checks"]["source_before_or_at_activation"]
        and summary.get("mitigation_checks", {}).get(
            "no_touch_before_activation", False
        )
        else "FAIL"
    )
    freeze = _freeze_recommendation(summary, inventory, execution)
    blockers = _freeze_blockers(summary, inventory, execution)

    print("\n=== FINAL DECISION TEMPLATE ===")
    print("1) Canonical doctrine status")
    print(f"- {doctrine_status}")
    print("\n2) BOS coverage")
    print(f"- confirmed_bos_count = {coverage['confirmed_bos_count']}")
    print(f"- raw_canonical_ob_count = {coverage['raw_canonical_bos_ob_count']}")
    print(f"- coverage_fraction = {coverage['coverage_fraction']}")
    print("\n3) Live-safety")
    print(f"- {live_safety}")
    print("\n4) Inventory usability")
    print(
        f"- fraction_with_top_inventory_within_1atr = {inventory['fraction_with_top_inventory_within_1atr']}"
    )
    print(
        f"- fraction_with_top_inventory_within_2atr = {inventory['fraction_with_top_inventory_within_2atr']}"
    )
    print(
        f"- median_distance_to_top_inventory_atr = {inventory['median_distance_to_top_inventory_atr']}"
    )
    print("\n5) BOS vs OB execution value")
    print(f"- BOS expectancy = {execution['bos_expectancy']}")
    print(f"- OB first-touch expectancy = {execution['ob_first_touch_expectancy']}")
    print(
        f"- OB mean-threshold expectancy = {execution['ob_mean_threshold_expectancy']}"
    )
    print(
        f"- stop compression improvement = {execution['stop_compression_improvement']}"
    )
    print(f"- entry improvement = {execution['entry_improvement']}")
    print(f"- verdict = \"{execution['verdict']}\"")
    print("\n6) Freeze recommendation")
    print(f"- {freeze}")
    print("\n7) If DO NOT FREEZE")
    if blockers:
        for blocker in blockers:
            print(f"- {blocker}")
    else:
        print("- none")

    print("\n=== SAMPLE BULL OB WINDOWS ===")
    for i in range(int(result["bull_window_count"])):
        window = frames[f"bull_window_{i}"]
        print(f"\n--- bull window {i + 1} ---")
        print(window.to_string(index=True))

    print("\n=== SAMPLE BEAR OB WINDOWS ===")
    for i in range(int(result["bear_window_count"])):
        window = frames[f"bear_window_{i}"]
        print(f"\n--- bear window {i + 1} ---")
        print(window.to_string(index=True))

    if args.html and result["html_path"] is not None:
        print(f"\nWrote chart to: {result['html_path']}")
    else:
        print("\nHTML output skipped. Pass --html to generate charts.")
    print("\nWrote diagnostic artifacts:")
    for name, path in sorted(result["artifact_paths"].items()):
        print(f"  {name}: {path}")
    profile_path = CACHE_ROOT / VALIDATOR_NAME / "XAU_USD" / "H4" / "run-summary.json"
    graph_result.profiler.write_json(profile_path)
    print(f"Profiler summary: {profile_path}")


if __name__ == "__main__":
    main()
