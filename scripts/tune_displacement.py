from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.research.displacement_research import (
    build_displacement_research_table,
)
from src.validation.indicators.displacement import (
    DISPLACEMENT_DEFAULT_THRESHOLDS,
    DISPLACEMENT_TARGET_EVENT_COUNT_MAX,
    DISPLACEMENT_TARGET_EVENT_COUNT_MIDPOINT,
    DISPLACEMENT_TARGET_EVENT_COUNT_MIN,
    DISPLACEMENT_TUNING_BASELINE_THRESHOLDS,
    DISPLACEMENT_TUNING_GRID,
    build_displacement_analysis_base_frame,
    build_displacement_analysis_frame,
    displacement_candidate_in_target_range,
    displacement_candidate_passes_acceptance,
    summarize_displacement,
)

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/smc")
SWEEP_CSV = OUT_DIR / "displacement_tuning_sweep_XAU_USD_H4.csv"
SUMMARY_JSON = OUT_DIR / "displacement_tuning_summary_XAU_USD_H4.json"

TUNING_ORDER = (
    "body_atr_mult",
    "min_body_frac",
    "close_extreme_frac",
    "max_opposite_wick_frac",
)


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


def _threshold_key(thresholds: dict[str, float]) -> tuple[float, ...]:
    return (
        float(thresholds["body_atr_mult"]),
        float(thresholds["min_body_frac"]),
        float(thresholds["close_extreme_frac"]),
        float(thresholds["max_opposite_wick_frac"]),
    )


def _looseness(thresholds: dict[str, float]) -> dict[str, float]:
    return {
        "body_atr_mult_delta": float(
            DISPLACEMENT_TUNING_BASELINE_THRESHOLDS["body_atr_mult"]
            - thresholds["body_atr_mult"]
        ),
        "min_body_frac_delta": float(
            DISPLACEMENT_TUNING_BASELINE_THRESHOLDS["min_body_frac"]
            - thresholds["min_body_frac"]
        ),
        "close_extreme_frac_delta": float(
            thresholds["close_extreme_frac"]
            - DISPLACEMENT_TUNING_BASELINE_THRESHOLDS["close_extreme_frac"]
        ),
        "max_opposite_wick_frac_delta": float(
            thresholds["max_opposite_wick_frac"]
            - DISPLACEMENT_TUNING_BASELINE_THRESHOLDS["max_opposite_wick_frac"]
        ),
    }


def _evaluate_candidate(
    base_df: pd.DataFrame,
    *,
    thresholds: dict[str, float],
    baseline_candidate: dict[str, object],
    stage: str,
) -> dict[str, object]:
    df = build_displacement_analysis_frame(base_df, **thresholds)
    research = build_displacement_research_table(
        df,
        atr_length=int(thresholds["atr_length"]),
    )
    summary = summarize_displacement(df, research_table=research, **thresholds)
    candidate = dict(summary["candidate_comparison"])
    ratio_stats = candidate.pop("excursion_ratio_5_capped", {})
    if isinstance(ratio_stats, dict):
        for key, value in ratio_stats.items():
            candidate[f"excursion_ratio_5_capped_{key}"] = value
    candidate.update(
        {
            "stage": stage,
            "body_atr_mult": thresholds["body_atr_mult"],
            "min_body_frac": thresholds["min_body_frac"],
            "close_extreme_frac": thresholds["close_extreme_frac"],
            "max_opposite_wick_frac": thresholds["max_opposite_wick_frac"],
        }
    )
    candidate.update(_looseness(thresholds))
    candidate["total_looseness"] = float(
        candidate["body_atr_mult_delta"]
        + candidate["min_body_frac_delta"]
        + candidate["close_extreme_frac_delta"]
        + candidate["max_opposite_wick_frac_delta"]
    )
    candidate["event_distance_to_target"] = abs(
        float(candidate["event_count"]) - DISPLACEMENT_TARGET_EVENT_COUNT_MIDPOINT
    )
    candidate["in_target_range"] = displacement_candidate_in_target_range(candidate)
    candidate["accepted"] = displacement_candidate_passes_acceptance(
        candidate,
        baseline_candidate,
    )
    return candidate


def run_tuning(base_df: pd.DataFrame) -> dict[str, object]:
    baseline_thresholds = dict(DISPLACEMENT_TUNING_BASELINE_THRESHOLDS)
    baseline_candidate = _evaluate_candidate(
        base_df,
        thresholds=baseline_thresholds,
        baseline_candidate={},
        stage="baseline",
    )
    baseline_candidate["accepted"] = False

    results: list[dict[str, object]] = [baseline_candidate]
    seen = {_threshold_key(baseline_thresholds)}
    current = dict(baseline_thresholds)
    selected_candidate: dict[str, object] | None = None
    stopped_stage: str | None = None

    for param in TUNING_ORDER:
        stage_results: list[dict[str, object]] = []
        for value in DISPLACEMENT_TUNING_GRID[param]:
            thresholds = dict(current)
            thresholds[param] = float(value)
            key = _threshold_key(thresholds)
            if key in seen:
                continue
            seen.add(key)
            candidate = _evaluate_candidate(
                base_df,
                thresholds=thresholds,
                baseline_candidate=baseline_candidate,
                stage=param,
            )
            results.append(candidate)
            stage_results.append(candidate)

        stage_in_target = [row for row in stage_results if row["in_target_range"]]
        if stage_in_target:
            stopped_stage = param
            accepted = [row for row in stage_in_target if row["accepted"]]
            if accepted:
                selected_candidate = min(
                    accepted,
                    key=lambda row: (
                        row["total_looseness"],
                        row["event_distance_to_target"],
                    ),
                )
            break

        current[param] = float(DISPLACEMENT_TUNING_GRID[param][-1])

    results_df = pd.DataFrame(results).sort_values(
        ["accepted", "event_distance_to_target", "total_looseness"],
        ascending=[False, True, True],
    )
    return {
        "baseline_candidate": baseline_candidate,
        "selected_candidate": selected_candidate,
        "stopped_stage": stopped_stage,
        "results_df": results_df.reset_index(drop=True),
    }


def main() -> None:
    raw_df = pd.read_parquet(DATA_FILE)
    base_df = build_displacement_analysis_base_frame(raw_df, instrument="XAU_USD")
    tuning = run_tuning(base_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tuning["results_df"].to_csv(SWEEP_CSV, index=False)

    summary = {
        "dataset": str(DATA_FILE),
        "baseline_thresholds": DISPLACEMENT_TUNING_BASELINE_THRESHOLDS,
        "current_default_thresholds": DISPLACEMENT_DEFAULT_THRESHOLDS,
        "target_event_count_range": {
            "min": DISPLACEMENT_TARGET_EVENT_COUNT_MIN,
            "max": DISPLACEMENT_TARGET_EVENT_COUNT_MAX,
        },
        "baseline_candidate": tuning["baseline_candidate"],
        "selected_candidate": tuning["selected_candidate"],
        "stopped_stage": tuning["stopped_stage"],
        "defaults_should_change": tuning["selected_candidate"] is not None,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== DISPLACEMENT TUNING SUMMARY ===")
    _print_summary(summary)
    print(f"\nWrote sweep CSV to: {SWEEP_CSV}")
    print(f"Wrote summary JSON to: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
