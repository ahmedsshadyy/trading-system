from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.research.equal_hl_research import build_equal_hl_research_table
from src.indicators.smc.equal_hl import (
    EQHL_ACTIVE_COLUMNS,
    EQHL_DETECT_COLUMNS,
    EQHL_DIAGNOSTIC_COLUMNS,
    EQHL_LEGACY_COLUMNS,
    EQHL_LIFECYCLE_COLUMNS,
    EQHL_RANKED_ACTIVE_COLUMNS,
    add_equal_hl,
)
from src.validation.indicators.equal_hl import (
    EQHL_TUNING_BASELINE_PARAMS,
    EQHL_TUNING_GRID,
    apply_eqhl_score_candidate,
    build_equal_hl_active_snapshot_table,
    build_equal_hl_candidate_summary,
    build_equal_hl_calibration_event_table,
    equal_hl_candidate_passes_acceptance,
    _prepare_active_snapshots,
    _prepare_calibration_events,
    search_equal_hl_tradeable,
    summarize_equal_hl,
    summarize_equal_hl_score_candidate,
)

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/smc")
SWEEP_CSV = OUT_DIR / "equal_hl_tuning_sweep_XAU_USD_H4.csv"
SUMMARY_JSON = OUT_DIR / "equal_hl_tuning_summary_XAU_USD_H4.json"
COMPONENT_AUDIT_CSV = OUT_DIR / "equal_hl_score_component_audit_XAU_USD_H4.csv"
COARSE_CANDIDATES_CSV = OUT_DIR / "equal_hl_score_coarse_XAU_USD_H4.csv"
REFINED_CANDIDATES_CSV = OUT_DIR / "equal_hl_score_refined_XAU_USD_H4.csv"


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


def build_equal_hl_analysis_base_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    full_df = build_research_indicators(
        raw_df,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
    )
    drop_columns = (
        EQHL_DETECT_COLUMNS
        + EQHL_ACTIVE_COLUMNS
        + EQHL_RANKED_ACTIVE_COLUMNS
        + EQHL_LIFECYCLE_COLUMNS
        + EQHL_DIAGNOSTIC_COLUMNS
        + EQHL_LEGACY_COLUMNS
    )
    existing = [col for col in drop_columns if col in full_df.columns]
    return full_df.drop(columns=existing)


def _baseline_score_calibration(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    research = build_equal_hl_research_table(df)
    calibration_events = build_equal_hl_calibration_event_table(
        df, research_table=research
    )
    calibration_events, _ = _prepare_calibration_events(calibration_events)
    structural_benchmark = summarize_equal_hl_score_candidate(
        calibration_events,
        score_column="structural_score",
    )
    return calibration_events, search_equal_hl_tradeable(
        calibration_events,
        structural_benchmark=structural_benchmark,
    )


def _evaluate_threshold_candidate(
    base_df: pd.DataFrame,
    *,
    atr_tolerance: float,
    lookback_swings: int,
    accepted_v2_candidate: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, object]:
    df = add_equal_hl(
        base_df,
        atr_tolerance=float(atr_tolerance),
        lookback_swings=int(lookback_swings),
    )
    research = build_equal_hl_research_table(df)
    calibration_events = build_equal_hl_calibration_event_table(
        df, research_table=research
    )
    calibration_events, width_reference = _prepare_calibration_events(
        calibration_events
    )
    active_snapshots = _prepare_active_snapshots(
        build_equal_hl_active_snapshot_table(df),
        width_reference=width_reference,
    )
    scored_events = apply_eqhl_score_candidate(
        calibration_events,
        candidate_row=accepted_v2_candidate,
        width_reference=width_reference,
    )
    candidate = build_equal_hl_candidate_summary(
        scored_events,
        active_snapshots,
        score_column="tradeable_live_score",
        width_reference=width_reference,
    )
    candidate.update(
        {
            "atr_tolerance": float(atr_tolerance),
            "lookback_swings": int(lookback_swings),
            "score_repair_accepted": True,
        }
    )
    baseline_event_count = float(baseline.get("event_count", 0) or 0)
    candidate["inventory_growth_ratio"] = (
        float(candidate["event_count"]) / baseline_event_count
        if baseline_event_count > 0
        else float("nan")
    )
    candidate["accepted"] = equal_hl_candidate_passes_acceptance(candidate, baseline)
    return candidate


def run_tuning(base_df: pd.DataFrame) -> dict[str, object]:
    baseline_df = add_equal_hl(base_df, **EQHL_TUNING_BASELINE_PARAMS)
    baseline_research = build_equal_hl_research_table(baseline_df)
    baseline_summary = summarize_equal_hl(
        baseline_df,
        research_table=baseline_research,
    )
    calibration_events, score_calibration = _baseline_score_calibration(baseline_df)

    baseline_candidate = dict(baseline_summary["candidate_comparison"])
    baseline_candidate.update(EQHL_TUNING_BASELINE_PARAMS)
    baseline_candidate["inventory_growth_ratio"] = 1.0
    baseline_candidate["accepted"] = False

    threshold_results = pd.DataFrame()
    selected_threshold_candidate = None
    score_repair_accepted = bool(
        baseline_summary["score_calibration"].get("selected_candidate_accepted", False)
    )

    if score_repair_accepted:
        threshold_rows: list[dict[str, object]] = [baseline_candidate]
        for atr_tolerance in EQHL_TUNING_GRID["atr_tolerance"]:
            for lookback_swings in EQHL_TUNING_GRID["lookback_swings"]:
                if float(atr_tolerance) == float(
                    EQHL_TUNING_BASELINE_PARAMS["atr_tolerance"]
                ) and int(lookback_swings) == int(
                    EQHL_TUNING_BASELINE_PARAMS["lookback_swings"]
                ):
                    continue
                threshold_rows.append(
                    _evaluate_threshold_candidate(
                        base_df,
                        atr_tolerance=float(atr_tolerance),
                        lookback_swings=int(lookback_swings),
                        accepted_v2_candidate=baseline_summary["score_calibration"][
                            "selected_candidate"
                        ],
                        baseline=baseline_candidate,
                    )
                )
        threshold_results = pd.DataFrame(threshold_rows).sort_values(
            [
                "accepted",
                "top_decile_avg_r",
                "top_quartile_avg_r",
                "inventory_growth_ratio",
            ],
            ascending=[False, False, False, False],
        )
        accepted = threshold_results.loc[
            threshold_results["accepted"] == True
        ]  # noqa: E712
        selected_threshold_candidate = (
            accepted.iloc[0].to_dict() if not accepted.empty else None
        )

    return {
        "baseline_candidate": baseline_candidate,
        "baseline_summary": baseline_summary,
        "score_calibration": score_calibration,
        "score_repair_accepted": score_repair_accepted,
        "calibration_events": calibration_events,
        "results_df": threshold_results.reset_index(drop=True),
        "selected_candidate": selected_threshold_candidate,
    }


def main() -> None:
    raw_df = pd.read_parquet(DATA_FILE)
    base_df = build_equal_hl_analysis_base_frame(raw_df)
    tuning = run_tuning(base_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tuning["score_calibration"]["component_direction_audit"].to_csv(
        COMPONENT_AUDIT_CSV, index=False
    )
    tuning["score_calibration"]["coarse_candidates"].to_csv(
        COARSE_CANDIDATES_CSV, index=False
    )
    tuning["score_calibration"]["refined_candidates"].to_csv(
        REFINED_CANDIDATES_CSV, index=False
    )
    tuning["results_df"].to_csv(SWEEP_CSV, index=False)

    summary = {
        "dataset": str(DATA_FILE),
        "baseline_params": EQHL_TUNING_BASELINE_PARAMS,
        "grid": EQHL_TUNING_GRID,
        "baseline_candidate": tuning["baseline_candidate"],
        "score_repair_accepted": tuning["score_repair_accepted"],
        "score_calibration_selected_candidate": tuning["score_calibration"][
            "selected_candidate"
        ],
        "selected_candidate": tuning["selected_candidate"],
        "defaults_should_change": tuning["selected_candidate"] is not None,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== EQUAL H/L TUNING SUMMARY ===")
    _print_summary(summary)
    print(f"\nWrote component audit CSV to: {COMPONENT_AUDIT_CSV}")
    print(f"Wrote coarse candidate CSV to: {COARSE_CANDIDATES_CSV}")
    print(f"Wrote refined candidate CSV to: {REFINED_CANDIDATES_CSV}")
    print(f"Wrote threshold sweep CSV to: {SWEEP_CSV}")
    print(f"Wrote summary JSON to: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
