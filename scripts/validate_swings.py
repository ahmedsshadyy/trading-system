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
from src.indicators.foundation.volatility import add_atr
from src.indicators.structure.swings import add_swings
from src.validation.common import cleanup_validation_artifacts

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/structure")
CACHE_ROOT = Path("data/validation_cache")
VALIDATOR_NAME = "validate_swings"

SWING_WINDOW = 4
SWING_RETRACE = 0.7
SWING_CONFIRM_BARS = 2


def _build_context(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_candle_schema(df, require_volume=False)
    out = out.sort_values("timestamp").reset_index(drop=True)
    out = add_atr(out)
    return add_swings(
        out,
        window=SWING_WINDOW,
        min_retrace_atr=SWING_RETRACE,
        min_confirm_bars=SWING_CONFIRM_BARS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", dest="html", action="store_true", default=True)
    parser.add_argument("--no-html", dest="html", action="store_false")
    parser.add_argument("--start-ts", default="2026-01-01")
    parser.add_argument("--end-ts", default="2026-03-15")
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--invalidate-cache", action="store_true")
    parser.add_argument("--cleanup-stale", action="store_true")
    parser.add_argument("--max-artifact-age-days", type=int, default=30)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cleanup_stale:
        removed = cleanup_validation_artifacts(
            cache_root=CACHE_ROOT,
            max_age_days=args.max_artifact_age_days,
            report_roots=[OUT_DIR],
        )
        print(f"cleanup_removed: {len(removed)}")

    raw = pd.read_parquet(DATA_FILE)
    graph = get_builtin_graph(
        "validate_swings",
        instrument="XAU_USD",
        timeframe="H4",
    )
    context = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={
            "start_ts": args.start_ts,
            "end_ts": args.end_ts,
            "n_windows": args.n_windows,
            "html": args.html,
            "out_dir": str(OUT_DIR),
        },
        cache_root=CACHE_ROOT,
        force=args.force,
        invalidate_cache=args.invalidate_cache,
    )
    graph_result = execute_graph(
        graph, context=context, target="swings_validation_bundle"
    )
    result = graph_result.output().payload

    print("\n=== SWINGS SUMMARY ===")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")

    print("\n=== SAMPLE HIGH SWING WINDOWS ===")
    for i, win in enumerate(result["high_windows"], start=1):
        print(f"\n--- high window {i} ---")
        print(win.to_string(index=True))

    print("\n=== SAMPLE LOW SWING WINDOWS ===")
    for i, win in enumerate(result["low_windows"], start=1):
        print(f"\n--- low window {i} ---")
        print(win.to_string(index=True))

    if args.html and result["html_path"] is not None:
        print(f"\nWrote chart to: {result['html_path']}")
    else:
        print("\nHTML output skipped. Pass --html to generate charts.")
    profile_path = CACHE_ROOT / VALIDATOR_NAME / "XAU_USD" / "H4" / "run-summary.json"
    graph_result.profiler.write_json(profile_path)
    print(f"Profiler summary: {profile_path}")


if __name__ == "__main__":
    main()
