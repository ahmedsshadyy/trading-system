from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.foundation.volatility import add_atr
from src.indicators.research.fvg_research import build_fvg_research_table
from src.indicators.smc.fvg import collect_fvg_debug_tables
from src.indicators.smc.fvg_fill import add_fvg_fill
from src.indicators.smc.ifvg import add_ifvg
from src.validation.indicators.fvg import summarize_fvg, validate_fvg

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/smc")
PLOT_START = pd.Timestamp("2026-01-10", tz="UTC")


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


def _print_row_audit_table(
    title: str,
    rows: list[dict[str, object]],
) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("none")
        return
    print(pd.DataFrame(rows).to_string(index=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FVG/FVG fill/IFVG output.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip the old-no-expiry comparison pass and sample event windows. "
            "Keeps the chart and main summary."
        ),
    )
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Run the validation summaries without generating the HTML chart.",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=5,
        help="Number of sample bull/bear event windows to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = add_atr(df)
    debug_tables = collect_fvg_debug_tables(df)
    df = debug_tables["frame"]
    research_table = build_fvg_research_table(df, debug_tables=debug_tables)
    df = add_fvg_fill(df, debug_tables=debug_tables)
    df = add_ifvg(df, debug_tables=debug_tables)

    old_no_expiry_research_table = None
    if not args.fast:
        old_no_expiry_debug = collect_fvg_debug_tables(
            df[["timestamp", "open", "high", "low", "close", "atr_14"]].copy(),
            max_active_age_bars=len(df) + 1,
        )
        old_no_expiry_research_table = build_fvg_research_table(
            old_no_expiry_debug["frame"], debug_tables=old_no_expiry_debug
        )

    plot_end = pd.to_datetime(df["timestamp"], utc=True).max()
    plot_df = df[df["timestamp"] >= PLOT_START].copy()
    if plot_df.empty:
        raise ValueError(
            f"No XAU_USD H4 rows found for validation window starting {PLOT_START}."
        )

    title = (
        "FVG Validation — XAU_USD H4 "
        f"({PLOT_START.date()} to {plot_end.date()}, chart window only)"
    )

    if args.numeric_only:
        result = {
            "summary": summarize_fvg(
                df,
                full_df=df,
                debug_tables=debug_tables,
                research_table=research_table,
                old_no_expiry_research_table=old_no_expiry_research_table,
            ),
            "bull_windows": [],
            "bear_windows": [],
            "html_path": None,
        }
    else:
        result = validate_fvg(
            plot_df,
            full_df=df,
            debug_tables=debug_tables,
            outpath=OUT_DIR / "fvg_validation_XAU_USD_H4.html",
            title=title,
            n_windows=(0 if args.fast else args.n_windows),
            research_table=research_table,
            old_no_expiry_research_table=old_no_expiry_research_table,
        )
        result["summary"] = summarize_fvg(
            df,
            full_df=df,
            debug_tables=debug_tables,
            research_table=research_table,
            old_no_expiry_research_table=old_no_expiry_research_table,
        )

    print("\n=== FVG SUMMARY ===")
    _print_summary(result["summary"])

    if result["summary"].get("global_reconciliation_report") is not None:
        print("\n=== GLOBAL RECONCILIATION REPORT ===")
        _print_summary(result["summary"]["global_reconciliation_report"])

    global_report = result["summary"].get("global_reconciliation_report") or {}
    if global_report.get("continuation_without_touch_audit") is not None:
        print("\n=== GLOBAL CONTINUATION WITHOUT TOUCH AUDIT ===")
        _print_summary(global_report["continuation_without_touch_audit"])

    if result["summary"].get("window_forensic_reconciliation") is not None:
        print("\n=== FORENSIC WINDOW RECONCILIATION ===")
        _print_summary(result["summary"]["window_forensic_reconciliation"])

    if result["summary"].get("overlap_fill_forensics") is not None:
        print("\n=== OVERLAP FILL FORENSICS ===")
        _print_summary(result["summary"]["overlap_fill_forensics"])

    if result["summary"].get("core_fill_ownership_reconciliation") is not None:
        print("\n=== CORE FILL OWNERSHIP RECONCILIATION ===")
        _print_summary(result["summary"]["core_fill_ownership_reconciliation"])

    if result["summary"].get("live_historical_fill_split") is not None:
        print("\n=== LIVE VS HISTORICAL FILL SPLIT ===")
        _print_summary(result["summary"]["live_historical_fill_split"])

    if result["summary"].get("fill_selection_reconciliation") is not None:
        print("\n=== FILL SELECTION RECONCILIATION ===")
        _print_summary(result["summary"]["fill_selection_reconciliation"])
        for side in ("bull", "bear"):
            side_summary = result["summary"]["fill_selection_reconciliation"].get(side)
            if side_summary is None:
                continue
            _print_row_audit_table(
                f"{side.upper()} STRUCTURAL VS FILL ROW AUDIT",
                side_summary.get("row_level_mismatch_audit", []),
            )

    if result["summary"].get("ifvg_summary") is not None:
        print("\n=== IFVG SUMMARY ===")
        _print_summary(result["summary"]["ifvg_summary"])

    if result["summary"].get("ifvg_source_linkage_audit") is not None:
        print("\n=== IFVG SOURCE LINKAGE AUDIT ===")
        _print_summary(result["summary"]["ifvg_source_linkage_audit"])

    if result["summary"].get("ifvg_candidate_filter_audit") is not None:
        print("\n=== IFVG CANDIDATE FILTER AUDIT ===")
        _print_summary(result["summary"]["ifvg_candidate_filter_audit"])

    if result["summary"].get("ifvg_lifecycle_audit") is not None:
        print("\n=== IFVG LIFECYCLE AUDIT ===")
        _print_summary(result["summary"]["ifvg_lifecycle_audit"])

    if result["summary"].get("ifvg_distributional_fingerprint") is not None:
        print("\n=== IFVG DISTRIBUTIONAL FINGERPRINT ===")
        _print_summary(result["summary"]["ifvg_distributional_fingerprint"])

    if result["summary"].get("ifvg_universe_reconciliation") is not None:
        print("\n=== IFVG UNIVERSE RECONCILIATION ===")
        _print_summary(result["summary"]["ifvg_universe_reconciliation"])

    if result["summary"].get("ifvg_active_pool_policy_audit") is not None:
        print("\n=== IFVG ACTIVE POOL POLICY AUDIT ===")
        _print_summary(result["summary"]["ifvg_active_pool_policy_audit"])

    if result["bull_windows"]:
        print("\n=== SAMPLE BULL FVG WINDOWS ===")
        for i, win in enumerate(result["bull_windows"], start=1):
            print(f"\n--- bull window {i} ---")
            print(win.to_string(index=True))

    if result["bear_windows"]:
        print("\n=== SAMPLE BEAR FVG WINDOWS ===")
        for i, win in enumerate(result["bear_windows"], start=1):
            print(f"\n--- bear window {i} ---")
            print(win.to_string(index=True))

    if result["html_path"] is not None:
        print(f"\nWrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
