"""
scripts/audit_equal_hl_features.py

Standalone EQH/EQL feature science diagnostic.

Runs ONLY the feature audit — no score grid search, no chart output, no full validation.

Usage:
    python scripts/audit_equal_hl_features.py
    python scripts/audit_equal_hl_features.py --instrument XAU_USD --timeframe H4
    python scripts/audit_equal_hl_features.py --no-csv   # skip saving CSV files

Output:
    1. Univariate IC table — rank_corr per feature per time split + stability classification
    2. Monotonicity table — 5-bucket avg_r per feature (full sample and halves)
    3. Feature-outcome Spearman matrix — features vs realized_r, hold, failed, swept, mfe, mae
    4. Feature-feature Spearman matrix — pairwise feature correlations
    5. Formation-delay bucket breakdown — avg_r per bucket per side and structural quartile
    6. Side-split IC — EQH vs EQL rank_corr to diagnose side asymmetry
    7. Active snapshot IC — pressure/distance/crowding features vs realized_r
       (labeled from calibration events, tested at each bar a cluster was active)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.research.equal_hl_research import build_equal_hl_research_table
from src.validation.indicators.equal_hl import (
    EQHL_MONOTONICITY_N_BUCKETS,
    build_eqhl_correlation_matrices,
    build_eqhl_feature_monotonicity_audit,
    build_eqhl_feature_science_table,
    build_eqhl_univariate_ic_table,
    build_equal_hl_active_snapshot_table,
    build_equal_hl_calibration_event_table,
    _prepare_calibration_events,
    _cross_trade_metrics,
)

DATA_DIR = Path("data/raw")
OUT_DIR = Path("notebooks/smc")


# ── Formatting helpers ─────────────────────────────────────────────────────────


def _fmt_corr(v: float) -> str:
    if not np.isfinite(v):
        return "   nan"
    sign = "+" if v >= 0 else "-"
    return f"{sign}{abs(v):.3f}"


def _fmt_r(v: float) -> str:
    if not np.isfinite(v):
        return "  nan "
    return f"{v:+.3f}"


def _section(title: str) -> None:
    width = 80
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_df(df: pd.DataFrame, *, float_fmt: str = "{:.3f}") -> None:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", float_fmt.format)
    print(df.to_string())
    pd.reset_option("display.width")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.float_format")


# ── Monotonicity bucket display ────────────────────────────────────────────────

_DETECT_TIME_FEATURES = [
    "formation_delay",
    "width_atr",
    "touch_count",
    "structural_score",
    "wick_ratio",
    "atr_percentile",
    "distance_atr_at_detect",
]


def _print_monotonicity_table(
    feature_science_table: pd.DataFrame,
    *,
    features: list[str] | None = None,
    n_buckets: int = EQHL_MONOTONICITY_N_BUCKETS,
) -> None:
    """Print per-feature 5-bucket avg_r for full sample, first half, second half."""
    raw_features = features if features is not None else _DETECT_TIME_FEATURES
    present = [f for f in raw_features if f in feature_science_table.columns]
    if not present:
        print("  (no raw features present)")
        return

    for feature in present:
        print(f"\n  {feature}")
        print(
            f"  {'':20s}  {'b0':>8}  {'b1':>8}  {'b2':>8}  {'b3':>8}  {'b4':>8}  {'mono':>6}"
        )
        segments = {"full": feature_science_table}
        if "subperiod" in feature_science_table.columns:
            first = feature_science_table.loc[
                feature_science_table["subperiod"] == "first_half"
            ]
            second = feature_science_table.loc[
                feature_science_table["subperiod"] == "second_half"
            ]
            if not first.empty:
                segments["first_half"] = first
            if not second.empty:
                segments["second_half"] = second

        for seg_name, seg_df in segments.items():
            feat_vals = pd.to_numeric(
                seg_df.get(feature, pd.Series(dtype=float)), errors="coerce"
            )
            tgt_vals = pd.to_numeric(
                seg_df.get("realized_r", pd.Series(dtype=float)), errors="coerce"
            )
            frame = pd.DataFrame({"f": feat_vals, "r": tgt_vals}).dropna()
            if len(frame) < n_buckets * 3:
                print(f"  {seg_name:20s}  (insufficient data: {len(frame)} rows)")
                continue
            try:
                frame["bucket"] = pd.qcut(
                    frame["f"], n_buckets, labels=False, duplicates="drop"
                )
            except Exception:
                print(f"  {seg_name:20s}  (bucketing failed)")
                continue
            bucket_avg = frame.groupby("bucket")["r"].mean().reindex(range(n_buckets))
            bucket_vals = [_fmt_r(float(v)) for v in bucket_avg]
            while len(bucket_vals) < 5:
                bucket_vals.append("   nan")
            vals_arr = bucket_avg.dropna().to_numpy()
            if len(vals_arr) >= 2:
                n_asc = int(np.sum(np.diff(vals_arr) > 0))
                mono = f"{n_asc}/{len(vals_arr)-1}"
            else:
                mono = "n/a"
            print(f"  {seg_name:20s}  {'  '.join(bucket_vals)}  {mono:>6}")


# ── Active snapshot Stage 2 features ──────────────────────────────────────────

_ACTIVE_PRESSURE_FEATURES = [
    "active_age",
    "distance_atr",
    "bars_since_touch",
    "touches_since_detect",
    "max_pen_atr",
    "inside_zone",
    "signed_dist_atr",
    "nearest_same_side_atr",
]


# ── Formation delay cross-tab ──────────────────────────────────────────────────


def _print_formation_delay_crosstab(calibration_events: pd.DataFrame) -> None:
    if (
        calibration_events.empty
        or "formation_delay_bucket" not in calibration_events.columns
    ):
        print("  (no formation_delay_bucket column)")
        return

    buckets = sorted(calibration_events["formation_delay_bucket"].dropna().unique())
    # By side
    print("\n  formation_delay_bucket × side (avg_r / win_rate / count)")
    header = f"  {'bucket':12s}  {'eqh avg_r':>9}  {'eqh win%':>8}  {'eqh n':>5}  "
    header += f"{'eql avg_r':>9}  {'eql win%':>8}  {'eql n':>5}  {'all avg_r':>9}"
    print(header)
    for bucket in buckets:
        b_df = calibration_events.loc[
            calibration_events["formation_delay_bucket"] == bucket
        ]
        eqh = b_df.loc[b_df["side"] == "eqh", "realized_r"].dropna()
        eql = b_df.loc[b_df["side"] == "eql", "realized_r"].dropna()
        all_r = b_df["realized_r"].dropna()
        eqh_wr = float((eqh > 0).mean()) if len(eqh) > 0 else np.nan
        eql_wr = float((eql > 0).mean()) if len(eql) > 0 else np.nan
        print(
            f"  {bucket:12s}  {_fmt_r(float(eqh.mean()) if len(eqh) > 0 else np.nan):>9}  "
            f"{eqh_wr:>7.1%}  {len(eqh):>5}  "
            f"{_fmt_r(float(eql.mean()) if len(eql) > 0 else np.nan):>9}  "
            f"{eql_wr:>7.1%}  {len(eql):>5}  "
            f"{_fmt_r(float(all_r.mean()) if len(all_r) > 0 else np.nan):>9}"
        )

    # By structural quartile
    if "structural_score_quartile" in calibration_events.columns:
        print("\n  formation_delay_bucket × structural_score_quartile (avg_r / count)")
        quartiles = sorted(
            calibration_events["structural_score_quartile"]
            .dropna()
            .replace("unknown", pd.NA)
            .dropna()
            .unique()
        )
        col_width = 10
        header = f"  {'bucket':12s}  " + "  ".join(
            f"{'Q'+str(q):>{col_width}}" for q in quartiles
        )
        print(header)
        for bucket in buckets:
            b_df = calibration_events.loc[
                calibration_events["formation_delay_bucket"] == bucket
            ]
            cells = []
            for q in quartiles:
                q_df = b_df.loc[
                    b_df["structural_score_quartile"] == q, "realized_r"
                ].dropna()
                if len(q_df) > 0:
                    cells.append(f"{_fmt_r(float(q_df.mean()))} ({len(q_df)})")
                else:
                    cells.append("   nan (0)")
            print(f"  {bucket:12s}  " + "  ".join(f"{c:>{col_width}}" for c in cells))


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="EQH/EQL feature science diagnostic")
    parser.add_argument("--instrument", default="XAU_USD")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    args = parser.parse_args()

    data_file = DATA_DIR / f"{args.instrument}_{args.timeframe}.parquet"
    if not data_file.exists():
        print(f"ERROR: data file not found: {data_file}")
        sys.exit(1)

    tag = f"{args.instrument}_{args.timeframe}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_file} ...")
    raw_df = pd.read_parquet(data_file)
    print(f"Building indicators ({len(raw_df)} rows) ...")
    df = build_research_indicators(
        raw_df,
        instrument=args.instrument,
        include_vp=False,
        include_avwap=False,
    )

    print("Building research table ...")
    research_table = build_equal_hl_research_table(df)
    print("Building calibration events ...")
    calibration_events = build_equal_hl_calibration_event_table(
        df, research_table=research_table
    )
    calibration_events, width_reference = _prepare_calibration_events(
        calibration_events
    )
    print("Building active snapshots ...")
    active_snapshots = build_equal_hl_active_snapshot_table(df)

    feature_science_table = build_eqhl_feature_science_table(
        calibration_events, research_table
    )

    n_events = len(calibration_events)
    n_eqh = int((calibration_events["side"] == "eqh").sum())
    n_eql = int((calibration_events["side"] == "eql").sum())
    n_first = int(
        (calibration_events.get("subperiod", pd.Series()) == "first_half").sum()
    )
    n_second = int(
        (calibration_events.get("subperiod", pd.Series()) == "second_half").sum()
    )

    print()
    print(f"Events: {n_events}  |  EQH: {n_eqh}  EQL: {n_eql}")
    print(f"First half: {n_first}  |  Second half: {n_second}")
    print(f"Width reference (80th pct): {width_reference:.4f}")

    # ── 0. Raw signal health ──────────────────────────────────────────────────
    _section("0. RAW SIGNAL HEALTH  (realized_r ignoring score — is detection sound?)")
    if (
        "realized_r" in calibration_events.columns
        and "subperiod" in calibration_events.columns
    ):
        for segment in ("overall", "first_half", "second_half"):
            if segment == "overall":
                seg_df = calibration_events
            else:
                seg_df = calibration_events.loc[
                    calibration_events["subperiod"] == segment
                ]
            if seg_df.empty:
                continue
            r = seg_df["realized_r"].dropna()
            if r.empty:
                continue
            # Top-quartile and bottom-decile avg_r
            q75 = r.quantile(0.75)
            q10 = r.quantile(0.10)
            top_q_r = float(r.loc[r >= q75].mean())
            bot_d_r = float(r.loc[r <= q10].mean())
            mean_r = float(r.mean())
            win_rate = float((r > 0).mean())
            label = f"{segment:12s}" if segment != "overall" else "overall     "
            print(
                f"\n  {label}  n={len(r):4d}  "
                f"mean={mean_r:+.4f}  win%={win_rate:.1%}  "
                f"top_q={top_q_r:+.4f}  bot_d={bot_d_r:+.4f}  "
                f"gap={top_q_r - bot_d_r:+.4f}"
            )
            # Side split within segment
            for side in ("eqh", "eql"):
                side_r = seg_df.loc[seg_df["side"] == side, "realized_r"].dropna()
                if len(side_r) < 5:
                    continue
                print(
                    f"    {side:4s}           n={len(side_r):4d}  "
                    f"mean={float(side_r.mean()):+.4f}  win%={float((side_r > 0).mean()):.1%}"
                )

        # Verdict
        print()
        fh = calibration_events.loc[
            calibration_events["subperiod"] == "first_half", "realized_r"
        ].dropna()
        sh = calibration_events.loc[
            calibration_events["subperiod"] == "second_half", "realized_r"
        ].dropna()
        if not sh.empty:
            sh_mean = float(sh.mean())
            sh_top_q = (
                float(sh.loc[sh >= sh.quantile(0.75)].mean())
                if len(sh) >= 4
                else np.nan
            )
            sh_bot_d = (
                float(sh.loc[sh <= sh.quantile(0.10)].mean())
                if len(sh) >= 10
                else np.nan
            )
            sh_gap = (
                sh_top_q - sh_bot_d
                if np.isfinite(sh_top_q) and np.isfinite(sh_bot_d)
                else np.nan
            )
            if sh_mean <= 0:
                verdict = "DETECTION WEAK in second half — raw mean ≤ 0. Pattern may have stopped working."
            elif np.isfinite(sh_gap) and sh_gap <= 0:
                verdict = "SCORING PROBLEM — raw mean positive but rank ordering lost in second half."
            elif np.isfinite(sh_gap) and sh_gap > 0:
                verdict = "RANK ORDERING PRESERVED in second half — scoring improvement or ML can recover IC."
            else:
                verdict = "Insufficient data for verdict."
            print(f"  Verdict: {verdict}")
    else:
        print("  (realized_r or subperiod column missing)")

    # ── 1. Univariate IC table ─────────────────────────────────────────────────
    _section("1. UNIVARIATE IC TABLE  (Spearman rank_corr per feature per segment)")
    ic_table = build_eqhl_univariate_ic_table(feature_science_table)
    if not ic_table.empty:
        # Format nicely
        display_cols = [
            c
            for c in [
                "overall",
                "first_half",
                "second_half",
                "eqh",
                "eql",
                "monotonicity",
                "stability",
            ]
            if c in ic_table.columns
        ]
        print(f"\n  {'feature':28s}", end="")
        for col in display_cols:
            print(f"  {col:>11s}", end="")
        print()
        print("  " + "-" * (28 + len(display_cols) * 13))
        for feat, row in ic_table.iterrows():
            print(f"  {feat:28s}", end="")
            for col in display_cols:
                val = row[col]
                if col == "stability":
                    print(f"  {str(val):>11s}", end="")
                elif col == "monotonicity":
                    if np.isfinite(float(val)):
                        print(f"  {float(val):>11.2f}", end="")
                    else:
                        print(f"  {'nan':>11s}", end="")
                else:
                    print(f"  {_fmt_corr(float(val)):>11s}", end="")
            print()

    # ── 2. Monotonicity (5-bucket avg_r) ──────────────────────────────────────
    _section("2. MONOTONICITY  (5-bucket avg_r  |  low bucket → high bucket)")
    _print_monotonicity_table(feature_science_table)

    # ── 3. Formation delay cross-tab ──────────────────────────────────────────
    _section(
        "3. FORMATION DELAY BREAKDOWN  (avg_r by bucket × side / structural quartile)"
    )
    _print_formation_delay_crosstab(calibration_events)

    # ── 4. Feature-outcome correlation matrix ─────────────────────────────────
    _section("4. FEATURE → OUTCOME CORRELATION MATRIX  (Spearman)")
    corr_mats = build_eqhl_correlation_matrices(feature_science_table)

    fvo = corr_mats["feature_vs_outcome"]
    cvo = corr_mats["component_vs_outcome"]
    if not fvo.empty:
        print("\n  Raw features:")
        _print_df(fvo)
    if not cvo.empty:
        print("\n  Processed components:")
        _print_df(cvo)

    # ── 5. Feature-feature correlation matrix ─────────────────────────────────
    _section("5. FEATURE × FEATURE CORRELATION MATRIX  (Spearman pairwise)")
    fvf = corr_mats["feature_vs_feature"]
    if not fvf.empty:
        _print_df(fvf)

    # ── 6. Side-split IC for unstable features ────────────────────────────────
    _section("6. SIDE-SPLIT IC  (EQH vs EQL — to diagnose side asymmetry)")
    mono_audit = build_eqhl_feature_monotonicity_audit(feature_science_table)
    if not mono_audit.empty:
        side_rows = mono_audit.loc[mono_audit["segment"].isin(["eqh", "eql"])][
            ["feature", "segment", "count", "rank_corr", "monotonicity_ratio"]
        ].pivot(
            index="feature",
            columns="segment",
            values=["rank_corr", "monotonicity_ratio"],
        )
        if not side_rows.empty:
            side_rows.columns = [f"{m}_{s}" for m, s in side_rows.columns]
            _print_df(side_rows)

    # ── 7. Active snapshot IC  (Stage 2 — fwd_r_3 as outcome) ────────────────
    _section(
        "7. ACTIVE SNAPSHOT IC  (Stage 2 features vs fwd_r_3 — causal per-bar forward return)"
    )
    snapshot_ic_table = pd.DataFrame()
    snapshot_mono_audit = pd.DataFrame()
    if active_snapshots.empty or "fwd_r_3" not in active_snapshots.columns:
        print("  (no active snapshot data or fwd_r_3 not populated)")
    else:
        # Rename fwd_r_3 → realized_r so the audit functions see the right outcome column
        snap_labeled = active_snapshots.rename(columns={"fwd_r_3": "realized_r"}).copy()
        present_pressure = [
            f for f in _ACTIVE_PRESSURE_FEATURES if f in snap_labeled.columns
        ]
        n_snap = len(snap_labeled)
        n_snap_eqh = int((snap_labeled.get("side", pd.Series()) == "eqh").sum())
        n_snap_eql = int((snap_labeled.get("side", pd.Series()) == "eql").sum())
        n_valid = int(snap_labeled["realized_r"].notna().sum())
        print(
            f"\n  Snapshot rows: {n_snap}  |  EQH: {n_snap_eqh}  EQL: {n_snap_eql}  |  labeled: {n_valid}"
        )
        print(f"  Features present: {present_pressure}")
        print(f"  Outcome: fwd_r_3 = direction × (close[t+3] - close[t]) / ATR[t]")

        snapshot_mono_audit = build_eqhl_feature_monotonicity_audit(snap_labeled)
        if not snapshot_mono_audit.empty:
            snap_pressure_audit = snapshot_mono_audit.loc[
                snapshot_mono_audit["feature"].isin(present_pressure)
            ]
            if not snap_pressure_audit.empty:
                snapshot_ic_table = build_eqhl_univariate_ic_table(snap_labeled)
                snapshot_ic_table = snapshot_ic_table.loc[
                    snapshot_ic_table.index.isin(present_pressure)
                ]
                if not snapshot_ic_table.empty:
                    display_cols = [
                        c
                        for c in [
                            "overall",
                            "first_half",
                            "second_half",
                            "eqh",
                            "eql",
                            "monotonicity",
                            "stability",
                        ]
                        if c in snapshot_ic_table.columns
                    ]
                    print(f"\n  {'feature':28s}", end="")
                    for col in display_cols:
                        print(f"  {col:>11s}", end="")
                    print()
                    print("  " + "-" * (28 + len(display_cols) * 13))
                    for feat, row in snapshot_ic_table.iterrows():
                        print(f"  {feat:28s}", end="")
                        for col in display_cols:
                            val = row[col]
                            if col == "stability":
                                print(f"  {str(val):>11s}", end="")
                            elif col == "monotonicity":
                                if np.isfinite(float(val)):
                                    print(f"  {float(val):>11.2f}", end="")
                                else:
                                    print(f"  {'nan':>11s}", end="")
                            else:
                                print(f"  {_fmt_corr(float(val)):>11s}", end="")
                        print()

                print("\n  Monotonicity (5-bucket avg_r — snapshot level):")
                _print_monotonicity_table(snap_labeled, features=present_pressure)

    # ── 8. Regime breakdown ───────────────────────────────────────────────────
    _section(
        "8. REGIME BREAKDOWN  (trend_aligned × subperiod — is context the missing piece?)"
    )
    if (
        "trend_aligned" in calibration_events.columns
        and "subperiod" in calibration_events.columns
        and "realized_r" in calibration_events.columns
    ):
        print(
            f"\n  {'segment':12s}  {'trend_aligned':>13s}  {'side':>4s}  {'n':>5s}  {'avg_r':>7s}  {'win%':>6s}"
        )
        print("  " + "-" * 60)
        for period in ("first_half", "second_half"):
            period_df = calibration_events.loc[
                calibration_events["subperiod"] == period
            ]
            if period_df.empty:
                continue
            for aligned_val in (1.0, 0.0):
                label = "aligned" if aligned_val == 1.0 else "counter"
                aligned_df = period_df.loc[
                    pd.to_numeric(period_df["trend_aligned"], errors="coerce")
                    == aligned_val
                ]
                for side in ("both", "eqh", "eql"):
                    if side == "both":
                        grp = aligned_df["realized_r"].dropna()
                    else:
                        grp = aligned_df.loc[
                            aligned_df["side"] == side, "realized_r"
                        ].dropna()
                    if len(grp) < 5:
                        continue
                    print(
                        f"  {period:12s}  {label:>13s}  {side:>4s}  {len(grp):>5d}  "
                        f"{float(grp.mean()):>+7.4f}  {float((grp > 0).mean()):>5.1%}"
                    )
            # NaN trend_aligned (no trend state available)
            unknown = period_df.loc[
                pd.to_numeric(period_df.get("trend_aligned"), errors="coerce").isna(),
                "realized_r",
            ].dropna()
            if len(unknown) >= 5:
                print(
                    f"  {period:12s}  {'no trend data':>13s}  {'all':>4s}  {len(unknown):>5d}  "
                    f"{float(unknown.mean()):>+7.4f}  {float((unknown > 0).mean()):>5.1%}"
                )

        # IC of trend_aligned specifically in each half
        print()
        for period in ("first_half", "second_half"):
            period_df = calibration_events.loc[
                calibration_events["subperiod"] == period
            ]
            if period_df.empty or "trend_aligned" not in period_df.columns:
                continue
            x = pd.to_numeric(period_df["trend_aligned"], errors="coerce")
            y = pd.to_numeric(period_df["realized_r"], errors="coerce")
            valid = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(valid) >= 10 and valid["x"].nunique() > 1:
                ic = float(
                    valid["x"]
                    .rank(method="average")
                    .corr(valid["y"].rank(method="average"))
                )
                print(f"  trend_aligned IC  {period:12s}: {ic:+.4f}")
    else:
        print(
            "  (trend_aligned not yet populated — run with enriched calibration events)"
        )

    print()

    # ── CSV output ────────────────────────────────────────────────────────────
    if not args.no_csv:
        ic_csv = OUT_DIR / f"equal_hl_feature_ic_table_{tag}.csv"
        mono_csv = OUT_DIR / f"equal_hl_feature_monotonicity_{tag}.csv"
        fvo_csv = OUT_DIR / f"equal_hl_feature_vs_outcome_corr_{tag}.csv"
        fvf_csv = OUT_DIR / f"equal_hl_feature_vs_feature_corr_{tag}.csv"
        snap_ic_csv = OUT_DIR / f"equal_hl_snapshot_ic_table_{tag}.csv"
        snap_mono_csv = OUT_DIR / f"equal_hl_snapshot_monotonicity_{tag}.csv"

        if not ic_table.empty:
            ic_table.to_csv(ic_csv)
            print(f"Saved: {ic_csv}")

        mono_full = build_eqhl_feature_monotonicity_audit(feature_science_table)
        if not mono_full.empty:
            mono_full.to_csv(mono_csv, index=False)
            print(f"Saved: {mono_csv}")

        if not fvo.empty:
            fvo.to_csv(fvo_csv)
            print(f"Saved: {fvo_csv}")

        if not fvf.empty:
            fvf.to_csv(fvf_csv)
            print(f"Saved: {fvf_csv}")

        if not snapshot_ic_table.empty:
            snapshot_ic_table.to_csv(snap_ic_csv)
            print(f"Saved: {snap_ic_csv}  (Stage 2 IC vs fwd_r_3)")

        if not snapshot_mono_audit.empty:
            snapshot_mono_audit.to_csv(snap_mono_csv, index=False)
            print(f"Saved: {snap_mono_csv}  (Stage 2 monotonicity vs fwd_r_3)")


if __name__ == "__main__":
    main()
