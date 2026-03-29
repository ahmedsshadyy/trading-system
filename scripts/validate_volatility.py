"""
scripts/validate_volatility.py

Interactive HTML validation chart for the volatility feature layer.
Filters to 2025-01-10 → end of available data.

Layout (4 rows, shared x-axis):
  Row 1  Candlestick — orange tint = ATR %-rank > 70 (high vol regime)
                       triangles    = wide-range bars
  Row 2  Volatility regime gauges — ATR %-rank & BB-width %-rank
         (0 = calmest ever seen,  100 = most volatile ever seen)
         Dotted lines at 30 (low) and 70 (high) as reference thresholds.
  Row 3  Vol model comparison — GARCH, Realized Vol 20, SV Filtered
         All three should broadly agree; divergence flags model uncertainty.
  Row 4  Volatility shock residuals — how unusual is the current vol spike?
         Bars beyond ±2σ (red/blue) are statistically extreme.

Output: notebooks/foundation/volatility_validation_{instrument}_{timeframe}.html
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.indicators.foundation.volatility import add_volatility_features
from src.indicators.pipelines.build_research import build_research_indicators
from src.validation.common.reporting import (
    get_logger,
    load_windowed_parquet,
    mark_report_written,
    report_fingerprint,
    should_write_report,
)
from src.validation.indicators.volatility import validate_volatility

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATE_FROM = "2026-01-10"
OUT_DIR = Path("notebooks/foundation")
RUNS = (("XAU_USD", "H4"),)
DEFAULT_TAIL_ROWS = 1500


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------


def _build_vol_chart(df: pd.DataFrame, *, title: str) -> go.Figure:
    ts = pd.to_datetime(
        df["timestamp"] if "timestamp" in df.columns else df.index,
        utc=True,
        errors="coerce",
    )
    x = ts

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.38, 0.20, 0.24, 0.18],
        subplot_titles=[
            (
                "Price  |  <span style='background:rgba(255,165,0,0.25)'>orange tint</span>"
                " = ATR %-rank > 70 (elevated vol)  "
                "| ▲ = wide-range bar"
            ),
            (
                "Volatility Regime  —  ATR %-rank  &  BB-width %-rank  "
                "(0 = calmest,  100 = most volatile in lookback window)"
            ),
            (
                "Vol Model Comparison  —  GARCH(1,1),  Realized Vol 20,  SV Filtered  "
                "[same units: agree = healthy,  diverge = regime shift]"
            ),
            (
                "Volatility Shock  —  standardised GARCH residuals  "
                "(|z| > 2σ = statistically extreme spike)"
            ),
        ],
    )

    # ── Row 1: Candlestick ─────────────────────────────────────────────────
    if all(c in df.columns for c in ("open", "high", "low", "close")):
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
                increasing_line_color="rgba(34,139,34,0.9)",
                decreasing_line_color="rgba(190,0,0,0.9)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # ATR %-rank background shading: mark contiguous high-vol zones
    if "atr_pct_rank_100" in df.columns:
        rank = pd.to_numeric(df["atr_pct_rank_100"], errors="coerce").fillna(0)
        high_vol = (rank > 70).to_numpy()
        in_zone = False
        zone_start_ts = None
        for i, (is_hv, t_val) in enumerate(zip(high_vol, x)):
            if is_hv and not in_zone:
                zone_start_ts = t_val
                in_zone = True
            elif not is_hv and in_zone:
                fig.add_vrect(
                    x0=zone_start_ts,
                    x1=x.iloc[i - 1],
                    fillcolor="rgba(255,165,0,0.12)",
                    line_width=0,
                    layer="below",
                )
                in_zone = False
        if in_zone:
            fig.add_vrect(
                x0=zone_start_ts,
                x1=x.iloc[-1],
                fillcolor="rgba(255,165,0,0.12)",
                line_width=0,
                layer="below",
            )

    # Wide-range bar markers
    if "wide_range_bar_flag" in df.columns and "high" in df.columns:
        mask_wrb = df["wide_range_bar_flag"].astype(int) == 1
        if mask_wrb.any():
            fig.add_trace(
                go.Scatter(
                    x=x[mask_wrb],
                    y=df.loc[mask_wrb, "high"],
                    mode="markers",
                    name="Wide Range Bar ▲",
                    marker=dict(
                        symbol="triangle-up",
                        size=9,
                        color="rgba(255,140,0,0.85)",
                        line=dict(color="white", width=0.5),
                    ),
                ),
                row=1,
                col=1,
            )

    # ── Row 2: ATR %-rank + BB-width %-rank ───────────────────────────────
    for col, color, name in (
        ("atr_pct_rank_100", "rgba(30,100,220,0.85)", "ATR %-rank"),
        ("bb_width_pct_rank_100", "rgba(150,0,200,0.75)", "BB-width %-rank"),
    ):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x, y=s, mode="lines", name=name, line=dict(color=color, width=1.8)
            ),
            row=2,
            col=1,
        )

    # Reference thresholds
    for level, label, pos in (
        (70, "High vol (70)", "top right"),
        (30, "Low vol (30)", "bottom right"),
    ):
        fig.add_hline(
            y=level,
            line_dash="dot",
            line_color="rgba(100,100,100,0.50)",
            annotation_text=label,
            annotation_position=pos,
            row=2,
            col=1,
        )

    # ── Row 3: Vol model comparison ────────────────────────────────────────
    for col, color, width, name in (
        ("garch_vol_1_1", "rgba(210,40,40,0.85)", 2.2, "GARCH vol"),
        ("rv_close_20", "rgba(30,100,220,0.75)", 1.6, "Realized vol (20)"),
        ("sv_vol_filtered", "rgba(0,155,90,0.80)", 1.6, "SV filtered"),
    ):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x, y=s, mode="lines", name=name, line=dict(color=color, width=width)
            ),
            row=3,
            col=1,
        )

    # ── Row 4: Volatility shock residuals ─────────────────────────────────
    if "vol_shock_std_resid" in df.columns:
        resid = pd.to_numeric(df["vol_shock_std_resid"], errors="coerce").fillna(0)
        bar_colors = [
            (
                "rgba(200,0,0,0.80)"
                if v > 2
                else "rgba(0,0,200,0.80)" if v < -2 else "rgba(120,120,120,0.55)"
            )
            for v in resid
        ]
        fig.add_trace(
            go.Bar(
                x=x,
                y=resid,
                name="Vol shock (z-score)",
                marker_color=bar_colors,
                showlegend=True,
            ),
            row=4,
            col=1,
        )

        for level, label, pos in (
            (2.0, "+2σ threshold", "top right"),
            (-2.0, "−2σ threshold", "bottom right"),
        ):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="rgba(180,0,0,0.55)",
                annotation_text=label,
                annotation_position=pos,
                row=4,
                col=1,
            )

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        template="plotly_white",
        height=1350,
        hovermode="x unified",
        barmode="relative",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )

    for i in range(1, 5):
        ax = "xaxis" if i == 1 else f"xaxis{i}"
        fig.update_layout(**{ax: {"rangeslider": {"visible": False}}})

    fig.update_yaxes(title_text="Price (XAU/USD)", row=1, col=1)
    fig.update_yaxes(title_text="Percentile rank [0–100]", range=[0, 105], row=2, col=1)
    fig.update_yaxes(title_text="Volatility estimate", rangemode="tozero", row=3, col=1)
    fig.update_yaxes(title_text="Std deviations (σ)", row=4, col=1)

    return fig


# ---------------------------------------------------------------------------
# Summary printer + main
# ---------------------------------------------------------------------------


def _print_summary(d: object, *, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                _print_summary(v, indent=indent + 2)
            else:
                print(f"{prefix}{k}: {v}")
    else:
        print(f"{prefix}{d}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate volatility features and optionally produce an HTML chart."
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Write the HTML chart. Default mode writes no artifacts.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load the full dataset instead of a trailing validation window.",
    )
    parser.add_argument(
        "--tail-rows",
        type=int,
        default=DEFAULT_TAIL_ROWS,
        help=f"Trailing rows to load when --full is not set. Default: {DEFAULT_TAIL_ROWS}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate HTML even if the validation input fingerprint is unchanged.",
    )
    parser.add_argument(
        "--with-regime",
        action="store_true",
        help="Run full indicator pipeline to include regime in vol-by-regime audit (slow).",
    )
    args = parser.parse_args()
    logger = get_logger("validate_volatility")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        if not data_file.exists():
            logger.warning("skip %s %s: %s not found", instrument, timeframe, data_file)
            continue

        logger.info("validating %s %s", instrument, timeframe)

        raw = load_windowed_parquet(data_file, full=args.full, tail_rows=args.tail_rows)

        live_df = add_volatility_features(raw, include_research_only=False)
        research_df = add_volatility_features(raw, include_research_only=True)

        if args.with_regime:
            enriched = build_research_indicators(raw, instrument=instrument)
            # Merge regime column into research_df
            if "regime" in enriched.columns:
                research_df = research_df.copy()
                research_df["regime"] = enriched["regime"].values

        logger.info("rows loaded=%s", len(research_df))

        title = f"Volatility Features — {instrument} {timeframe}  |  {DATE_FROM} → end"

        # Numerics: full dataset
        result = validate_volatility(
            research_df,
            live_df=live_df,
            research_df=research_df,
            outpath=None,
            title=title,
        )
        print()
        _print_summary(result["summary"])

        if args.html:
            ts = pd.to_datetime(
                (
                    research_df["timestamp"]
                    if "timestamp" in research_df.columns
                    else research_df.index
                ),
                utc=True,
                errors="coerce",
            )
            plot_df = research_df[ts >= pd.Timestamp(DATE_FROM, tz="UTC")].copy()
            html_path = OUT_DIR / f"volatility_validation_{instrument}_{timeframe}.html"
            fingerprint = report_fingerprint(
                plot_df,
                extra={
                    "instrument": instrument,
                    "timeframe": timeframe,
                    "mode": "volatility",
                    "date_from": DATE_FROM,
                },
            )
            if should_write_report(
                html_path, fingerprint=fingerprint, force=args.force
            ):
                fig = _build_vol_chart(plot_df, title=title)
                fig.write_html(str(html_path))
                mark_report_written(html_path, fingerprint=fingerprint)
                logger.info("chart saved -> %s", html_path)
            else:
                logger.info("chart unchanged, skipped -> %s", html_path)
        else:
            logger.info("html disabled; summary only")


if __name__ == "__main__":
    main()
