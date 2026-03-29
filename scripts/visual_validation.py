"""
Visual Indicator Validation Script.

Loads XAU_USD H4 data, runs indicator pipeline, and generates
interactive Plotly HTML charts with all structural detections marked
for visual comparison against TradingView.

Also prints text spot-checks for BOS, CHoCH, FVG, Sweeps, VP levels,
prev day/week H/L, and BB width for manual verification.

Usage: poetry run python scripts/visual_validation.py
Output: HTML charts in notebooks/ + text spot-checks in terminal
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.indicators import build_all_indicators
from src.indicators._helpers.schema import normalize_candle_schema


def load_candles(instrument: str, timeframe: str, engine) -> pd.DataFrame:
    query = f"""
        SELECT timestamp, open, high, low, close, volume, spread
        FROM candles
        WHERE instrument = '{instrument}' AND timeframe = '{timeframe}'
        ORDER BY timestamp ASC
    """
    return normalize_candle_schema(pd.read_sql(query, engine), require_volume=True)


def build_validation_chart(
    df: pd.DataFrame,
    title: str,
    start_date: str,
    end_date: str,
) -> go.Figure:
    """Build candlestick chart with all structural detections overlaid."""

    ts_start = pd.Timestamp(start_date, tz="UTC")
    ts_end = pd.Timestamp(end_date, tz="UTC")
    mask = (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
    segment = df[mask].copy().reset_index(drop=True)

    if len(segment) == 0:
        print(f"  No data for {start_date} to {end_date}")
        return None

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=[
            f"{title} — Structural Detections",
            "RSI 14",
            "ADX 14",
            "Volume Ratio",
        ],
    )

    ts = segment["timestamp"]

    # Row 1: Candlestick + overlays
    fig.add_trace(
        go.Candlestick(
            x=ts,
            open=segment["open"],
            high=segment["high"],
            low=segment["low"],
            close=segment["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ts, y=segment["ema_20"], name="EMA 20", line=dict(color="blue", width=1)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts, y=segment["ema_50"], name="EMA 50", line=dict(color="orange", width=1)
        ),
        row=1,
        col=1,
    )

    # Swing Highs / Lows
    sh = segment[segment["swing_high"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sh["timestamp"],
            y=sh["high"] * 1.001,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="red"),
            name="Swing High",
        ),
        row=1,
        col=1,
    )

    sl = segment[segment["swing_low"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sl["timestamp"],
            y=sl["low"] * 0.999,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="green"),
            name="Swing Low",
        ),
        row=1,
        col=1,
    )

    # BOS
    bos_b = segment[segment["bos_bull"] == 1]
    fig.add_trace(
        go.Scatter(
            x=bos_b["timestamp"],
            y=bos_b["high"] * 1.002,
            mode="markers",
            marker=dict(symbol="star", size=12, color="lime"),
            name="BOS Bull",
        ),
        row=1,
        col=1,
    )

    bos_r = segment[segment["bos_bear"] == 1]
    fig.add_trace(
        go.Scatter(
            x=bos_r["timestamp"],
            y=bos_r["low"] * 0.998,
            mode="markers",
            marker=dict(symbol="star", size=12, color="magenta"),
            name="BOS Bear",
        ),
        row=1,
        col=1,
    )

    # CHoCH
    ch_b = segment[segment["choch_bull"] == 1]
    fig.add_trace(
        go.Scatter(
            x=ch_b["timestamp"],
            y=ch_b["high"] * 1.003,
            mode="markers",
            marker=dict(symbol="diamond", size=12, color="cyan"),
            name="CHoCH Bull",
        ),
        row=1,
        col=1,
    )

    ch_r = segment[segment["choch_bear"] == 1]
    fig.add_trace(
        go.Scatter(
            x=ch_r["timestamp"],
            y=ch_r["low"] * 0.997,
            mode="markers",
            marker=dict(symbol="diamond", size=12, color="yellow"),
            name="CHoCH Bear",
        ),
        row=1,
        col=1,
    )

    # FVG zones
    for _, row in segment[segment["fvg_bull"] == 1].iterrows():
        if not np.isnan(row.get("fvg_bull_low", np.nan)):
            fig.add_shape(
                type="rect",
                x0=row["timestamp"],
                x1=row["timestamp"] + pd.Timedelta(hours=16),
                y0=row["fvg_bull_low"],
                y1=row["fvg_bull_high"],
                fillcolor="rgba(0,255,0,0.1)",
                line=dict(color="green", width=1),
                row=1,
                col=1,
            )

    for _, row in segment[segment["fvg_bear"] == 1].iterrows():
        if not np.isnan(row.get("fvg_bear_low", np.nan)):
            fig.add_shape(
                type="rect",
                x0=row["timestamp"],
                x1=row["timestamp"] + pd.Timedelta(hours=16),
                y0=row["fvg_bear_low"],
                y1=row["fvg_bear_high"],
                fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="red", width=1),
                row=1,
                col=1,
            )

    # OB zones
    for _, row in segment[segment["ob_bull"] == 1].iterrows():
        if not np.isnan(row.get("ob_bull_low", np.nan)):
            fig.add_shape(
                type="rect",
                x0=row["timestamp"],
                x1=row["timestamp"] + pd.Timedelta(hours=24),
                y0=row["ob_bull_low"],
                y1=row["ob_bull_high"],
                fillcolor="rgba(0,100,255,0.15)",
                line=dict(color="blue", width=2, dash="dash"),
                row=1,
                col=1,
            )

    for _, row in segment[segment["ob_bear"] == 1].iterrows():
        if not np.isnan(row.get("ob_bear_low", np.nan)):
            fig.add_shape(
                type="rect",
                x0=row["timestamp"],
                x1=row["timestamp"] + pd.Timedelta(hours=24),
                y0=row["ob_bear_low"],
                y1=row["ob_bear_high"],
                fillcolor="rgba(255,100,0,0.15)",
                line=dict(color="orange", width=2, dash="dash"),
                row=1,
                col=1,
            )

    # Sweeps
    sw_h = segment[segment["sweep_high"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sw_h["timestamp"],
            y=sw_h["high"],
            mode="markers",
            marker=dict(symbol="x", size=14, color="red", line=dict(width=2)),
            name="Sweep High",
        ),
        row=1,
        col=1,
    )

    sw_l = segment[segment["sweep_low"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sw_l["timestamp"],
            y=sw_l["low"],
            mode="markers",
            marker=dict(symbol="x", size=14, color="green", line=dict(width=2)),
            name="Sweep Low",
        ),
        row=1,
        col=1,
    )

    # Displacement candles
    disp = segment[segment["displacement_candle"] == 1]
    fig.add_trace(
        go.Scatter(
            x=disp["timestamp"],
            y=disp["close"],
            mode="markers",
            marker=dict(
                symbol="square",
                size=8,
                color="white",
                line=dict(width=1, color="black"),
            ),
            name="Displacement",
        ),
        row=1,
        col=1,
    )

    # RSI divergence
    div_bear = segment[segment["rsi_div_bearish"] == 1]
    fig.add_trace(
        go.Scatter(
            x=div_bear["timestamp"],
            y=div_bear["high"] * 1.004,
            mode="markers+text",
            text="D",
            textposition="top center",
            marker=dict(symbol="triangle-down", size=8, color="red"),
            name="RSI Div Bear",
            textfont=dict(size=8),
        ),
        row=1,
        col=1,
    )

    div_bull = segment[segment["rsi_div_bullish"] == 1]
    fig.add_trace(
        go.Scatter(
            x=div_bull["timestamp"],
            y=div_bull["low"] * 0.996,
            mode="markers+text",
            text="D",
            textposition="bottom center",
            marker=dict(symbol="triangle-up", size=8, color="green"),
            name="RSI Div Bull",
            textfont=dict(size=8),
        ),
        row=1,
        col=1,
    )

    # Row 2: RSI
    fig.add_trace(
        go.Scatter(
            x=ts, y=segment["rsi_14"], name="RSI", line=dict(color="purple", width=1)
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # Row 3: ADX + regime background
    fig.add_trace(
        go.Scatter(
            x=ts, y=segment["adx_14"], name="ADX", line=dict(color="brown", width=1)
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=25, line_dash="dash", line_color="orange", opacity=0.5, row=3, col=1
    )
    fig.add_hline(y=20, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)

    # Row 4: Volume ratio
    fig.add_trace(
        go.Bar(
            x=ts,
            y=segment.get("vol_ratio", pd.Series()),
            name="Vol Ratio",
            marker_color="gray",
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=1.5, line_dash="dash", line_color="blue", opacity=0.5, row=4, col=1)

    fig.update_layout(
        height=1200,
        title_text=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        legend=dict(font=dict(size=9)),
    )
    return fig


def main():
    engine = create_engine(os.getenv("DATABASE_URL"))
    instrument = "XAU_USD"
    tf = "H4"

    print(f"Loading {instrument} {tf}...")
    df = load_candles(instrument, tf, engine)
    print(f"  {len(df):,} candles")

    print("Running indicator pipeline...")
    df = build_all_indicators(df, instrument=instrument, include_vp=True)
    print(f"  {len(df.columns)} columns")

    # --- Charts ---
    periods = [
        ("2025-12-01", "2026-03-14", "Recent Dec25-Mar26"),
        ("2024-06-01", "2024-09-01", "Mid 2024 Gold Rally"),
        ("2023-01-01", "2023-04-01", "Early 2023"),
    ]

    output_dir = Path(__file__).parent.parent / "notebooks"
    output_dir.mkdir(exist_ok=True)

    for start, end, label in periods:
        print(f"\nChart: {label}")
        fig = build_validation_chart(df, f"XAU_USD H4 — {label}", start, end)
        if fig:
            fname = output_dir / f"validation_{start}_{end}.html"
            fig.write_html(str(fname))
            print(f"  Saved: {fname}")

    # --- Text Spot-Checks ---
    recent = df[df["timestamp"] >= pd.Timestamp("2026-01-01", tz="UTC")]

    print("\n" + "=" * 70)
    print("BOS / CHoCH SPOT-CHECK (2026)")
    print("=" * 70)
    for name, col in [
        ("BOS Bull", "bos_bull"),
        ("BOS Bear", "bos_bear"),
        ("CHoCH Bull", "choch_bull"),
        ("CHoCH Bear", "choch_bear"),
    ]:
        events = recent[recent[col] == 1].tail(5)
        if len(events):
            print(f"\n  Last 5 {name}:")
            for _, r in events.iterrows():
                print(
                    f"    {r['timestamp']}  C={r['close']:.2f}  "
                    f"SH={r['last_swing_high']:.2f}  SL={r['last_swing_low']:.2f}  "
                    f"trend={r['trend_state']}"
                )

    print("\n" + "=" * 70)
    print("FVG SPOT-CHECK (2026)")
    print("=" * 70)
    for name, col, lo_c, hi_c in [
        ("Bull", "fvg_bull", "fvg_bull_low", "fvg_bull_high"),
        ("Bear", "fvg_bear", "fvg_bear_low", "fvg_bear_high"),
    ]:
        fvgs = recent[recent[col] == 1].tail(5)
        if len(fvgs):
            print(f"\n  Last 5 {name} FVGs:")
            for _, r in fvgs.iterrows():
                print(
                    f"    {r['timestamp']}  zone={r[lo_c]:.2f}–{r[hi_c]:.2f}  "
                    f"size={r.get('fvg_size_atr', np.nan):.2f} ATR"
                )

    print("\n" + "=" * 70)
    print("SWEEP SPOT-CHECK (2026)")
    print("=" * 70)
    for name, col in [("High", "sweep_high"), ("Low", "sweep_low")]:
        sweeps = recent[recent[col] == 1].tail(5)
        if len(sweeps):
            print(f"\n  Last 5 Sweep {name}:")
            for _, r in sweeps.iterrows():
                print(
                    f"    {r['timestamp']}  H={r['high']:.2f}  L={r['low']:.2f}  "
                    f"C={r['close']:.2f}  mag={r.get('sweep_magnitude', np.nan):.2f} ATR"
                )

    print("\n" + "=" * 70)
    print("BB WIDTH SPOT-CHECK (last 5 candles)")
    print("=" * 70)
    last5 = df.tail(5)
    for _, r in last5.iterrows():
        print(
            f"  {r['timestamp']}  bb_width={r.get('bb_width', np.nan):.6f}  "
            f"pct={r.get('bb_width_pct_50', np.nan):.1f}  "
            f"below_40={r.get('bb_width_below_40', np.nan)}"
        )

    print("\n" + "=" * 70)
    print("VOLUME PROFILE + REFERENCE LEVELS (last candle)")
    print("=" * 70)
    last = df.iloc[-1]
    print(f"  VP POC:  {last.get('vp_poc', np.nan):.2f}")
    print(f"  VP VAH:  {last.get('vp_vah', np.nan):.2f}")
    print(f"  VP VAL:  {last.get('vp_val', np.nan):.2f}")
    print(f"  POC dist (ATR): {last.get('vp_poc_distance_atr', np.nan):.2f}")
    print(f"  Prev Day H: {last.get('prev_day_high', np.nan)}")
    print(f"  Prev Day L: {last.get('prev_day_low', np.nan)}")
    print(f"  Prev Week H: {last.get('prev_week_high', np.nan)}")
    print(f"  Prev Week L: {last.get('prev_week_low', np.nan)}")
    print(f"  Near round #: {last.get('near_round_number', np.nan)}")
    print(f"  Regime: {last.get('regime_label', 'N/A')}")

    print("\nDone. Open HTML files in notebooks/ and compare against TradingView.")


if __name__ == "__main__":
    main()
