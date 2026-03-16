"""
Per-detector validation charts.

Generates one clean chart per detector type — just candlesticks + that
one detection. Zoomed to a readable window for candle-by-candle comparison.

Usage: poetry run python scripts/validate_detectors.py
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


def load_candles(instrument, timeframe, engine):
    query = f"""
        SELECT timestamp, open, high, low, close, volume, spread
        FROM candles
        WHERE instrument = '{instrument}' AND timeframe = '{timeframe}'
        ORDER BY timestamp ASC
    """
    return pd.read_sql(query, engine)


def base_candle_chart(seg, title):
    """Create a plain candlestick chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=seg["timestamp"],
            open=seg["open"],
            high=seg["high"],
            low=seg["low"],
            close=seg["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=seg["timestamp"],
            y=seg["ema_20"],
            name="EMA 20",
            line=dict(color="rgba(100,149,237,0.5)", width=1),
        )
    )
    fig.update_layout(
        height=600,
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        legend=dict(font=dict(size=10), x=0.01, y=0.99),
    )
    return fig


def chart_swings(seg):
    """Chart 1: Swing Highs and Lows only."""
    fig = base_candle_chart(seg, "Detector: Swing Highs (▼) & Lows (▲)")

    sh = seg[seg["swing_high"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sh["timestamp"],
            y=sh["high"],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=12, color="red"),
            text=[f"{v:.0f}" for v in sh["high"]],
            textposition="top center",
            textfont=dict(size=9, color="red"),
            name="Swing High",
        )
    )

    sl = seg[seg["swing_low"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sl["timestamp"],
            y=sl["low"],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=12, color="lime"),
            text=[f"{v:.0f}" for v in sl["low"]],
            textposition="bottom center",
            textfont=dict(size=9, color="lime"),
            name="Swing Low",
        )
    )
    return fig


def chart_bos(seg):
    """Chart 2: BOS only — with the broken level drawn as a line."""
    fig = base_candle_chart(seg, "Detector: Break of Structure (BOS)")

    # Draw last_swing_high / low as stepped lines
    fig.add_trace(
        go.Scatter(
            x=seg["timestamp"],
            y=seg["last_swing_high"],
            name="Last Swing High",
            line=dict(color="rgba(255,100,100,0.4)", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=seg["timestamp"],
            y=seg["last_swing_low"],
            name="Last Swing Low",
            line=dict(color="rgba(100,255,100,0.4)", width=1, dash="dot"),
        )
    )

    bos_b = seg[seg["bos_bull"] == 1]
    fig.add_trace(
        go.Scatter(
            x=bos_b["timestamp"],
            y=bos_b["close"],
            mode="markers+text",
            marker=dict(symbol="star", size=14, color="lime"),
            text=["BOS↑"] * len(bos_b),
            textposition="top center",
            textfont=dict(size=10, color="lime"),
            name="BOS Bull",
        )
    )

    bos_r = seg[seg["bos_bear"] == 1]
    fig.add_trace(
        go.Scatter(
            x=bos_r["timestamp"],
            y=bos_r["close"],
            mode="markers+text",
            marker=dict(symbol="star", size=14, color="magenta"),
            text=["BOS↓"] * len(bos_r),
            textposition="bottom center",
            textfont=dict(size=10, color="magenta"),
            name="BOS Bear",
        )
    )
    return fig


def chart_choch(seg):
    """Chart 3: CHoCH only — with trend state background."""
    fig = base_candle_chart(seg, "Detector: Change of Character (CHoCH)")

    # Color background by trend state
    for i in range(len(seg) - 1):
        t = seg["trend_state"].iloc[i]
        if t == 1:
            color = "rgba(0,200,0,0.07)"
        elif t == -1:
            color = "rgba(200,0,0,0.07)"
        else:
            continue
        fig.add_vrect(
            x0=seg["timestamp"].iloc[i],
            x1=seg["timestamp"].iloc[i + 1],
            fillcolor=color,
            layer="below",
            line_width=0,
        )

    ch_b = seg[seg["choch_bull"] == 1]
    fig.add_trace(
        go.Scatter(
            x=ch_b["timestamp"],
            y=ch_b["close"],
            mode="markers+text",
            marker=dict(symbol="diamond", size=14, color="cyan"),
            text=["CHoCH↑"] * len(ch_b),
            textposition="top center",
            textfont=dict(size=10, color="cyan"),
            name="CHoCH Bull",
        )
    )

    ch_r = seg[seg["choch_bear"] == 1]
    fig.add_trace(
        go.Scatter(
            x=ch_r["timestamp"],
            y=ch_r["close"],
            mode="markers+text",
            marker=dict(symbol="diamond", size=14, color="yellow"),
            text=["CHoCH↓"] * len(ch_r),
            textposition="bottom center",
            textfont=dict(size=10, color="yellow"),
            name="CHoCH Bear",
        )
    )

    fig.add_annotation(
        text="Green bg = bullish trend, Red bg = bearish trend",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )
    return fig


def chart_fvg(seg):
    """Chart 4: FVG zones + IFVG markers."""
    fig = base_candle_chart(seg, "Detector: FVG Zones + IFVG Confirmations")

    for _, row in seg[seg["fvg_bull"] == 1].iterrows():
        if not np.isnan(row.get("fvg_bull_low", np.nan)):
            fig.add_shape(
                type="rect",
                x0=row["timestamp"],
                x1=row["timestamp"] + pd.Timedelta(hours=20),
                y0=row["fvg_bull_low"],
                y1=row["fvg_bull_high"],
                fillcolor="rgba(0,200,0,0.2)",
                line=dict(color="lime", width=1),
            )
            fig.add_annotation(
                x=row["timestamp"],
                y=row["fvg_bull_high"],
                text=f"B {row['fvg_size_atr']:.1f}x",
                showarrow=False,
                font=dict(size=8, color="lime"),
                yshift=8,
            )

    for _, row in seg[seg["fvg_bear"] == 1].iterrows():
        if not np.isnan(row.get("fvg_bear_low", np.nan)):
            fig.add_shape(
                type="rect",
                x0=row["timestamp"],
                x1=row["timestamp"] + pd.Timedelta(hours=20),
                y0=row["fvg_bear_low"],
                y1=row["fvg_bear_high"],
                fillcolor="rgba(200,0,0,0.2)",
                line=dict(color="red", width=1),
            )
            fig.add_annotation(
                x=row["timestamp"],
                y=row["fvg_bear_low"],
                text=f"S {row['fvg_size_atr']:.1f}x",
                showarrow=False,
                font=dict(size=8, color="red"),
                yshift=-8,
            )

    # IFVG markers
    if "ifvg_bull" in seg.columns:
        ib = seg[seg["ifvg_bull"] == 1]
        if len(ib):
            fig.add_trace(
                go.Scatter(
                    x=ib["timestamp"],
                    y=ib["close"],
                    mode="markers+text",
                    marker=dict(symbol="diamond", size=14, color="cyan"),
                    text=["IFVG↑"] * len(ib),
                    textposition="top center",
                    textfont=dict(size=10, color="cyan"),
                    name="IFVG Bull",
                )
            )

    if "ifvg_bear" in seg.columns:
        ir = seg[seg["ifvg_bear"] == 1]
        if len(ir):
            fig.add_trace(
                go.Scatter(
                    x=ir["timestamp"],
                    y=ir["close"],
                    mode="markers+text",
                    marker=dict(symbol="diamond", size=14, color="yellow"),
                    text=["IFVG↓"] * len(ir),
                    textposition="bottom center",
                    textfont=dict(size=10, color="yellow"),
                    name="IFVG Bear",
                )
            )

    return fig


def chart_ob(seg):
    """Chart 5: Order Blocks only."""
    fig = base_candle_chart(seg, "Detector: Order Blocks (OB)")

    for _, row in seg[seg["ob_bull"] == 1].iterrows():
        if not np.isnan(row.get("ob_bull_low", np.nan)):
            fig.add_shape(
                type="rect",
                x0=row["timestamp"],
                x1=row["timestamp"] + pd.Timedelta(hours=32),
                y0=row["ob_bull_low"],
                y1=row["ob_bull_high"],
                fillcolor="rgba(0,100,255,0.2)",
                line=dict(color="dodgerblue", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=row["timestamp"],
                y=row["ob_bull_low"],
                text="Bull OB",
                showarrow=False,
                font=dict(size=9, color="dodgerblue"),
                yshift=-12,
            )

    for _, row in seg[seg["ob_bear"] == 1].iterrows():
        if not np.isnan(row.get("ob_bear_low", np.nan)):
            fig.add_shape(
                type="rect",
                x0=row["timestamp"],
                x1=row["timestamp"] + pd.Timedelta(hours=32),
                y0=row["ob_bear_low"],
                y1=row["ob_bear_high"],
                fillcolor="rgba(255,100,0,0.2)",
                line=dict(color="orange", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=row["timestamp"],
                y=row["ob_bear_high"],
                text="Bear OB",
                showarrow=False,
                font=dict(size=9, color="orange"),
                yshift=12,
            )
    return fig


def chart_sweeps(seg):
    """Chart 6: Liquidity Sweeps only."""
    fig = base_candle_chart(seg, "Detector: Liquidity Sweeps")

    fig.add_trace(
        go.Scatter(
            x=seg["timestamp"],
            y=seg["last_swing_high"],
            name="Last Swing High",
            line=dict(color="rgba(255,100,100,0.3)", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=seg["timestamp"],
            y=seg["last_swing_low"],
            name="Last Swing Low",
            line=dict(color="rgba(100,255,100,0.3)", width=1, dash="dot"),
        )
    )

    sw_h = seg[seg["sweep_high"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sw_h["timestamp"],
            y=sw_h["high"],
            mode="markers+text",
            marker=dict(symbol="x", size=16, color="red", line=dict(width=3)),
            text=[f"Sweep H\n{m:.1f}ATR" for m in sw_h["sweep_magnitude"]],
            textposition="top center",
            textfont=dict(size=9, color="red"),
            name="Sweep High",
        )
    )

    sw_l = seg[seg["sweep_low"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sw_l["timestamp"],
            y=sw_l["low"],
            mode="markers+text",
            marker=dict(symbol="x", size=16, color="lime", line=dict(width=3)),
            text=[f"Sweep L\n{m:.1f}ATR" for m in sw_l["sweep_magnitude"]],
            textposition="bottom center",
            textfont=dict(size=9, color="lime"),
            name="Sweep Low",
        )
    )
    return fig


def chart_displacement(seg):
    """Chart 7: Displacement candles only."""
    fig = base_candle_chart(seg, "Detector: Displacement Candles (body ≥ 1.5× ATR)")

    disp = seg[seg["displacement_candle"] == 1]
    fig.add_trace(
        go.Scatter(
            x=disp["timestamp"],
            y=disp["close"],
            mode="markers+text",
            marker=dict(
                symbol="square",
                size=10,
                color="white",
                line=dict(width=2, color="yellow"),
            ),
            text=[f"{r:.1f}x" for r in disp["displacement_body_atr"]],
            textposition="top center",
            textfont=dict(size=9, color="yellow"),
            name="Displacement",
        )
    )
    return fig


def chart_equal_hl(seg):
    """Chart 8: Equal Highs/Lows only."""
    fig = base_candle_chart(seg, "Detector: Equal Highs (---) & Equal Lows (---)")

    # Draw horizontal lines at equal high/low levels
    eq_h = seg[seg["equal_highs"] == 1]
    for _, row in eq_h.iterrows():
        fig.add_shape(
            type="line",
            x0=row["timestamp"] - pd.Timedelta(hours=16),
            x1=row["timestamp"] + pd.Timedelta(hours=16),
            y0=row["swing_high_price"],
            y1=row["swing_high_price"],
            line=dict(color="red", width=2, dash="dash"),
        )
    if len(eq_h):
        fig.add_trace(
            go.Scatter(
                x=eq_h["timestamp"],
                y=eq_h["swing_high_price"],
                mode="markers",
                marker=dict(symbol="line-ew", size=12, color="red"),
                name=f"Equal Highs ({len(eq_h)})",
            )
        )

    eq_l = seg[seg["equal_lows"] == 1]
    for _, row in eq_l.iterrows():
        fig.add_shape(
            type="line",
            x0=row["timestamp"] - pd.Timedelta(hours=16),
            x1=row["timestamp"] + pd.Timedelta(hours=16),
            y0=row["swing_low_price"],
            y1=row["swing_low_price"],
            line=dict(color="lime", width=2, dash="dash"),
        )
    if len(eq_l):
        fig.add_trace(
            go.Scatter(
                x=eq_l["timestamp"],
                y=eq_l["swing_low_price"],
                mode="markers",
                marker=dict(symbol="line-ew", size=12, color="lime"),
                name=f"Equal Lows ({len(eq_l)})",
            )
        )
    return fig


def chart_rsi_div(seg):
    """Chart 9: RSI Divergence only — price + RSI subplot."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=["Price — RSI Divergence", "RSI 14"],
    )

    fig.add_trace(
        go.Candlestick(
            x=seg["timestamp"],
            open=seg["open"],
            high=seg["high"],
            low=seg["low"],
            close=seg["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    div_bear = seg[seg["rsi_div_bearish"] == 1]
    fig.add_trace(
        go.Scatter(
            x=div_bear["timestamp"],
            y=div_bear["high"],
            mode="markers+text",
            text=["Bear Div"] * len(div_bear),
            textposition="top center",
            textfont=dict(size=10, color="red"),
            marker=dict(symbol="triangle-down", size=12, color="red"),
            name="Bearish Divergence",
        ),
        row=1,
        col=1,
    )

    div_bull = seg[seg["rsi_div_bullish"] == 1]
    fig.add_trace(
        go.Scatter(
            x=div_bull["timestamp"],
            y=div_bull["low"],
            mode="markers+text",
            text=["Bull Div"] * len(div_bull),
            textposition="bottom center",
            textfont=dict(size=10, color="lime"),
            marker=dict(symbol="triangle-up", size=12, color="lime"),
            name="Bullish Divergence",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=seg["timestamp"],
            y=seg["rsi_14"],
            name="RSI",
            line=dict(color="purple", width=1),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.4, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.4, row=2, col=1)

    # Mark RSI values at divergence points
    fig.add_trace(
        go.Scatter(
            x=div_bear["timestamp"],
            y=div_bear["rsi_14"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="red"),
            name="RSI at Bear Div",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=div_bull["timestamp"],
            y=div_bull["rsi_14"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="lime"),
            name="RSI at Bull Div",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(font=dict(size=10)),
    )
    return fig


def main():
    engine = create_engine(os.getenv("DATABASE_URL"))

    print("Loading XAU_USD H4...")
    df = load_candles("XAU_USD", "H4", engine)
    print(f"  {len(df):,} candles")

    print("Running indicators...")
    df = build_all_indicators(df, instrument="XAU_USD", include_vp=True)
    print(f"  {len(df.columns)} columns")

    # Use a ~6 week window — enough to see structure but not overwhelming
    start = pd.Timestamp("2026-02-01", tz="UTC")
    end = pd.Timestamp("2026-03-14", tz="UTC")
    seg = (
        df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        .copy()
        .reset_index(drop=True)
    )
    print(f"  Segment: {len(seg)} candles ({start.date()} to {end.date()})")

    output_dir = Path(__file__).parent.parent / "notebooks"
    output_dir.mkdir(exist_ok=True)

    charts = [
        ("01_swings", chart_swings),
        ("02_bos", chart_bos),
        ("03_choch", chart_choch),
        ("04_fvg_ifvg", chart_fvg),
        ("05_ob", chart_ob),
        ("06_sweeps", chart_sweeps),
        ("07_displacement", chart_displacement),
        ("08_equal_hl", chart_equal_hl),
        ("09_rsi_divergence", chart_rsi_div),
    ]

    for name, chart_fn in charts:
        print(f"  Building {name}...")
        fig = chart_fn(seg)
        fname = output_dir / f"detect_{name}.html"
        fig.write_html(str(fname))
        print(f"    Saved: {fname}")

    print(f"\nDone — {len(charts)} charts in {output_dir}/")
    print(
        "Open each in browser and compare against XAU/USD H4 on TradingView (Feb 1 - Mar 14, 2026)"
    )


if __name__ == "__main__":
    main()
