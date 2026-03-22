"""
scripts/compare_swing_configs.py

Compare swing detector parameter configurations side by side.
Prints summary stats and generates one validation chart per config.

Usage: poetry run python scripts/compare_swing_configs.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.indicators.foundation.volatility import add_atr
from src.indicators.structure.swings import add_swings

import plotly.graph_objects as go


def load_candles(instrument, timeframe, engine):
    query = f"""
        SELECT timestamp, open, high, low, close, volume, spread
        FROM candles
        WHERE instrument = '{instrument}' AND timeframe = '{timeframe}'
        ORDER BY timestamp ASC
    """
    return pd.read_sql(query, engine)


def summarize_swings(df, label):
    n = len(df)
    sh = df["swing_high"].sum()
    sl = df["swing_low"].sum()

    sh_bars = df[df["swing_high"] == 1].index.to_series().diff().dropna()
    sl_bars = df[df["swing_low"] == 1].index.to_series().diff().dropna()

    prom_h = df.loc[df["swing_high"] == 1, "swing_high_prominence_atr"].dropna()
    prom_l = df.loc[df["swing_low"] == 1, "swing_low_prominence_atr"].dropna()

    return {
        "config": label,
        "swing_highs": int(sh),
        "swing_lows": int(sl),
        "total_swings": int(sh + sl),
        "sh_rate_%": round(sh / n * 100, 2),
        "sl_rate_%": round(sl / n * 100, 2),
        "avg_sh_spacing": round(sh_bars.mean(), 1) if len(sh_bars) > 0 else None,
        "avg_sl_spacing": round(sl_bars.mean(), 1) if len(sl_bars) > 0 else None,
        "med_sh_spacing": round(sh_bars.median(), 1) if len(sh_bars) > 0 else None,
        "med_sl_spacing": round(sl_bars.median(), 1) if len(sl_bars) > 0 else None,
        "avg_sh_prom_atr": round(prom_h.mean(), 3) if len(prom_h) > 0 else None,
        "avg_sl_prom_atr": round(prom_l.mean(), 3) if len(prom_l) > 0 else None,
        "med_sh_prom_atr": round(prom_h.median(), 3) if len(prom_h) > 0 else None,
        "med_sl_prom_atr": round(prom_l.median(), 3) if len(prom_l) > 0 else None,
    }


def chart_swings(seg, title, fname):
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

    sh = seg[seg["swing_high"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sh["timestamp"],
            y=sh["high"],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=10, color="red"),
            text=[f"{v:.0f}" for v in sh["high"]],
            textposition="top center",
            textfont=dict(size=8, color="red"),
            name=f"SH ({len(sh)})",
        )
    )

    sl = seg[seg["swing_low"] == 1]
    fig.add_trace(
        go.Scatter(
            x=sl["timestamp"],
            y=sl["low"],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=10, color="lime"),
            text=[f"{v:.0f}" for v in sl["low"]],
            textposition="bottom center",
            textfont=dict(size=8, color="lime"),
            name=f"SL ({len(sl)})",
        )
    )

    fig.update_layout(
        height=600,
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
    )
    fig.write_html(str(fname))


def main():
    engine = create_engine(os.getenv("DATABASE_URL"))

    print("Loading XAU_USD H4...")
    df = load_candles("XAU_USD", "H4", engine)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)
    print(f"  {len(df):,} candles")

    df = add_atr(df)

    configs = [
        {"window": 4, "min_retrace_atr": 0.3},
        {"window": 4, "min_retrace_atr": 0.5},
        {"window": 4, "min_retrace_atr": 0.7},
        {"window": 4, "min_retrace_atr": 1.0},
        {"window": 6, "min_retrace_atr": 0.3},  # current — too noisy
        {"window": 6, "min_retrace_atr": 0.5},
        {"window": 6, "min_retrace_atr": 0.7},
        {"window": 6, "min_retrace_atr": 1.0},
        {"window": 8, "min_retrace_atr": 0.3},
        {"window": 8, "min_retrace_atr": 0.5},
        {"window": 8, "min_retrace_atr": 0.7},
    ]

    # Zoom window for charts
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-03-14", tz="UTC")

    output_dir = Path(__file__).parent.parent / "notebooks" / "swing_configs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for cfg in configs:
        label = f"w{cfg['window']}_ret{cfg['min_retrace_atr']}"
        print(f"\n  Config: {label}")

        result = add_swings(df, **cfg)
        stats = summarize_swings(result, label)
        results.append(stats)

        for k, v in stats.items():
            if k != "config":
                print(f"    {k}: {v}")

        # Chart zoomed segment
        seg = result[
            (result["timestamp"] >= start) & (result["timestamp"] <= end)
        ].copy()
        chart_swings(
            seg,
            f"Swings: {label} ({stats['swing_highs']}H/{stats['swing_lows']}L total)",
            output_dir / f"swings_{label}.html",
        )
        print(f"    Chart: {output_dir / f'swings_{label}.html'}")

    # Summary table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    summary.to_csv(output_dir / "swing_config_comparison.csv", index=False)
    print(f"\nSaved to {output_dir}/swing_config_comparison.csv")
    print(f"Charts in {output_dir}/")
    print("Open each chart and compare against TradingView XAU/USD H4 (Jan-Mar 2026)")


if __name__ == "__main__":
    main()
