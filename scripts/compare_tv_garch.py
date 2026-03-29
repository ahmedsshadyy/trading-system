"""
scripts/compare_tv_garch.py

Standalone TradingView GARCH-proxy comparison script.

Runs the canonical GARCH vs TradingView 'GARCH Volatility Estimation'
benchmark comparison and prints a structured report.

DOCTRINAL NOTE
--------------
The TradingView proxy is NOT return-based GARCH(1,1).  It is a recursive
variance of (close − EMA(close)) with fixed alpha/beta.  Mismatch is
expected and NOT an implementation error.  Compare rank behavior and
high-vol episode overlap, not raw values.

Usage
-----
    python scripts/compare_tv_garch.py
    python scripts/compare_tv_garch.py --html    # also write HTML chart
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.indicators.foundation.volatility import (
    add_volatility_features,
    compute_tv_garch_proxy,
    compute_tv_hist_vol,
)
from src.validation.common.reporting import get_logger, load_windowed_parquet

OUT_DIR = Path("notebooks/foundation")
RUNS = (("XAU_USD", "H4"),)


def _pct_rank(s: pd.Series, window: int = 100) -> pd.Series:
    return s.rolling(window, min_periods=window).apply(
        lambda w: float(np.count_nonzero(w <= w[-1]) / len(w)), raw=True
    )


def _compare_pair(a: pd.Series, b: pd.Series) -> dict:
    from scipy.stats import spearmanr

    both = a.notna() & b.notna()
    if both.sum() < 10:
        return {"n": int(both.sum()), "available": False}
    av, bv = a[both].to_numpy(float), b[both].to_numpy(float)
    pearson = float(np.corrcoef(av, bv)[0, 1])
    spearman = float(spearmanr(av, bv).correlation)
    mae = float(np.mean(np.abs(av - bv)))
    rmse = float(np.sqrt(np.mean((av - bv) ** 2)))
    ratio = av / np.where(bv > 0, bv, np.nan)
    ap90, bp90 = np.percentile(av, 90), np.percentile(bv, 90)
    ap10, bp10 = np.percentile(av, 10), np.percentile(bv, 10)
    return {
        "n": int(both.sum()),
        "pearson_corr": round(pearson, 4),
        "spearman_corr": round(spearman, 4),
        "mae": round(mae, 8),
        "rmse": round(rmse, 8),
        "mean_ratio": round(float(np.nanmean(ratio)), 4),
        "median_ratio": round(float(np.nanmedian(ratio)), 4),
        "top_decile_overlap": round(float(np.mean((av >= ap90) & (bv >= bp90))), 4),
        "bottom_decile_overlap": round(float(np.mean((av <= ap10) & (bv <= bp10))), 4),
    }


def _shock_response(
    df: pd.DataFrame, series_dict: dict[str, pd.Series], top_n: int = 50
) -> dict:
    close = pd.to_numeric(df["close"], errors="coerce")
    log_ret_abs = np.log(close / close.shift(1)).abs()
    shock_idx = log_ret_abs.nlargest(top_n).index
    n = len(df)
    offsets = (-1, 0, 1, 3, 5)
    results = {}
    for name, series in series_dict.items():
        arr = series.to_numpy(dtype=float)
        offset_means = {}
        for off in offsets:
            vals = []
            for si in shock_idx:
                if si not in df.index:
                    continue
                pos = df.index.get_loc(si)
                tgt = pos + off
                if 0 <= tgt < n and np.isfinite(arr[tgt]):
                    vals.append(arr[tgt])
            offset_means[off] = round(float(np.mean(vals)), 8) if vals else None
        t0, tm1, t3 = offset_means.get(0), offset_means.get(-1), offset_means.get(3)
        jump = round((t0 - tm1) / tm1, 4) if (t0 and tm1 and tm1 > 0) else None
        pers = round((t3 - t0) / t0, 4) if (t3 and t0 and t0 > 0) else None
        results[name] = {
            "avg_at": {str(k): v for k, v in offset_means.items()},
            "jump": jump,
            "persistence_t3": pers,
        }
    return results


def _build_comparison_chart(
    df: pd.DataFrame,
    tv_proxy: pd.Series,
    tv_hv: pd.Series,
    rv20_tv: pd.Series,
    garch: pd.Series,
    *,
    title: str,
    date_from: str = "2026-01-10",
) -> go.Figure:
    ts = pd.to_datetime(
        df["timestamp"] if "timestamp" in df.columns else df.index,
        utc=True,
        errors="coerce",
    )
    mask = ts >= pd.Timestamp(date_from, tz="UTC")
    df_p = df[mask]
    x = ts[mask]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.40, 0.30, 0.30],
        subplot_titles=[
            "Price",
            "GARCH Vol comparison  |  canonical (red) vs TV proxy (blue, EMA-deviation based)",
            "Realized Vol comparison  |  TV hist vol % (purple) vs RV20 TV-scaled % (green)",
        ],
    )

    if all(c in df_p.columns for c in ("open", "high", "low", "close")):
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=df_p["open"],
                high=df_p["high"],
                low=df_p["low"],
                close=df_p["close"],
                name="OHLC",
                showlegend=False,
                increasing_line_color="rgba(34,139,34,0.9)",
                decreasing_line_color="rgba(180,0,0,0.9)",
            ),
            row=1,
            col=1,
        )

    for series, color, name, row in (
        (garch[mask], "rgba(210,40,40,0.9)", "GARCH canonical", 2),
        (tv_proxy[mask], "rgba(30,100,220,0.8)", "TV GARCH proxy", 2),
        (tv_hv[mask], "rgba(150,0,200,0.85)", "TV hist vol %", 3),
        (rv20_tv[mask], "rgba(0,155,80,0.85)", "RV20 TV-scaled %", 3),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=series,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.8),
                connectgaps=False,
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        height=1100,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1),
    )
    for i in range(1, 4):
        ax = "xaxis" if i == 1 else f"xaxis{i}"
        fig.update_layout(**{ax: {"rangeslider": {"visible": False}}})
    return fig


def _print_dict(d: object, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                _print_dict(v, indent + 2)
            else:
                print(f"{prefix}{k}: {v}")
    else:
        print(f"{prefix}{d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TradingView GARCH comparison audit.")
    parser.add_argument(
        "--html", action="store_true", help="Write HTML comparison chart."
    )
    parser.add_argument("--full", action="store_true", help="Use full dataset.")
    parser.add_argument("--tail-rows", type=int, default=2000)
    args = parser.parse_args()
    logger = get_logger("compare_tv_garch")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        if not data_file.exists():
            logger.warning("skip %s %s: not found", instrument, timeframe)
            continue

        logger.info("running %s %s", instrument, timeframe)
        raw = load_windowed_parquet(data_file, full=args.full, tail_rows=args.tail_rows)
        df = add_volatility_features(raw, include_research_only=True)

        garch = pd.to_numeric(df.get("garch_vol_1_1"), errors="coerce")
        tv_proxy = compute_tv_garch_proxy(df)
        tv_hv = compute_tv_hist_vol(df)
        rv20_raw = (
            pd.to_numeric(df.get("rv_close_20"), errors="coerce")
            if "rv_close_20" in df.columns
            else None
        )
        rv20_tv = (
            rv20_raw * 100.0 * float(np.sqrt(365))
            if rv20_raw is not None
            else pd.Series(dtype=float)
        )

        print(f"\n{'='*60}\n  {instrument} {timeframe}\n{'='*60}")
        print("\n--- DOCTRINAL NOTE ---")
        print(
            "TV GARCH proxy = recursive (close − EMA)^2 variance, NOT return-based GARCH(1,1)."
        )
        print("Mismatch is expected. Focus on rank correlation and episode overlap.\n")

        print("--- Family A: canonical GARCH vs TV GARCH proxy ---")
        _print_dict(_compare_pair(garch, tv_proxy))

        print("\n--- Percentile-rank comparison: GARCH vs TV proxy ---")
        gr = _pct_rank(garch)
        tr = _pct_rank(tv_proxy)
        _print_dict(_compare_pair(gr, tr))

        if rv20_tv is not None and rv20_tv.notna().sum() > 0:
            print("\n--- Family B: TV hist vol vs RV20 TV-scaled ---")
            _print_dict(_compare_pair(tv_hv, rv20_tv))

        print("\n--- Shock response (top 50 absolute-return bars) ---")
        series_for_shock = {
            "garch_vol_1_1": garch,
            "tv_garch_proxy": tv_proxy,
        }
        if rv20_tv is not None:
            series_for_shock["rv20_tv_scaled"] = rv20_tv
            series_for_shock["tv_hist_vol"] = tv_hv
        _print_dict(_shock_response(df, series_for_shock))

        if args.html:
            title = f"TV GARCH Comparison — {instrument} {timeframe}"
            fig = _build_comparison_chart(
                df, tv_proxy, tv_hv, rv20_tv, garch, title=title
            )
            html_path = OUT_DIR / f"tv_garch_comparison_{instrument}_{timeframe}.html"
            fig.write_html(str(html_path))
            logger.info("chart saved -> %s", html_path)


if __name__ == "__main__":
    main()
