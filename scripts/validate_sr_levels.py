"""
scripts/validate_sr_levels.py

Interactive HTML validation chart for the S/R level engine.
Filters to 2025-01-10 → end of available data.

Layout (4 rows, shared x-axis):
  Row 1  Price + all active levels + nearest S/R lines + break events
  Row 2  ATR-normalised distance to nearest support & resistance
  Row 3  Strength of nearest support & resistance  [0 = weak, 1 = strong]
  Row 4  Active level count over time

Support = blue lines / metrics.  Resistance = red lines / metrics.
Level line opacity encodes strength: stronger levels are darker.
Dotted lines = invalidated / retired levels.

Output: notebooks/foundation/sr_levels_validation_{instrument}_{timeframe}.html
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

from src.dag_runtime import GraphRunContext, execute_graph
from src.dag_runtime.builtin_graphs import get_builtin_graph
from src.indicators.foundation.sr_levels import (
    SR_SIDE_SUPPORT,
    SR_STATE_ACTIVE,
    SR_STATE_INVALIDATED,
    SR_STATE_RETIRED,
)
from src.validation.common import (
    cleanup_validation_artifacts,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATE_FROM = "2026-01-10"
OUT_DIR = Path("notebooks/foundation")
CACHE_ROOT = Path("data/validation_cache")
VALIDATOR_NAME = "validate_sr_levels"
RUNS = (("XAU_USD", "H4"),)

_SUP_SOLID = "rgba(30, 110, 255, 0.90)"
_RES_SOLID = "rgba(210, 30, 30, 0.90)"
_SUP_FILL = "rgba(30, 110, 255, 0.12)"
_RES_FILL = "rgba(210, 30, 30, 0.12)"
_SUP_TMPL = "rgba(30, 110, 255, {a})"
_RES_TMPL = "rgba(210, 30, 30, {a})"

MAX_LEVEL_LINES = 150  # cap to keep the chart fast


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------


def _build_sr_chart(
    enriched: pd.DataFrame,
    full_out: pd.DataFrame,
    registry: dict,
    *,
    title: str,
) -> go.Figure:
    """Build the 4-panel chart.

    Accepts already-computed `full_out` (output of add_sr_levels) and
    `registry` (after update_sr_lifecycle) to avoid re-running the pipeline.
    """

    # ── 2. Slice to plot window ────────────────────────────────────────────
    all_ts = pd.to_datetime(
        enriched["timestamp"] if "timestamp" in enriched.columns else enriched.index,
        utc=True,
        errors="coerce",
    )
    out_ts = pd.to_datetime(
        full_out["timestamp"] if "timestamp" in full_out.columns else full_out.index,
        utc=True,
        errors="coerce",
    )
    start_ts = pd.Timestamp(DATE_FROM, tz="UTC")
    plot = full_out[out_ts >= start_ts].copy()
    x = out_ts[out_ts >= start_ts]

    if plot.empty:
        raise ValueError(f"No data on or after {DATE_FROM}")

    plot_start = x.iloc[0]
    plot_end = x.iloc[-1]

    # ── 3. Figure skeleton ─────────────────────────────────────────────────
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.48, 0.17, 0.17, 0.18],
        subplot_titles=[
            (
                "Price — Support <span style='color:rgb(30,110,255)'>■</span> blue  "
                "| Resistance <span style='color:rgb(210,30,30)'>■</span> red  "
                "| opacity = level strength  | dotted = invalidated"
            ),
            "Distance to Nearest Level  (ATR units — 0.15 ATR = touch zone)",
            "Strength of Nearest Level  (0 = weak / fading,  1 = strong / fresh)",
            "Active Level Count  (how many live levels exist at each bar)",
        ],
    )

    # ── 4. Row 1: Candlestick ──────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=plot["open"],
            high=plot["high"],
            low=plot["low"],
            close=plot["close"],
            name="OHLC",
            increasing_line_color="rgba(34,139,34,0.9)",
            decreasing_line_color="rgba(180,0,0,0.9)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # ── 5. Level lines ─────────────────────────────────────────────────────
    # Filter to levels that overlap the plot window (active or invalidated within it)
    visible: list = []
    for lev in registry.values():
        # Determine line span
        if lev.source_live_from_idx >= len(all_ts):
            continue
        lev_start = all_ts.iloc[lev.source_live_from_idx]

        if lev.state == SR_STATE_INVALIDATED and lev.invalidation_idx >= 0:
            end_i = min(lev.invalidation_idx, len(all_ts) - 1)
            lev_end = all_ts.iloc[end_i]
        elif lev.state == SR_STATE_RETIRED and lev.retirement_idx >= 0:
            end_i = min(lev.retirement_idx, len(all_ts) - 1)
            lev_end = all_ts.iloc[end_i]
        else:
            lev_end = plot_end

        # Skip levels that ended before the window
        if lev_end < plot_start:
            continue
        visible.append((lev, lev_start, lev_end))

    # Sort strongest first so strong levels are drawn on top
    visible.sort(key=lambda t: t[0].level_strength, reverse=True)

    for lev, lev_start, lev_end in visible[:MAX_LEVEL_LINES]:
        x0 = max(lev_start, plot_start)
        x1 = lev_end

        strength = float(np.clip(lev.level_strength, 0.0, 1.0))
        alpha = round(0.18 + 0.65 * strength, 2)
        is_sup = lev.side == SR_SIDE_SUPPORT
        color = (_SUP_TMPL if is_sup else _RES_TMPL).format(a=alpha)
        is_dead = lev.state in (SR_STATE_INVALIDATED, SR_STATE_RETIRED)
        dash = "dot" if is_dead else "solid"

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[lev.level_price, lev.level_price],
                mode="lines",
                line=dict(color=color, width=1.2, dash=dash),
                showlegend=False,
                hovertemplate=(
                    f"<b>{'Support' if is_sup else 'Resistance'}</b> [{lev.source_family}]<br>"
                    f"price = {lev.level_price:.4f}<br>"
                    f"strength = {lev.level_strength:.3f}<br>"
                    f"touches = {lev.touch_count}  |  weakens = {lev.weaken_count}<br>"
                    f"state = {'active' if lev.state == SR_STATE_ACTIVE else 'invalidated/retired'}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # ── 6. Nearest S/R tracking lines (thick, dashed) ─────────────────────
    for col, color, name in (
        ("nearest_support_price", _SUP_SOLID, "Nearest Support"),
        ("nearest_resistance_price", _RES_SOLID, "Nearest Resistance"),
    ):
        if col not in plot.columns:
            continue
        s = pd.to_numeric(plot[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=s,
                mode="lines",
                name=name,
                line=dict(color=color, width=2.5, dash="dash"),
                connectgaps=False,
            ),
            row=1,
            col=1,
        )

    # ── 7. Break-of-level markers ──────────────────────────────────────────
    for col, color, symbol, name in (
        ("support_broken_this_bar", _SUP_SOLID, "triangle-down", "Support Break ▼"),
        ("resistance_broken_this_bar", _RES_SOLID, "triangle-up", "Resistance Break ▲"),
    ):
        if col not in plot.columns:
            continue
        mask = plot[col].astype(int) == 1
        if not mask.any():
            continue
        sub = plot[mask]
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=sub["close"],
                mode="markers",
                name=name,
                marker=dict(
                    symbol=symbol,
                    size=12,
                    color=color,
                    line=dict(color="white", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # ── 8. Row 2: ATR-normalised distance ──────────────────────────────────
    for col, color, name in (
        ("nearest_support_distance_atr", _SUP_SOLID, "← Support distance (ATR)"),
        ("nearest_resistance_distance_atr", _RES_SOLID, "← Resistance distance (ATR)"),
    ):
        if col not in plot.columns:
            continue
        s = pd.to_numeric(plot[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=s,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.8),
                connectgaps=False,
            ),
            row=2,
            col=1,
        )

    # Touch-zone threshold reference
    fig.add_hline(
        y=0.15,
        line_dash="dot",
        line_color="rgba(100,100,100,0.45)",
        annotation_text="touch zone (0.15 ATR)",
        annotation_position="bottom right",
        row=2,
        col=1,
    )

    # ── 9. Row 3: Strength of nearest S/R ─────────────────────────────────
    for col, color, name in (
        ("nearest_support_strength", _SUP_SOLID, "Support strength"),
        ("nearest_resistance_strength", _RES_SOLID, "Resistance strength"),
    ):
        if col not in plot.columns:
            continue
        s = pd.to_numeric(plot[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=s,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.8),
                connectgaps=False,
            ),
            row=3,
            col=1,
        )

    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="rgba(100,100,100,0.45)",
        annotation_text="mid-strength (0.5)",
        annotation_position="bottom right",
        row=3,
        col=1,
    )

    # ── 10. Row 4: Active level counts ─────────────────────────────────────
    for col, color, fill, name in (
        ("active_support_count", _SUP_SOLID, _SUP_FILL, "Active supports"),
        ("active_resistance_count", _RES_SOLID, _RES_FILL, "Active resistances"),
    ):
        if col not in plot.columns:
            continue
        s = pd.to_numeric(plot[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=s,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=fill,
            ),
            row=4,
            col=1,
        )

    # ── 11. Layout ──────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        template="plotly_white",
        height=1450,
        hovermode="x unified",
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
    fig.update_yaxes(title_text="ATR units", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="Strength [0–1]", row=3, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="# Levels", row=4, col=1, rangemode="tozero")

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
        description="Validate S/R levels and optionally produce an HTML chart."
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate the HTML chart. Default run prints summary only.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--invalidate-cache", action="store_true")
    parser.add_argument("--cleanup-stale", action="store_true")
    parser.add_argument("--max-artifact-age-days", type=int, default=30)
    args = parser.parse_args()
    include_chart = args.html

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cleanup_stale:
        removed = cleanup_validation_artifacts(
            cache_root=CACHE_ROOT,
            max_age_days=args.max_artifact_age_days,
            report_roots=[OUT_DIR],
        )
        print(f"cleanup_removed: {len(removed)}")

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        if not data_file.exists():
            print(f"[SKIP] {instrument} {timeframe}: {data_file} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {instrument} {timeframe}")
        print(f"{'=' * 60}")

        raw = pd.read_parquet(data_file)
        graph = get_builtin_graph(
            "validate_sr_levels",
            instrument=instrument,
            timeframe=timeframe,
        )
        context = GraphRunContext(
            graph_name=graph.graph_name,
            symbol=instrument,
            timeframe=timeframe,
            inputs={"raw_input": raw},
            config={"html": include_chart, "out_dir": str(OUT_DIR)},
            cache_root=CACHE_ROOT,
            force=args.force,
            invalidate_cache=args.invalidate_cache,
        )
        graph_result = execute_graph(
            graph, context=context, target="sr_validation_bundle"
        )
        result = graph_result.output().payload

        print(f"  Total bars: {result['row_count']}")
        print()
        _print_summary(result["summary"])

        if include_chart and result["html_path"] is not None:
            print(f"\n  Chart saved → {result['html_path']}")
        else:
            print("\n  [chart skipped — pass --html to generate HTML]")
        profile_path = (
            CACHE_ROOT / VALIDATOR_NAME / instrument / timeframe / "run-summary.json"
        )
        graph_result.profiler.write_json(profile_path)
        print(f"  Profiler summary → {profile_path}")


if __name__ == "__main__":
    main()
