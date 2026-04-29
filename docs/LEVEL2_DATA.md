# Level 2 / Order Flow Integration: NinjaTrader → Python Pipeline

This document is the complete engineering plan for streaming Level 2 (DOM),
volumetric footprint, and cumulative delta data from NinjaTrader 8 into the
existing Python indicator pipeline via a TCP socket bridge.

This is to be done at a later stage (MVP2 or smth) after I've tested it via Human in the loop

---

## Table of Contents

1. [What NinjaTrader Exposes](#1-what-ninjatrader-exposes)
2. [What the Current Pipeline Lacks](#2-what-the-current-pipeline-lacks)
3. [Architecture Overview](#3-architecture-overview)
4. [Instrument Mapping: Futures ↔ Spot](#4-instrument-mapping-futures--spot)
5. [Step 1 — NinjaScript TCP Server (C#)](#5-step-1--ninjascript-tcp-server-c)
6. [Step 2 — Python Socket Client & Ring Buffer](#6-step-2--python-socket-client--ring-buffer)
7. [Step 3 — Wire Into the Live Pipeline](#7-step-3--wire-into-the-live-pipeline)
8. [Step 4 — Order Flow Indicator Module](#8-step-4--order-flow-indicator-module)
9. [Step 5 — Upgrade Volume Profile With Real Footprint](#9-step-5--upgrade-volume-profile-with-real-footprint)
10. [Step 6 — Order Flow–Enhanced SMC Indicators](#10-step-6--order-flow-enhanced-smc-indicators)
11. [Step 7 — DAG Integration](#11-step-7--dag-integration)
12. [Step 8 — Validation & Parity Testing](#12-step-8--validation--parity-testing)
13. [Column Reference](#13-column-reference)
14. [Constraints & Limitations](#14-constraints--limitations)
15. [Execution Roadmap](#15-execution-roadmap)

---

## 1. What NinjaTrader Exposes

NinjaTrader 8 NinjaScript (C# 5.0 / .NET 4.8) provides programmatic access
to three tiers of order flow data:

### Tier 1: Level 2 / DOM (Depth of Market)

Via `OnMarketDepth(MarketDepthEventArgs e)`:

| Field | Type | Description |
|-------|------|-------------|
| `e.MarketDataType` | enum | `MarketDataType.Ask` or `MarketDataType.Bid` |
| `e.Price` | double | Price level |
| `e.Volume` | long | Resting limit order volume at that level |
| `e.Position` | int | Row index in the book (0 = best bid/ask) |
| `e.Operation` | enum | `Insert`, `Update`, `Remove` |

This is a per-update callback — each DOM change fires it. You reconstruct
the full book by accumulating updates.

### Tier 2: Volumetric Bars (Footprint)

Via `OrderFlowVolumetricBars` indicator (requires Lifetime/Lease license +
data provider with historical bid/ask ticks):

| Field | Type | Description |
|-------|------|-------------|
| Per price level: `Volumes[bar].BidVolume` | long | Volume transacted at the bid |
| Per price level: `Volumes[bar].AskVolume` | long | Volume transacted at the ask |
| Per price level: delta | long | `AskVolume - BidVolume` |
| Per bar: `Volumes[bar].TotalBuyingVolume` | long | Aggregate bar buy volume |
| Per bar: `Volumes[bar].TotalSellingVolume` | long | Aggregate bar sell volume |
| Per bar: cumulative delta | long | Running buy − sell from session start |

This gives you the actual traded volume at each price tick within a bar —
the raw material for footprint charts, delta profiles, and imbalance
detection.

### Tier 3: Cumulative Delta

Via `OrderFlowCumulativeDelta` indicator:

- Running total of (ask-transacted volume − bid-transacted volume) from
  session open.
- Available as a bar series: open/high/low/close of the delta within each
  bar.
- Requires same bid/ask tick data as volumetric bars.

### What Is NOT Exposed

| Data | Status |
|------|--------|
| Volume Profile Value Area (NT's built-in) | **Not in the API** — the indicator renders visually but does not expose POC/VAH/VAL programmatically. |
| VWAP | Exposed via `OrderFlowVWAP` — accessible in NinjaScript. |
| Raw tick stream | Accessible via `OnMarketData()` for last price/volume, but not directly as a tick replay API. |

**Key constraint**: Volumetric bar BidAsk delta classification requires
that the data provider supplies historical bid/ask tick data. This works
for futures (CME, NYMEX, COMEX). It does **not** work for Forex spot —
there is no centralized auction, so bid/ask attribution is undefined.

---

## 2. What the Current Pipeline Lacks

The existing pipeline uses tick-volume proxies for all order flow concepts.
This is explicitly documented in `src/indicators/foundation/volume.py`:

> *"Signed pressure features here are tick-volume-backed pressure proxies,
> not true bid/ask delta, net volume, or real signed traded volume."*

### Current Proxy → Real Data Mapping

| Current Feature | Module | What It Does | NT Replacement |
|-----------------|--------|-------------|----------------|
| `pressure_signed` | `volume.py` | Wick heuristic: assigns direction based on close position within body/wick | Real bar delta from footprint (ask vol − bid vol) |
| `vol_ratio`, `vol_zscore_20` | `volume.py` | Tick volume normalized to rolling baseline | Same computation but on real traded volume |
| `vp_poc`, `vp_vah`, `vp_val` | `volume_profile.py` | Distributes aggregate bar volume evenly across `n_bins` price bins | Exact volume at each price tick from footprint |
| No equivalent | — | — | L2 DOM depth & imbalance (resting orders) |
| No equivalent | — | — | Cumulative delta (session-level buy/sell pressure) |
| No equivalent | — | — | Per-level imbalance ratios (aggressive vs passive) |
| No equivalent | — | — | Absorption detection (high volume + small range at price level) |

### Impact on Existing Indicators

The following SMC indicators would gain signal quality from real order flow:

- **Order Blocks** (`smc/ob.py`): Currently detected from price impulse
  structure. With footprint data, you can validate whether aggressive
  absorption actually occurred at the OB level (high delta imbalance at
  that price row). This separates institutional OBs from noise.

- **FVG Fill** (`smc/fvg_fill.py`): Tracks gap fills from price action.
  DOM data at the FVG boundary tells you whether resting orders are
  stacked there (likely to hold) or thin (likely to break through).

- **SMT Divergence** (`smt.py`): Detects swing divergence between
  correlated pairs. Adding cumulative delta means you can detect delta
  divergence at the same swing points — price makes new high but delta
  doesn't — a higher-signal confirmation.

- **Displacement** (`smc/displacement.py`): Scores impulse candles. Real
  delta confirms whether the impulse had genuine aggressive buying/selling
  behind it or was a thin-liquidity spike.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    NinjaTrader 8                         │
│                                                          │
│  NTOrderFlowServer.cs (NinjaScript Indicator)            │
│  ┌────────────────────────────────────────────────┐      │
│  │  OnBarUpdate()  → bar JSON (OHLCV + footprint) │      │
│  │  OnMarketDepth()→ DOM update JSON              │      │
│  │  CumDelta       → delta OHLC per bar           │      │
│  └────────────────────┬───────────────────────────┘      │
│                       │ TCP :5100                         │
│                       │ newline-delimited JSON             │
└───────────────────────┼──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│                 Python Ingest Layer                       │
│                                                          │
│  src/ingest/nt_socket_client.py                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  NTSocketClient                                │      │
│  │    ├── asyncio TCP reader                      │      │
│  │    ├── JSON parse + dispatch                   │      │
│  │    └── OrderFlowBuffer (per instrument)        │      │
│  │         ├── bars: deque[BarMessage]             │      │
│  │         ├── dom: DOMSnapshot (latest)           │      │
│  │         ├── to_bar_dataframe() → DataFrame      │      │
│  │         └── get_dom_features() → dict           │      │
│  └────────────────────┬───────────────────────────┘      │
│                       │                                   │
│  Instrument mapping:  │  GC → XAU_USD                    │
│  (futures → spot)     │  CL → USOIL                      │
│                       │  6J → USD_JPY                     │
│                       │  DX → DXY                         │
└───────────────────────┼──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│                 Live Pipeline                             │
│                                                          │
│  src/scanner/live_scanner.py                             │
│  ┌────────────────────────────────────────────────┐      │
│  │  1. fetch OANDA candles (existing OHLCV flow)  │      │
│  │  2. build_live_indicators(df)                  │      │
│  │     └── 28-stage DAG (unchanged)               │      │
│  │  3. merge order flow columns from NT buffer    │      │
│  │     └── of_* columns joined on timestamp       │      │
│  │  4. add_order_flow_features(df)                │      │
│  │     └── derived features (delta div, absorb)   │      │
│  │  5. attach DOM snapshot to latest bar          │      │
│  │  6. evaluate signals                           │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  New modules:                                            │
│    src/indicators/foundation/order_flow.py               │
│    src/indicators/foundation/volume_profile.py (upgrade) │
│    src/ingest/nt_socket_client.py                        │
│    src/ingest/nt_symbol_map.py                           │
└──────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Existing pipeline stays untouched.** Order flow columns are merged
   alongside the existing DAG output, not injected into the DAG stages.
   This means the 28-stage indicator chain keeps working identically with
   or without NT data.

2. **Graceful degradation.** If the NT socket is down or data is missing,
   `of_*` columns are NaN and `add_order_flow_features()` returns the
   frame unchanged. Existing tick-volume proxies remain as fallback.

3. **Futures → spot alignment by bar timestamp.** Both OANDA (spot) and NT
   (futures) produce bars at the same clock boundaries (e.g., H1 bars at
   :00). The merge is a left join on timestamp — the spot frame is the
   primary, and futures order flow columns attach where timestamps match.

---

## 4. Instrument Mapping: Futures ↔ Spot

NinjaTrader streams futures data. The existing pipeline uses OANDA spot
symbols. The mapping:

| NT Futures Symbol | CME/Exchange | Pipeline Spot Symbol | Notes |
|-------------------|-------------|---------------------|-------|
| `GC MM-YY` | COMEX | `XAU_USD` | Gold futures. Near-lockstep with spot on H1+. Contango spread is ~0.1% and irrelevant for order flow features. |
| `CL MM-YY` | NYMEX | `USOIL` | Crude oil futures. Tight to spot. |
| `6J MM-YY` | CME | `USD_JPY` | JPY futures (inverted: 6J = JPY/USD). Invert price before delta comparison. |
| `DX MM-YY` | ICE | `DXY` | Dollar index futures. Direct mapping. |
| `6E MM-YY` | CME | `EUR_USD` | Euro FX futures. Available if you extend cross-asset. |
| `6B MM-YY` | CME | `GBP_USD` | Cable futures. Same. |
| `ES MM-YY` | CME | — | S&P 500 e-mini. Not currently in CONTEXT_SYMBOLS but useful for risk-on/off context. |
| `NQ MM-YY` | CME | — | Nasdaq e-mini. Same. |

### Timestamp Alignment

Futures bars (CME) and spot bars (OANDA) use the same UTC clock boundaries
for standard timeframes (H1, H4). Alignment procedure:

1. Round both timestamps to the bar boundary: `ts.floor(freq)`.
2. Left-join spot (primary) onto futures (order flow) on the rounded
   timestamp.
3. For timeframes where session differences matter (e.g., futures have a
   17:00 CT close, spot is continuous): trim to overlapping session hours
   before computing session-level cumulative delta.

### Contract Rollover

Futures contracts expire quarterly (GC, 6J, DX) or monthly (CL). The NT
symbol changes (e.g., `GC 06-26` → `GC 08-26`). The Python ingest layer
must:

1. Strip the contract month from the NT instrument name when mapping.
2. Use a configurable regex: `r"^([A-Z0-9]+)\s+\d{2}-\d{2}$"` → group 1
   is the root symbol.
3. Alternatively, configure NT to use continuous contract symbols if your
   data provider supports them.

---

## 5. Step 1 — NinjaScript TCP Server (C#)

### File: `NTOrderFlowServer.cs`

Deploy into `Documents/NinjaTrader 8/bin/Custom/Indicators/`.

```csharp
#region Using declarations
using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class NTOrderFlowServer : Indicator
    {
        private TcpListener listener;
        private readonly List<TcpClient> clients = new List<TcpClient>();
        private readonly object clientLock = new object();
        private Thread acceptThread;
        private volatile bool running;

        // --- Cumulative delta indicator reference ---
        private OrderFlowCumulativeDelta cumDelta;

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Port", Order = 1, GroupName = "Server")]
        public int Port { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Stream DOM", Order = 2, GroupName = "Server")]
        public bool StreamDOM { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "DOM Throttle Ms", Order = 3, GroupName = "Server")]
        public int DOMThrottleMs { get; set; }
        #endregion

        private DateTime lastDOMBroadcast = DateTime.MinValue;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Streams bar footprint + DOM data over TCP.";
                Name = "NTOrderFlowServer";
                Calculate = Calculate.OnBarClose;
                IsOverlay = true;
                Port = 5100;
                StreamDOM = true;
                DOMThrottleMs = 250;
            }
            else if (State == State.Configure)
            {
                // Add cumulative delta as a hosted indicator.
                cumDelta = OrderFlowCumulativeDelta(
                    CumulativeDeltaType.BidAsk,
                    CumulativeDeltaPeriod.Session,
                    0
                );
            }
            else if (State == State.DataLoaded)
            {
                running = true;
                listener = new TcpListener(IPAddress.Loopback, Port);
                listener.Start();
                acceptThread = new Thread(AcceptLoop)
                {
                    IsBackground = true,
                    Name = "NT_OF_Accept"
                };
                acceptThread.Start();
                Print("NTOrderFlowServer listening on port " + Port);
            }
            else if (State == State.Terminated)
            {
                running = false;
                try { listener?.Stop(); } catch { }
                lock (clientLock)
                {
                    foreach (var c in clients)
                        try { c.Close(); } catch { }
                    clients.Clear();
                }
            }
        }

        private void AcceptLoop()
        {
            while (running)
            {
                try
                {
                    var client = listener.AcceptTcpClient();
                    client.NoDelay = true;
                    lock (clientLock) { clients.Add(client); }
                    Print("NTOrderFlowServer: client connected");
                }
                catch (SocketException)
                {
                    break; // listener stopped
                }
            }
        }

        // -------------------------------------------------------
        // Bar close: send OHLCV + volumetric footprint + cum delta
        // -------------------------------------------------------
        protected override void OnBarUpdate()
        {
            if (CurrentBar < 1) return;

            var sb = new StringBuilder(512);
            sb.Append('{');
            sb.AppendFormat(
                "\"type\":\"bar\","
                + "\"ts\":\"{0:O}\","
                + "\"instrument\":\"{1}\","
                + "\"o\":{2},\"h\":{3},\"l\":{4},\"c\":{5},"
                + "\"vol\":{6},",
                Time[0],
                Instrument.FullName,
                Open[0], High[0], Low[0], Close[0],
                Volume[0]
            );

            // --- Volumetric footprint data ---
            // Requires OrderFlowVolumetricBars to be loaded as a
            // BarsType on the chart. Access via:
            //   NinjaTrader.NinjaScript.BarsTypes.VolumetricBarsType
            // This block emits per-price-level bid/ask volume.
            sb.Append("\"footprint\":[");
            try
            {
                var volBars = Bars.BarsSeries as
                    NinjaTrader.NinjaScript.BarsTypes
                    .VolumetricBarsType;
                if (volBars != null)
                {
                    var volumes = volBars.Volumes[CurrentBar];
                    double tickSize = Instrument.MasterInstrument.TickSize;
                    double lo = Low[0];
                    double hi = High[0];
                    bool first = true;
                    for (double p = lo; p <= hi; p += tickSize)
                    {
                        long bidVol, askVol;
                        volumes.GetBidAskVolumeAtPrice(p,
                            out bidVol, out askVol);
                        if (bidVol == 0 && askVol == 0) continue;
                        if (!first) sb.Append(',');
                        sb.AppendFormat(
                            "{{\"p\":{0},\"b\":{1},\"a\":{2}}}",
                            p, bidVol, askVol
                        );
                        first = false;
                    }
                }
            }
            catch { /* Volumetric not available — send empty */ }
            sb.Append("],");

            // --- Cumulative delta ---
            double cumDeltaClose = 0;
            double cumDeltaHigh = 0;
            double cumDeltaLow = 0;
            try
            {
                if (cumDelta != null
                    && cumDelta.BarsArray[1] != null
                    && cumDelta.BarsArray[1].Count > 0)
                {
                    cumDeltaClose = cumDelta.DeltaClose[0];
                    cumDeltaHigh = cumDelta.DeltaHigh[0];
                    cumDeltaLow = cumDelta.DeltaLow[0];
                }
            }
            catch { }

            sb.AppendFormat(
                "\"cum_delta_close\":{0},"
                + "\"cum_delta_high\":{1},"
                + "\"cum_delta_low\":{2}",
                cumDeltaClose, cumDeltaHigh, cumDeltaLow
            );
            sb.Append("}\n");

            Broadcast(sb.ToString());
        }

        // -------------------------------------------------------
        // DOM updates: stream L2 book changes (throttled)
        // -------------------------------------------------------
        protected override void OnMarketDepth(MarketDepthEventArgs e)
        {
            if (!StreamDOM) return;

            // Throttle: at most one broadcast per DOMThrottleMs
            var now = DateTime.UtcNow;
            if ((now - lastDOMBroadcast).TotalMilliseconds < DOMThrottleMs)
                return;
            lastDOMBroadcast = now;

            var msg = string.Format(
                "{{\"type\":\"dom\","
                + "\"ts\":\"{0:O}\","
                + "\"instrument\":\"{1}\","
                + "\"side\":\"{2}\","
                + "\"price\":{3},"
                + "\"volume\":{4},"
                + "\"pos\":{5},"
                + "\"op\":\"{6}\""
                + "}}\n",
                now,
                Instrument.FullName,
                e.MarketDataType == MarketDataType.Ask ? "ask" : "bid",
                e.Price,
                e.Volume,
                e.Position,
                e.Operation
            );

            Broadcast(msg);
        }

        private void Broadcast(string message)
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            lock (clientLock)
            {
                for (int i = clients.Count - 1; i >= 0; i--)
                {
                    try
                    {
                        if (!clients[i].Connected)
                        {
                            clients.RemoveAt(i);
                            continue;
                        }
                        clients[i].GetStream().Write(data, 0, data.Length);
                    }
                    catch
                    {
                        try { clients[i].Close(); } catch { }
                        clients.RemoveAt(i);
                    }
                }
            }
        }
    }
}
```

### NT Setup Checklist

1. Copy `NTOrderFlowServer.cs` to
   `Documents/NinjaTrader 8/bin/Custom/Indicators/`.
2. In NT8 → Tools → NinjaScript Editor → compile.
3. Open a chart with **Volumetric Bars** as the bar type (right-click →
   Data Series → Bars type → `OrderFlowVolumetricBars`). This is required
   for footprint data.
4. Add the `NTOrderFlowServer` indicator to the chart.
5. Configure the port (default 5100) and DOM streaming.
6. Repeat for each instrument chart you want to stream (GC, CL, 6J, DX).

### Wire Protocol

Newline-delimited JSON (`\n`-terminated). Two message types:

**Bar message** (on bar close):
```json
{
  "type": "bar",
  "ts": "2026-04-22T14:00:00.0000000Z",
  "instrument": "GC 06-26",
  "o": 2350.5, "h": 2355.0, "l": 2349.8, "c": 2354.2,
  "vol": 12500,
  "footprint": [
    {"p": 2349.8, "b": 120, "a": 85},
    {"p": 2349.9, "b": 95,  "a": 210},
    {"p": 2350.0, "b": 45,  "a": 380}
  ],
  "cum_delta_close": 4520,
  "cum_delta_high": 5100,
  "cum_delta_low": 3200
}
```

**DOM message** (throttled, on book change):
```json
{
  "type": "dom",
  "ts": "2026-04-22T14:00:01.2340000Z",
  "instrument": "GC 06-26",
  "side": "bid",
  "price": 2354.0,
  "volume": 85,
  "pos": 0,
  "op": "Update"
}
```

---

## 6. Step 2 — Python Socket Client & Ring Buffer

### File: `src/ingest/nt_socket_client.py`

```python
"""
Async TCP client for NinjaTrader NTOrderFlowServer.

Connects to localhost:5100, accumulates bar + DOM messages into
per-instrument ring buffers. Exposes DataFrames for the live pipeline
to merge onto its candle data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regex to strip contract month: "GC 06-26" → "GC"
_CONTRACT_RE = re.compile(r"^([A-Z0-9]+)\s+\d{2}-\d{2}$")


@dataclass(slots=True)
class FootprintLevel:
    price: float
    bid_vol: float
    ask_vol: float

    @property
    def delta(self) -> float:
        return self.ask_vol - self.bid_vol

    @property
    def total_vol(self) -> float:
        return self.bid_vol + self.ask_vol


@dataclass(slots=True)
class BarMessage:
    timestamp: datetime
    instrument: str
    o: float
    h: float
    l: float
    c: float
    vol: float
    cum_delta_close: float
    cum_delta_high: float
    cum_delta_low: float
    footprint: list[FootprintLevel]


@dataclass
class DOMSnapshot:
    """Reconstructed order book from incremental DOM updates."""
    timestamp: datetime
    instrument: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)


@dataclass
class OrderFlowBuffer:
    """Per-instrument ring buffer of order flow data."""
    max_bars: int = 2000
    bars: deque[BarMessage] = field(default_factory=deque)
    dom: DOMSnapshot | None = None

    def push_bar(self, bar: BarMessage) -> None:
        self.bars.append(bar)
        if len(self.bars) > self.max_bars:
            self.bars.popleft()

    def update_dom(self, msg: dict) -> None:
        if self.dom is None:
            self.dom = DOMSnapshot(
                timestamp=datetime.fromisoformat(msg["ts"]),
                instrument=msg["instrument"],
            )
        side_book = (
            self.dom.bids if msg["side"] == "bid" else self.dom.asks
        )
        price = msg["price"]
        volume = msg["volume"]
        op = msg.get("op", "Update")

        if op == "Remove" or volume == 0:
            side_book.pop(price, None)
        else:
            side_book[price] = volume

        self.dom.timestamp = datetime.fromisoformat(msg["ts"])

    def to_bar_dataframe(self) -> pd.DataFrame:
        """Convert accumulated bars → DataFrame with of_* columns.

        Designed for left-join onto the candle DataFrame on timestamp.
        """
        if not self.bars:
            return pd.DataFrame()

        records = []
        prev_cum_delta = 0.0

        for bar in self.bars:
            total_bid = sum(lv.bid_vol for lv in bar.footprint)
            total_ask = sum(lv.ask_vol for lv in bar.footprint)
            bar_delta = total_ask - total_bid

            # Imbalance: price levels where one side > 3× the other.
            buy_imbalances = sum(
                1 for lv in bar.footprint
                if lv.bid_vol > 0 and lv.ask_vol / lv.bid_vol > 3.0
            )
            sell_imbalances = sum(
                1 for lv in bar.footprint
                if lv.ask_vol > 0 and lv.bid_vol / lv.ask_vol > 3.0
            )

            # Delta change from previous bar.
            delta_change = bar.cum_delta_close - prev_cum_delta
            prev_cum_delta = bar.cum_delta_close

            # POC from footprint (price with max total volume).
            poc_price = np.nan
            if bar.footprint:
                poc_level = max(bar.footprint, key=lambda lv: lv.total_vol)
                poc_price = poc_level.price

            records.append({
                "timestamp": bar.timestamp,
                "of_bar_delta": bar_delta,
                "of_bid_vol": total_bid,
                "of_ask_vol": total_ask,
                "of_cum_delta": bar.cum_delta_close,
                "of_cum_delta_high": bar.cum_delta_high,
                "of_cum_delta_low": bar.cum_delta_low,
                "of_delta_change": delta_change,
                "of_buy_imbalances": buy_imbalances,
                "of_sell_imbalances": sell_imbalances,
                "of_footprint_levels": len(bar.footprint),
                "of_poc_price": poc_price,
            })

        return pd.DataFrame(records)

    def get_footprint_levels(
        self, n_bars: int = 80,
    ) -> list[list[FootprintLevel]]:
        """Raw footprint levels for last n_bars. Used by VP upgrade."""
        bars = list(self.bars)[-n_bars:]
        return [bar.footprint for bar in bars]

    def get_dom_features(self) -> dict[str, float] | None:
        """Extract features from the latest DOM snapshot."""
        if self.dom is None:
            return None
        bid_depth = sum(self.dom.bids.values())
        ask_depth = sum(self.dom.asks.values())
        total = bid_depth + ask_depth

        # Top-of-book (best bid/ask size).
        best_bid_vol = (
            self.dom.bids[max(self.dom.bids)] if self.dom.bids else 0.0
        )
        best_ask_vol = (
            self.dom.asks[min(self.dom.asks)] if self.dom.asks else 0.0
        )

        return {
            "dom_bid_depth": bid_depth,
            "dom_ask_depth": ask_depth,
            "dom_imbalance_ratio": (
                (bid_depth - ask_depth) / total if total > 0 else 0.0
            ),
            "dom_levels_bid": len(self.dom.bids),
            "dom_levels_ask": len(self.dom.asks),
            "dom_best_bid_vol": best_bid_vol,
            "dom_best_ask_vol": best_ask_vol,
            "dom_spread_levels": (
                (min(self.dom.asks) - max(self.dom.bids))
                if self.dom.bids and self.dom.asks
                else np.nan
            ),
        }


class NTSocketClient:
    """Async TCP client that connects to NinjaTrader and fills buffers."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5100,
        symbol_map: dict[str, str] | None = None,
        max_bars: int = 2000,
    ):
        self.host = host
        self.port = port
        self.symbol_map = symbol_map or {}
        self.max_bars = max_bars
        self.buffers: dict[str, OrderFlowBuffer] = defaultdict(
            lambda: OrderFlowBuffer(max_bars=self.max_bars)
        )
        self._running = False
        self._connected = False

    def _resolve_symbol(self, nt_instrument: str) -> str:
        """Map NT futures symbol to pipeline spot symbol.

        Tries exact match first, then strips contract month.
        """
        if nt_instrument in self.symbol_map:
            return self.symbol_map[nt_instrument]
        m = _CONTRACT_RE.match(nt_instrument)
        root = m.group(1) if m else nt_instrument
        return self.symbol_map.get(root, root)

    async def connect_and_stream(self) -> None:
        """Main loop. Connects to NT, reads messages, fills buffers.

        Reconnects automatically on disconnect with 5s backoff.
        """
        self._running = True
        while self._running:
            try:
                reader, _ = await asyncio.open_connection(
                    self.host, self.port,
                )
                self._connected = True
                logger.info(
                    "Connected to NTOrderFlowServer at %s:%d",
                    self.host, self.port,
                )
                await self._read_loop(reader)
            except (ConnectionRefusedError, OSError) as exc:
                self._connected = False
                logger.warning(
                    "NT connection failed (%s) — retrying in 5s", exc,
                )
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
        self._connected = False

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        while self._running:
            line = await reader.readline()
            if not line:
                logger.warning("NT connection closed by remote")
                self._connected = False
                return
            try:
                msg = json.loads(line)
                self._dispatch(msg)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.debug("Malformed NT message: %s", exc)

    def _dispatch(self, msg: dict) -> None:
        symbol = self._resolve_symbol(msg.get("instrument", ""))
        buf = self.buffers[symbol]

        if msg["type"] == "bar":
            footprint = [
                FootprintLevel(
                    price=lv["p"],
                    bid_vol=lv["b"],
                    ask_vol=lv["a"],
                )
                for lv in msg.get("footprint", [])
            ]
            buf.push_bar(BarMessage(
                timestamp=datetime.fromisoformat(msg["ts"]),
                instrument=symbol,
                o=msg["o"],
                h=msg["h"],
                l=msg["l"],
                c=msg["c"],
                vol=msg["vol"],
                cum_delta_close=msg.get("cum_delta_close", 0.0),
                cum_delta_high=msg.get("cum_delta_high", 0.0),
                cum_delta_low=msg.get("cum_delta_low", 0.0),
                footprint=footprint,
            ))

        elif msg["type"] == "dom":
            buf.update_dom(msg)

    # --- Public API for the scanner ---

    def get_order_flow_frame(self, symbol: str) -> pd.DataFrame:
        """DataFrame of of_* columns, keyed by timestamp.

        Left-join this onto your candle DataFrame.
        """
        return self.buffers[symbol].to_bar_dataframe()

    def get_dom_snapshot(self, symbol: str) -> dict[str, float] | None:
        """Latest DOM features for a symbol. Attach to latest bar."""
        return self.buffers[symbol].get_dom_features()

    def get_footprint_levels(
        self, symbol: str, n_bars: int = 80,
    ) -> list[list[FootprintLevel]]:
        """Raw footprint data for VP computation."""
        return self.buffers[symbol].get_footprint_levels(n_bars)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        self._running = False
```

### File: `src/ingest/nt_symbol_map.py`

```python
"""
Futures → spot symbol mapping for NinjaTrader integration.

The root symbol (without contract month) is the lookup key.
Contract month is stripped automatically by NTSocketClient.
"""

# Root symbol → pipeline symbol.
NT_FUTURES_TO_SPOT: dict[str, str] = {
    "GC": "XAU_USD",
    "CL": "USOIL",
    "6J": "USD_JPY",
    "DX": "DXY",
    "6E": "EUR_USD",
    "6B": "GBP_USD",
    "6A": "AUD_USD",
    "6C": "USD_CAD",
    "6S": "USD_CHF",
    "6N": "NZD_USD",
    "ES": "SP500",
    "NQ": "NASDAQ",
}

# Instruments where 6J-style inversion is needed.
# 6J quotes JPY/USD, but pipeline uses USD/JPY.
# Delta sign must be flipped when mapping.
INVERTED_FUTURES = {"6J"}
```

---

## 7. Step 3 — Wire Into the Live Pipeline

The merge point is **after** `build_live_indicators()` returns and
**before** signal evaluation. This keeps the existing DAG untouched.

### File: `src/scanner/live_scanner.py` (scaffold)

```python
"""
Live scanner: runs the indicator pipeline + merges NT order flow.

This is the orchestration layer. It:
1. Fetches OANDA candles (existing data source).
2. Runs the 28-stage indicator DAG via build_live_indicators().
3. Merges order flow columns from the NTSocketClient buffer.
4. Computes derived order flow features.
5. Attaches the latest DOM snapshot.
6. Evaluates trade signals.
"""

from __future__ import annotations

import asyncio
import logging

import pandas as pd

from src.indicators.foundation.order_flow import add_order_flow_features
from src.indicators.pipelines.build_live import build_live_indicators
from src.ingest.nt_socket_client import NTSocketClient
from src.ingest.nt_symbol_map import NT_FUTURES_TO_SPOT

logger = logging.getLogger(__name__)


def _align_timestamps(
    primary: pd.DataFrame,
    of_frame: pd.DataFrame,
    freq: str = "h",
) -> pd.DataFrame:
    """Left-join order flow onto primary candles by floored timestamp."""
    p = primary.copy()
    o = of_frame.copy()
    p["_merge_ts"] = pd.to_datetime(p["timestamp"], utc=True).dt.floor(freq)
    o["_merge_ts"] = pd.to_datetime(o["timestamp"], utc=True).dt.floor(freq)
    o = o.drop(columns=["timestamp"])
    merged = p.merge(o, on="_merge_ts", how="left").drop(columns=["_merge_ts"])
    return merged


async def run_live_scanner(
    instrument: str = "XAU_USD",
    timeframe: str = "H1",
    nt_host: str = "127.0.0.1",
    nt_port: int = 5100,
    poll_interval_s: int = 60,
) -> None:
    """Main scanner loop."""

    # 1. Start NT socket client.
    nt_client = NTSocketClient(
        host=nt_host,
        port=nt_port,
        symbol_map=NT_FUTURES_TO_SPOT,
    )
    nt_task = asyncio.create_task(nt_client.connect_and_stream())

    try:
        while True:
            # 2. Fetch candles from OANDA (placeholder — wire your broker).
            candle_df = await _fetch_candles(instrument, timeframe)

            # 3. Run existing indicator pipeline (unchanged).
            features = build_live_indicators(
                candle_df,
                instrument=instrument,
                timeframe=timeframe,
                include_cross_asset=True,
            )

            # 4. Merge order flow columns from NT buffer.
            of_frame = nt_client.get_order_flow_frame(instrument)
            if not of_frame.empty:
                freq = {"H1": "h", "H4": "4h", "M15": "15min"}.get(
                    timeframe, "h"
                )
                features = _align_timestamps(features, of_frame, freq=freq)

            # 5. Compute derived order flow features.
            features = add_order_flow_features(features)

            # 6. Attach DOM snapshot to the latest bar.
            dom = nt_client.get_dom_snapshot(instrument)
            if dom:
                for col, val in dom.items():
                    features.loc[features.index[-1], col] = val

            # 7. Evaluate signals.
            # features now has the full 28-stage indicator stack
            # PLUS of_* order flow columns and dom_* columns.
            await _evaluate_signals(features, instrument)

            await asyncio.sleep(poll_interval_s)

    finally:
        nt_client.stop()
        nt_task.cancel()


async def _fetch_candles(
    instrument: str, timeframe: str,
) -> pd.DataFrame:
    """Placeholder — replace with your OANDA / broker fetch."""
    raise NotImplementedError("Wire your candle data source here")


async def _evaluate_signals(
    features: pd.DataFrame, instrument: str,
) -> None:
    """Placeholder — replace with your signal evaluation logic."""
    raise NotImplementedError("Wire your signal logic here")
```

### Merge semantics

The merge is a **left join** — the candle DataFrame (from OANDA) is the
primary frame. Order flow columns attach where timestamps match. Bars
without NT data (e.g., during NT downtime or non-overlapping session
hours) get NaN in the `of_*` and `dom_*` columns. All downstream order
flow features handle NaN gracefully.

---

## 8. Step 4 — Order Flow Indicator Module

### File: `src/indicators/foundation/order_flow.py`

```python
"""
Order flow indicators derived from NinjaTrader volumetric data.

Real bid/ask delta and footprint features. Only populated when of_*
columns are present from NT merge; returns the frame unchanged otherwise.

Column prefix: ``of_`` for raw ingested columns (from NT buffer),
``ofi_`` for derived features computed here.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Rolling windows for derived features.
DELTA_MA_PERIOD = 10
DELTA_ZSCORE_PERIOD = 50
ABSORPTION_VOL_MULT = 2.0
ABSORPTION_RANGE_ATR_THRESH = 0.5
IMBALANCE_EMA_PERIOD = 14


def add_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived order flow features from of_* columns.

    If of_bar_delta is not present (no NT data), returns df unchanged.
    """
    if "of_bar_delta" not in df.columns:
        return df

    out = df.copy()

    # --- Delta features ---
    out["ofi_delta_ma"] = (
        out["of_bar_delta"].rolling(DELTA_MA_PERIOD, min_periods=1).mean()
    )
    delta_std = out["of_bar_delta"].rolling(
        DELTA_ZSCORE_PERIOD, min_periods=10,
    ).std()
    delta_mean = out["of_bar_delta"].rolling(
        DELTA_ZSCORE_PERIOD, min_periods=10,
    ).mean()
    out["ofi_delta_zscore"] = np.where(
        delta_std > 0,
        (out["of_bar_delta"] - delta_mean) / delta_std,
        0.0,
    )

    # --- Cumulative delta features ---
    out["ofi_cum_delta_slope"] = (
        out["of_cum_delta"]
        .rolling(DELTA_MA_PERIOD, min_periods=2)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    )
    # Cum delta bar range (intra-bar delta volatility).
    out["ofi_cum_delta_range"] = (
        out["of_cum_delta_high"] - out["of_cum_delta_low"]
    )

    # --- Delta divergence from price ---
    # Price up + delta weakening = bearish divergence.
    # Price down + delta strengthening = bullish divergence.
    price_up = out["close"] > out["close"].shift(1)
    price_dn = out["close"] < out["close"].shift(1)
    delta_weakening = out["ofi_delta_ma"] < out["ofi_delta_ma"].shift(1)
    delta_strengthening = out["ofi_delta_ma"] > out["ofi_delta_ma"].shift(1)

    out["ofi_delta_divergence"] = np.select(
        [price_up & delta_weakening, price_dn & delta_strengthening],
        [-1, 1],
        default=0,
    )

    # --- Buy/sell pressure ratio ---
    total_vol = out["of_bid_vol"] + out["of_ask_vol"]
    out["ofi_buy_pct"] = np.where(
        total_vol > 0, out["of_ask_vol"] / total_vol, 0.5,
    )

    # --- Imbalance strength ---
    total_imbalances = out["of_buy_imbalances"] + out["of_sell_imbalances"]
    out["ofi_imbalance_strength"] = np.where(
        out["of_footprint_levels"] > 0,
        total_imbalances / out["of_footprint_levels"],
        0.0,
    )
    out["ofi_imbalance_bias"] = np.where(
        total_imbalances > 0,
        (out["of_buy_imbalances"] - out["of_sell_imbalances"])
        / total_imbalances,
        0.0,
    )
    out["ofi_imbalance_ema"] = (
        out["ofi_imbalance_bias"]
        .ewm(span=IMBALANCE_EMA_PERIOD, min_periods=1)
        .mean()
    )

    # --- Absorption detection ---
    # High volume + small price range = absorption at that level.
    if "atr_14" in out.columns:
        bar_range = out["high"] - out["low"]
        atr = out["atr_14"]
        vol_baseline = total_vol.rolling(20, min_periods=5).mean()
        out["ofi_absorption_flag"] = (
            (total_vol > vol_baseline * ABSORPTION_VOL_MULT)
            & (bar_range < atr * ABSORPTION_RANGE_ATR_THRESH)
        ).astype(int)
    else:
        out["ofi_absorption_flag"] = 0

    # --- Footprint POC distance from close ---
    if "of_poc_price" in out.columns and "atr_14" in out.columns:
        out["ofi_poc_dist_atr"] = np.where(
            out["atr_14"] > 0,
            (out["close"] - out["of_poc_price"]) / out["atr_14"],
            0.0,
        )

    return out
```

---

## 9. Step 5 — Upgrade Volume Profile With Real Footprint

The existing `compute_volume_profile` in `volume_profile.py` distributes
aggregate bar volume across price bins via linear interpolation. With
footprint data from NT, you compute the exact volume at each traded price.

### Addition to `src/indicators/foundation/volume_profile.py`

```python
def compute_volume_profile_from_footprint(
    footprint_bars: list[list["FootprintLevel"]],
    n_recent_bars: int = 80,
) -> dict:
    """Volume Profile from actual traded volume at each price tick.

    Uses real bid/ask volume from NT volumetric data instead of
    distributing aggregate bar volume across OHLC bins. This gives
    exact POC, VAH, VAL.

    Parameters
    ----------
    footprint_bars : list of list of FootprintLevel
        Raw footprint data from NTSocketClient.get_footprint_levels().
    n_recent_bars : int
        Number of recent bars to include.

    Returns
    -------
    dict with keys: poc, vah, val, profile (full price→volume map).
    """
    price_volume: dict[float, float] = {}
    for bar_levels in footprint_bars[-n_recent_bars:]:
        for lv in bar_levels:
            price_volume[lv.price] = (
                price_volume.get(lv.price, 0.0) + lv.total_vol
            )

    if not price_volume:
        return {"poc": None, "vah": None, "val": None, "profile": {}}

    # POC = price with maximum total volume.
    poc_price = max(price_volume, key=price_volume.get)

    # Value Area = 70% of total volume, expanding outward from POC.
    total = sum(price_volume.values())
    target = total * 0.70
    sorted_prices = sorted(price_volume.keys())
    poc_idx = sorted_prices.index(poc_price)

    accumulated = price_volume[poc_price]
    lo_idx, hi_idx = poc_idx, poc_idx

    while accumulated < target and (
        lo_idx > 0 or hi_idx < len(sorted_prices) - 1
    ):
        expand_up = (
            price_volume[sorted_prices[hi_idx + 1]]
            if hi_idx < len(sorted_prices) - 1
            else 0.0
        )
        expand_dn = (
            price_volume[sorted_prices[lo_idx - 1]]
            if lo_idx > 0
            else 0.0
        )
        if expand_up >= expand_dn and hi_idx < len(sorted_prices) - 1:
            hi_idx += 1
            accumulated += expand_up
        elif lo_idx > 0:
            lo_idx -= 1
            accumulated += expand_dn
        else:
            hi_idx += 1
            accumulated += expand_up

    return {
        "poc": poc_price,
        "vah": sorted_prices[hi_idx],
        "val": sorted_prices[lo_idx],
        "profile": price_volume,
    }
```

### Integration in the scanner

```python
# In live_scanner.py, after merging of_* columns:
from src.indicators.foundation.volume_profile import (
    compute_volume_profile_from_footprint,
)

footprint = nt_client.get_footprint_levels("XAU_USD", n_bars=80)
if footprint and any(bar for bar in footprint):
    vp = compute_volume_profile_from_footprint(footprint)
    # Overwrite the proxy VP columns on the latest bar.
    if vp["poc"] is not None:
        features.loc[features.index[-1], "vp_poc"] = vp["poc"]
        features.loc[features.index[-1], "vp_vah"] = vp["vah"]
        features.loc[features.index[-1], "vp_val"] = vp["val"]
```

This means for historical bars you still use the existing OHLCV-based VP
(which is a reasonable approximation), and for the live frontier you get
exact VP from real footprint data.

---

## 10. Step 6 — Order Flow–Enhanced SMC Indicators

These are **not** replacements for existing indicators. They are
supplementary columns that add order-flow confirmation to existing SMC
signals. The existing indicators continue to work with or without NT data.

### 10.1 OB Confirmation (delta at order block levels)

When an order block is detected by `add_ob()`, check whether the footprint
at the OB price range shows aggressive absorption:

```python
def _ob_delta_confirmation(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Add ofi_ob_*_confirmed columns.

    An OB is 'confirmed' if the footprint delta at the OB price range
    shows significant opposing flow (absorption).
    """
    out = features.copy()
    for side in ("bull", "bear"):
        ob_flag = f"ob_{side}"
        ob_lo = f"ob_{side}_low"
        ob_hi = f"ob_{side}_high"
        out_col = f"ofi_ob_{side}_confirmed"

        if ob_flag not in out.columns or "of_bar_delta" not in out.columns:
            continue

        # For bullish OB: expect strong negative delta (selling absorbed).
        # For bearish OB: expect strong positive delta (buying absorbed).
        expected_sign = -1 if side == "bull" else 1
        threshold = out["of_bar_delta"].rolling(50, min_periods=10).std() * 1.5

        out[out_col] = (
            (out[ob_flag] == 1)
            & (out["of_bar_delta"] * expected_sign > threshold)
        ).astype(int)

    return out
```

### 10.2 FVG Hold Probability (DOM depth at gap boundary)

When price approaches an active FVG, the DOM tells you whether resting
orders are stacked at the boundary (likely to hold) or thin (likely to
break):

```python
def _fvg_dom_strength(
    features: pd.DataFrame,
    dom_snapshot: dict[str, float] | None,
) -> pd.DataFrame:
    """Add dom_fvg_*_strength to the latest bar.

    Uses DOM depth at the FVG boundary to estimate hold probability.
    """
    if dom_snapshot is None:
        return features

    out = features.copy()
    latest = out.index[-1]

    for side in ("bull", "bear"):
        active_col = f"fvg_{side}_active"
        boundary_col = (
            f"fvg_{side}_active_low" if side == "bull"
            else f"fvg_{side}_active_high"
        )
        if active_col not in out.columns:
            continue
        if out.loc[latest, active_col] != 1:
            continue

        # DOM depth on the side that defends the FVG.
        # Bull FVG: bid depth defends (buyers at the gap low).
        # Bear FVG: ask depth defends (sellers at the gap high).
        defending_depth = (
            dom_snapshot["dom_bid_depth"]
            if side == "bull"
            else dom_snapshot["dom_ask_depth"]
        )
        opposing_depth = (
            dom_snapshot["dom_ask_depth"]
            if side == "bull"
            else dom_snapshot["dom_bid_depth"]
        )
        total = defending_depth + opposing_depth
        out.loc[latest, f"dom_fvg_{side}_strength"] = (
            defending_depth / total if total > 0 else 0.5
        )

    return out
```

### 10.3 SMT + Delta Divergence (composite signal)

Augment existing SMT swing divergence with cumulative delta divergence at
the same swing points:

```python
def _smt_delta_confirmation(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Add ofi_smt_*_delta_confirm columns.

    When SMT detects a swing divergence between correlated instruments,
    check whether cumulative delta also diverges at that swing —
    producing a triple divergence (price + partner + delta).
    """
    out = features.copy()
    if "of_cum_delta" not in out.columns:
        return out

    # Find SMT detection columns.
    smt_cols = [
        c for c in out.columns
        if c.endswith("_detect_flag") and c.startswith("smt_")
    ]

    for smt_col in smt_cols:
        # e.g., smt_dxy_high_detect_flag → ofi_smt_dxy_high_delta_confirm
        base = smt_col.replace("_detect_flag", "")
        confirm_col = f"ofi_{base}_delta_confirm"

        # At SMT detection bars, check if delta also diverges.
        is_high = "high" in smt_col
        if is_high:
            # Price made HH, partner didn't → bearish SMT.
            # Confirm: cum delta also didn't make new high.
            delta_confirms = (
                out["of_cum_delta"]
                < out["of_cum_delta"].rolling(20).max().shift(1)
            )
        else:
            # Price made LL, partner didn't → bullish SMT.
            # Confirm: cum delta also didn't make new low.
            delta_confirms = (
                out["of_cum_delta"]
                > out["of_cum_delta"].rolling(20).min().shift(1)
            )

        out[confirm_col] = (
            (out[smt_col] == 1) & delta_confirms
        ).astype(int)

    return out
```

---

## 11. Step 7 — DAG Integration

Phase 1 (above) keeps order flow outside the DAG. Once stable, Phase 2
brings it into the graph for caching and fingerprinting.

### Phase 2: Order Flow as a DAG Source Node

Add to `src/dag_runtime/builtin_graphs.py`:

```python
# New source node for order flow data.
NodeManifest(
    name="order_flow_input",
    node_kind="source",
    semantic_class="A",
    compute_fn=lambda ctx, deps: NodeOutput(
        frames={"frame": ctx.inputs.get("order_flow_input", pd.DataFrame())}
    ),
    cache_policy=CachePolicy(materialize=False),  # Live data, no cache
    upstream_nodes=[],
)

# New compute node that merges order flow onto the indicator frame.
NodeManifest(
    name="order_flow_merge",
    node_kind="compute",
    semantic_class="A",
    compute_fn=_merge_order_flow,
    cache_policy=CachePolicy(materialize=False),
    upstream_nodes=["regime", "order_flow_input"],  # After regime, before output
)

# Derived features node.
NodeManifest(
    name="order_flow_features",
    node_kind="compute",
    semantic_class="A",
    compute_fn=lambda ctx, deps: NodeOutput(
        frames={"frame": add_order_flow_features(deps["order_flow_merge"].frames["frame"])}
    ),
    cache_policy=CachePolicy(materialize=False),
    upstream_nodes=["order_flow_merge"],
)
```

This makes the graph:

```
raw_input → normalize → atr → ... → regime
                                        ↘
                          order_flow_input → order_flow_merge → order_flow_features
```

The `order_flow_input` source node receives its data from
`GraphRunContext.inputs["order_flow_input"]`, which the scanner populates
from the NT buffer. When NT data is absent, the input is an empty
DataFrame and the merge is a no-op.

---

## 12. Step 8 — Validation & Parity Testing

### 12.1 Parity: Pipeline With vs Without Order Flow

The existing indicator columns must be identical regardless of whether
NT data is merged. This is guaranteed by the architecture (merge happens
after the DAG), but should be verified:

```python
def test_pipeline_parity_with_order_flow():
    """of_* columns don't alter existing indicator values."""
    raw = load_test_candles("XAU_USD", "H1")

    # Without order flow.
    base = build_live_indicators(raw, instrument="XAU_USD")

    # With order flow (synthetic).
    of_frame = _make_synthetic_order_flow(raw)
    merged = base.merge(of_frame, on="timestamp", how="left")
    with_of = add_order_flow_features(merged)

    # All original columns must be identical.
    for col in base.columns:
        pd.testing.assert_series_equal(
            base[col], with_of[col], check_names=False,
        )
```

### 12.2 Order Flow Feature Validation

```python
def test_delta_divergence_detection():
    """ofi_delta_divergence fires when price and delta disagree."""
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],  # Trending up
        "of_bar_delta": [500, 400, 300, 200, 100],  # Delta weakening
        "of_bid_vol": [250, 300, 350, 400, 450],
        "of_ask_vol": [750, 700, 650, 600, 550],
        "of_buy_imbalances": [3, 2, 2, 1, 1],
        "of_sell_imbalances": [1, 2, 2, 3, 3],
        "of_footprint_levels": [20, 20, 20, 20, 20],
        "of_cum_delta": [500, 900, 1200, 1400, 1500],
        "of_cum_delta_high": [550, 950, 1250, 1450, 1550],
        "of_cum_delta_low": [450, 850, 1150, 1350, 1450],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "atr_14": [2.0] * 5,
    })
    result = add_order_flow_features(df)
    # Bars 1-4: price up + delta weakening → bearish divergence (-1).
    assert (result["ofi_delta_divergence"].iloc[1:] == -1).all()


def test_graceful_degradation():
    """Pipeline returns frame unchanged when NT data is absent."""
    df = pd.DataFrame({
        "close": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
    })
    result = add_order_flow_features(df)
    assert "ofi_delta_ma" not in result.columns  # No of_* → no ofi_*
    pd.testing.assert_frame_equal(df, result)
```

### 12.3 Volume Profile Accuracy Test

```python
def test_footprint_vp_vs_proxy_vp():
    """Footprint VP should be more concentrated than proxy VP."""
    from src.ingest.nt_socket_client import FootprintLevel

    # Simulate footprint with known distribution.
    footprint_bars = []
    for _ in range(80):
        levels = [
            FootprintLevel(price=100.0 + i * 0.1, bid_vol=10, ask_vol=10)
            for i in range(20)
        ]
        # Concentrate volume at 101.0 (the real POC).
        levels[10] = FootprintLevel(price=101.0, bid_vol=500, ask_vol=500)
        footprint_bars.append(levels)

    result = compute_volume_profile_from_footprint(footprint_bars)
    assert result["poc"] == 101.0
    assert result["val"] <= 101.0 <= result["vah"]
```

### 12.4 Validation Script

Add `scripts/validate_order_flow.py` following the pattern of existing
validation scripts:

```python
"""
Validate order flow integration.

Usage:
    python -m scripts.validate_order_flow \
        --data data/raw/XAU_USD_H1.parquet \
        --of-data data/order_flow/XAU_USD_H1.parquet
"""
# Checks:
# 1. of_* columns are present after merge.
# 2. ofi_* derived columns are computed.
# 3. No NaN in of_* columns where NT data exists.
# 4. ofi_delta_divergence has valid values (-1, 0, 1).
# 5. ofi_absorption_flag is binary.
# 6. Pipeline parity: existing columns unchanged.
# 7. DOM features have valid ranges.
```

---

## 13. Column Reference

### Raw Columns from NT (of_* prefix)

Produced by `OrderFlowBuffer.to_bar_dataframe()`. Present after merge.

| Column | Type | Description |
|--------|------|-------------|
| `of_bar_delta` | float | Ask volume − bid volume for the bar |
| `of_bid_vol` | float | Total bid-transacted volume |
| `of_ask_vol` | float | Total ask-transacted volume |
| `of_cum_delta` | float | Session cumulative delta (close) |
| `of_cum_delta_high` | float | Intra-bar cumulative delta high |
| `of_cum_delta_low` | float | Intra-bar cumulative delta low |
| `of_delta_change` | float | Change in cum delta from previous bar |
| `of_buy_imbalances` | int | Price levels where ask/bid > 3× |
| `of_sell_imbalances` | int | Price levels where bid/ask > 3× |
| `of_footprint_levels` | int | Number of traded price levels in bar |
| `of_poc_price` | float | Footprint POC (price with max volume) |

### Derived Columns (ofi_* prefix)

Produced by `add_order_flow_features()`.

| Column | Type | Description |
|--------|------|-------------|
| `ofi_delta_ma` | float | 10-bar MA of bar delta |
| `ofi_delta_zscore` | float | Z-score of bar delta (50-bar window) |
| `ofi_cum_delta_slope` | float | OLS slope of cum delta over 10 bars |
| `ofi_cum_delta_range` | float | Intra-bar cum delta range (high − low) |
| `ofi_delta_divergence` | int | −1 = bearish, 0 = neutral, 1 = bullish |
| `ofi_buy_pct` | float | Ask volume / total volume (0–1) |
| `ofi_imbalance_strength` | float | Imbalanced levels / total levels |
| `ofi_imbalance_bias` | float | (buy − sell imbalances) / total (−1 to 1) |
| `ofi_imbalance_ema` | float | EMA(14) of imbalance bias |
| `ofi_absorption_flag` | int | 1 if high volume + small range |
| `ofi_poc_dist_atr` | float | (close − footprint POC) / ATR |

### DOM Columns (dom_* prefix)

Produced by `OrderFlowBuffer.get_dom_features()`. Attached to latest bar only.

| Column | Type | Description |
|--------|------|-------------|
| `dom_bid_depth` | float | Total resting bid volume |
| `dom_ask_depth` | float | Total resting ask volume |
| `dom_imbalance_ratio` | float | (bid − ask) / total (−1 to 1) |
| `dom_levels_bid` | int | Number of bid price levels |
| `dom_levels_ask` | int | Number of ask price levels |
| `dom_best_bid_vol` | float | Volume at best bid |
| `dom_best_ask_vol` | float | Volume at best ask |
| `dom_spread_levels` | float | Best ask − best bid (in price) |

### SMC Confirmation Columns

| Column | Type | Description |
|--------|------|-------------|
| `ofi_ob_bull_confirmed` | int | OB + absorption delta confirmed |
| `ofi_ob_bear_confirmed` | int | OB + absorption delta confirmed |
| `dom_fvg_bull_strength` | float | DOM defense ratio at bull FVG boundary |
| `dom_fvg_bear_strength` | float | DOM defense ratio at bear FVG boundary |
| `ofi_smt_{partner}_{side}_delta_confirm` | int | SMT + cum delta triple divergence |

---

## 14. Constraints & Limitations

### Hard Constraints

1. **Forex spot has no real order flow.** BidAsk delta classification
   requires a centralized auction (exchange). Spot FX (OANDA) has no
   exchange — bid/ask attribution is undefined. You must use futures
   equivalents (6E, 6J, 6B, GC, CL) for real order flow. The pipeline
   uses OANDA spot for OHLCV and maps futures order flow onto it via
   timestamp alignment.

2. **NinjaTrader uses C# 5.0 / .NET 4.8.** No Python interop inside NT.
   All communication is via the TCP socket. The NinjaScript addon must be
   self-contained C#.

3. **Volumetric bars require bid/ask tick data.** Your NT data provider
   (e.g., Kinetick, Rithmic, CQG) must supply historical bid/ask ticks.
   Not all providers do. Without this, you get DOM data but no footprint
   or cumulative delta.

4. **NT Volume Profile Value Area is not in the API.** NT's built-in
   Volume Profile indicator renders visually but does not expose POC, VAH,
   VAL programmatically. This is irrelevant — we compute VP ourselves from
   the raw footprint data, which is exposed.

### Soft Constraints

5. **Futures/spot price divergence.** GC and XAU_USD differ by a small
   contango spread (~0.1–0.5%). Order flow features (delta, imbalance
   ratios) are dimensionless or volume-based, so the price difference
   doesn't affect them. The footprint POC price will be in futures terms
   — if you use it as a level, adjust by the contango spread or ignore
   the sub-0.5% error.

6. **Session hour gaps.** Futures have a daily settlement break (typically
   17:00–18:00 ET for CME metals). Cumulative delta resets at session
   start. Spot OANDA is continuous. Bars during the futures break will
   have no order flow data (NaN in `of_*` columns).

7. **Contract rollover noise.** On rollover dates, volume migrates from
   the expiring to the new front-month contract. Cumulative delta may
   show artifacts. Consider resetting delta-based features on rollover
   dates. Alternatively, use continuous contract data if your provider
   supports it.

8. **DOM is a snapshot, not a time series.** Only the latest DOM state is
   stored (not historical). DOM features are meaningful only for the most
   recent bar. Historical backtest cannot use DOM features.

---

## 15. Execution Roadmap

### Phase 1: Socket Bridge (Week 1–2)

| Task | Deliverable | Priority |
|------|------------|----------|
| Write `NTOrderFlowServer.cs` | NT indicator that streams bar + DOM JSON | P0 |
| Write `src/ingest/nt_socket_client.py` | Python async TCP client + buffers | P0 |
| Write `src/ingest/nt_symbol_map.py` | Futures → spot mapping config | P0 |
| Manual test: connect, verify JSON messages | Confirm data flows | P0 |

### Phase 2: Pipeline Merge (Week 2–3)

| Task | Deliverable | Priority |
|------|------------|----------|
| Write `src/scanner/live_scanner.py` scaffold | Orchestration loop | P0 |
| Implement `_align_timestamps` merge | of_* columns on candle frame | P0 |
| Write `src/indicators/foundation/order_flow.py` | ofi_* derived features | P1 |
| Parity test: existing indicators unchanged | Test suite | P1 |

### Phase 3: Volume Profile Upgrade (Week 3)

| Task | Deliverable | Priority |
|------|------------|----------|
| Add `compute_volume_profile_from_footprint()` | Exact VP from footprint | P1 |
| Wire into scanner: overwrite `vp_poc/vah/val` on live bar | Integration | P1 |
| Accuracy test vs proxy VP | Validation | P2 |

### Phase 4: SMC Enhancement (Week 3–4)

| Task | Deliverable | Priority |
|------|------------|----------|
| OB delta confirmation (`ofi_ob_*_confirmed`) | New feature | P2 |
| FVG DOM strength (`dom_fvg_*_strength`) | New feature | P2 |
| SMT + delta triple divergence | New feature | P2 |
| Displacement delta confirmation | New feature | P2 |

### Phase 5: DAG Integration (Week 4–5)

| Task | Deliverable | Priority |
|------|------------|----------|
| Add `order_flow_input` source node | DAG source | P2 |
| Add `order_flow_merge` compute node | DAG merge | P2 |
| Add `order_flow_features` compute node | DAG compute | P2 |
| Fingerprint + cache for order flow | DAG caching | P3 |

### Phase 6: Hardening (Ongoing)

| Task | Deliverable | Priority |
|------|------------|----------|
| Reconnection backoff + circuit breaker | Resilience | P2 |
| Contract rollover detection + delta reset | Data quality | P2 |
| `scripts/validate_order_flow.py` | Full validation | P2 |
| Profiler coverage for order flow merge | Observability | P3 |
| DOM history buffer (optional ring for N snapshots) | Research | P3 |
