# Data Engineering: Architecture, Protocols & Performance

This document defines the data engineering standards, architecture decisions,
and operational protocols for the trading system's indicator pipeline. It is
written to the same level of rigor expected at a senior (L5+) production
systems role: every section states the *what*, the *why*, the design
constraints, the current implementation status, and the concrete next actions.

**Sections:**
1. [Production Live Data Architecture](#production-live-data-architecture)
2. [DAG Orchestration](#dag-orchestration)
3. [Scheduler Strategy](#scheduler-strategy)
4. [Incremental Processing](#incremental-processing)
5. [Storage Tiering](#storage-tiering)
6. [Data Quality & Contracts](#data-quality--contracts)
7. [Batch vs Streaming — Research & Live Parity](#batch-vs-streaming--research--live-parity)
8. [Feature Store](#feature-store)
9. [Why This Work Matters For The Scanner And Live Retrieval](#why-this-work-matters-for-the-scanner-and-live-retrieval)
10. [Performance Audit Punch List](#performance-audit-punch-list) *(original 12-item audit)*

---

# Production Live Data Architecture

## Problem Statement

When the live scanner runs, new candle data arrives continuously. The system
must ingest new bars, compute indicators incrementally, and serve features to
the signal generator — all without corrupting the historical dataset that
research and backtesting depend on.

## Anti-Pattern: Write-Back Into Raw

```
broker API → new bars → append to data/raw/XAU_USD_H1.parquet → run pipeline
```

This is dangerous because:
- A single corrupted write (partial bar, duplicate timestamp, bad price)
  poisons the historical dataset permanently.
- Research results become non-reproducible — the raw file changes under you.
- If the live process crashes mid-write, pyarrow may leave a truncated parquet
  file. There is no atomic append for parquet.
- Rolling back requires manual intervention or backup restoration.

## Production Pattern: Separate Live Buffer

```
                    ┌─────────────────────────────────────────────────────┐
  broker API ──────►│  data/live/XAU_USD_H1/  (append-only ring buffer)  │
                    │  2026-04-22T14:00.parquet                          │
                    │  2026-04-22T15:00.parquet                          │
                    │  ...                                               │
                    └──────────────────┬──────────────────────────────────┘
                                       │
                                       │  concat(historical_raw, live_buffer)
                                       ▼
                    ┌──────────────────────────────────────────────────────┐
                    │  materialize_live_features(combined_df, ...)        │
                    │  → incremental plan (noop / incremental / full)     │
                    │  → persist to data/features/live/{symbol}/{tf}/     │
                    └──────────────────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼─────────────────────────────┐
              ▼                        ▼                             ▼
      signal generator          live dashboard              nightly promotion
      (reads features)         (reads features)          (live → raw, validated)
```

### Key Principles

| Principle | Implementation |
|-----------|---------------|
| **Raw is immutable** | `data/raw/` is never written to by the live process. Historical parquets are read-only during scanning. |
| **Live buffer is append-only** | Each new bar is written as a single small parquet file to `data/live/{symbol}_{tf}/`. File name is the bar's ISO timestamp. |
| **Pipeline input is concat** | `combined = pd.concat([historical_raw, live_buffer]).drop_duplicates("timestamp", keep="last")`. The live buffer overlaps the last N historical bars to handle bar updates (broker revises volume on close). |
| **Incremental plan decides work** | `resolve_incremental_plan` compares `input_fingerprint` + `last_processed_ts` against metadata. If only the frontier moved, it replays from `replay_from_ts`. If nothing moved, noop. |
| **Nightly promotion** | A scheduled job (cron or manual) reads the validated live buffer, runs quality checks, and appends to `data/raw/`. This is the only write path to raw. |
| **Ring buffer TTL** | Live buffer files older than 7 days are pruned after successful promotion. This caps disk growth. |

### Implementation Sketch

```python
# live_scanner.py (simplified)
def ingest_bar(bar: dict, symbol: str, timeframe: str) -> None:
    """Write a single bar to the live buffer."""
    ts = pd.Timestamp(bar["timestamp"], tz="UTC")
    buf_dir = Path(f"data/live/{symbol}_{timeframe}")
    buf_dir.mkdir(parents=True, exist_ok=True)
    path = buf_dir / f"{ts.isoformat()}.parquet"
    pd.DataFrame([bar]).to_parquet(path, index=False)

def scan_cycle(symbol: str, timeframe: str) -> pd.DataFrame:
    """Run one scan cycle: load, concat, compute, serve."""
    historical = pd.read_parquet(f"data/raw/{symbol}_{timeframe}.parquet")
    live_files = sorted(Path(f"data/live/{symbol}_{timeframe}").glob("*.parquet"))
    if live_files:
        live_buf = pd.concat([pd.read_parquet(f) for f in live_files])
        combined = pd.concat([historical, live_buf]).drop_duplicates(
            "timestamp", keep="last"
        )
    else:
        combined = historical

    result = materialize_live_features(
        combined,
        instrument=symbol,
        timeframe=timeframe,
        include_cross_asset=True,
        features_root="data/features",
    )
    return result.frame  # latest feature row → signal generator

def promote_live_to_raw(symbol: str, timeframe: str) -> None:
    """Nightly: validate and merge live buffer into raw."""
    raw_path = Path(f"data/raw/{symbol}_{timeframe}.parquet")
    raw = pd.read_parquet(raw_path)
    live_buf = _load_live_buffer(symbol, timeframe)

    # Quality gate
    assert_no_duplicate_timestamps(live_buf)
    assert_monotonic_timestamps(live_buf)
    assert_ohlc_valid(live_buf)

    merged = pd.concat([raw, live_buf]).drop_duplicates("timestamp", keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    write_parquet_atomic(merged, raw_path)

    _prune_promoted_live_files(symbol, timeframe)
```

### Why Not Just Use the Existing `materialize_live_features`?

`materialize_live_features` already handles incremental computation and
persistence of the *feature* layer. The gap is the *raw data ingestion*
layer — there is currently no code that manages the live buffer or the
promotion cycle. The pipeline assumes raw data already exists as a single
complete parquet file. The architecture above fills that gap.

---

# DAG Orchestration

## What It Is

A Directed Acyclic Graph (DAG) orchestrator manages the execution order of
pipeline stages, tracks dependencies between stages, enables caching at node
boundaries, and supports parallel execution of independent branches.

In production data systems (Airflow, Prefect, Dagster, dbt), DAGs are the
standard unit of orchestration. Each node declares its inputs and outputs,
and the orchestrator handles scheduling, retries, caching, and lineage.

## Current State

The system has a DAG runtime (`src/dag_runtime/`) with:
- `StageGraph`: Defines nodes and edges.
- `execute_graph`: Walks the graph, executing nodes in topological order.
- `CachePolicy`: Per-node configuration for materialization.
- `cache_store.py`: Full save/load infrastructure for node outputs.

**Problem:** The DAG is only partially realized.
- Node caching is active only for the proven-safe subset: Class A nodes,
  `swings`, and non-carried `rsi_divergence`.
- Carried-state Class B nodes remain `materialize=False` after a reproduced
  persistence parity regression on the wider cache rollout.
- Live and research cross-asset execution are now DAG-owned, but validator
  migration and scheduler adoption are still incomplete.

## Target Architecture

```
                         raw_input
                             │
                     normalize_candles
                             │
                   ┌─────────┼─────────┐
                   ▼         ▼         ▼
                  atr       ema      body_ratio
                   │         │         │
                   └─────┬───┘         │
                         │             │
                   ┌─────┼─────────────┘
                   ▼     ▼
                 swings (Class B, cached)
                   │
          ┌────────┼────────────────────────────┐
          ▼        ▼                            ▼
    trend_state  bos/choch               market_context_source
          │        │                            │
          ▼        ▼                     ┌──────┼──────┐
      fvg_stack  ob/mitigation           ▼      ▼      ▼
          │        │                partner_DXY  partner_JPY  ...
          └────┬───┘                     │      │      │
               │                         └──────┼──────┘
               ▼                                ▼
            regime                    smt_divergence (merge node)
               │                                │
               └────────────┬───────────────────┘
                            ▼
                  attach_cross_asset_context
                            │
                       final_frame
```

## Protocol

### P1 — Enable Node Caching

1. Set `materialize=True` on all Class A (stateless) nodes: `atr`, `ema`,
   `adx`, `rsi`, `macd`, `bb_width`, `body_ratio`, `rolling_atr_ratio`,
   `volume_features`, `prev_day_hl`, `prev_week_hl`, `round_number_flag`,
   `intraday_context`.
2. Set `materialize=True` on `swings` — the most expensive Class B node
   and the root dependency for all structural indicators.
3. Node fingerprint = `sha256(input_fingerprint + node_config_json)`.
4. Verify: on unchanged re-run, every materialized node should report a
   cache hit. Zero recomputation.
5. Keep `materialize=False` on remaining Class B nodes until parity tests
   confirm no drift from carried state.

### P2 — Add Branching

1. Identify independent subgraphs: `{atr, ema, adx, rsi, macd}` can run
   in parallel (they all depend only on `normalize_candles`).
2. Partner pipeline builds are independent of each other and of the
   primary indicator chain. Model them as branch nodes converging at
   `smt_divergence`.
3. `market_context` becomes a source node injected into the graph, giving
   it fingerprint-based invalidation.

### P3 — Move Cross-Asset Into the Graph

1. `build_global_market_context` becomes a graph node.
2. Each partner build becomes a sub-graph node.
3. `attach_cross_asset_context` becomes the final merge node.
4. The profiler and cache now cover the full pipeline end to end.

## What to Study

- **Airflow** / **Dagster**: Production DAG orchestrators. Understand
  operators, sensors, pools, retries, SLAs.
- **dbt**: SQL-layer DAG with built-in ref() lineage and incremental
  materialization. The mental model transfers directly to DataFrame
  pipelines.
- **Execution models**: Push-based (Airflow) vs pull-based (Spark lazy
  evaluation). This system is push-based today.

---

# Scheduler Strategy

## Problem Statement

The DAG runtime is now broad enough that serial execution is no longer the
right universal default for performance-sensitive paths. However, a scheduler
rollout can easily make the system worse if it is treated as "turn on parallel
everywhere":

- CPU can spike hard and starve the live scanner.
- concurrent parquet writes can increase disk contention and tail latency.
- graphs with mostly serial dependency chains gain little from extra workers.
- validator graphs can become harder to debug before topology and cache
  behavior are frozen.

The correct design is therefore:

- **global capability**
- **bounded resource usage**
- **graph-level opt-in**
- **serial by default until proven**

## Current State

The runtime already exposes the core contract in `src/dag_runtime/`:

- `ExecutionPolicy`
- `scheduler_mode = "serial" | "bounded_parallel"`
- `max_workers`
- `max_concurrent_cache_writes`

The executor can already run a bounded thread pool for ready sibling nodes.
That means the missing work is no longer the primitive itself; the missing work
is the **engineering policy** for where it should be enabled, with what limits,
and under what test gates.

## Architecture Decision

The scheduler should be **global in the DAG runtime**, not custom-built inside
individual indicators or scripts.

Why:
- one scheduling model is easier to reason about than many ad hoc ones
- cache-write throttling belongs in the runtime, not in indicator code
- profiler semantics should be uniform across research, live, and validators
- future DAG-backed graphs should inherit the same capability automatically

What "global" means here:
- every DAG-backed graph may use the scheduler
- no graph is forced to use it
- the default policy remains `serial`
- opt-in happens at the graph family level after parity and resource tests

## Default Resource Policy

For production use, the scheduler goal is **bounded usage**, not maximum
throughput.

Default policy for graphs that opt in:

| Setting | Default |
|---------|---------|
| `scheduler_mode` | `bounded_parallel` |
| `max_workers` | `min(4, os.cpu_count() or 1)` |
| `max_concurrent_cache_writes` | `1` |

Hard rules:
- cache writes stay serialized
- explain-only runs stay serial
- cache loads stay on the main thread
- validator graphs stay serial until they pass migration and warm-cache gates

This protects the machine from turning into a benchmark box that crushes the
scanner, browser, notebook, and file system while one graph is running.

## Which Indicators / Graph Families Should Use It

This is the most important decision. The scheduler should follow **graph
shape**, not popularity. Parallel scheduling helps when there is meaningful
width. It does very little when one dominant node owns most of the wall time.

### Enable First

#### 1. Research pipeline cross-asset branches

Use bounded scheduling on:
- `research_market_context_source`
- `research_partner_<symbol>` sibling nodes
- `research_smt_research_table`
- downstream bundle target selection

Why:
- research cross-asset now has the widest independent branch structure
- partner nodes are naturally parallel siblings
- research runs are the least latency-sensitive and the best place to prove
  scheduler correctness before enabling it for live

Expected gain:
- lower wall time on audit-capable research materializations
- no semantic change
- CPU rises, but in a bounded, controlled way

#### 2. Live pipeline cross-asset branches

Use bounded scheduling on:
- `live_market_context_source`
- `live_partner_<symbol>` sibling nodes
- `live_feature_bundle` closure after partner completion

Why:
- live cross-asset is the highest-value production target after research
- partner builds are independent and should not block each other serially
- the scheduler can reduce live feature latency without changing indicator math

Expected gain:
- lower end-to-end scan-cycle wall time
- narrower latency spikes during cross-asset attachment
- faster availability of the latest live-safe feature row

### Enable Selectively After That

#### 3. Foundation indicator sibling stages in production graphs

These stages are good candidates because they fan out from normalized candles
and do not depend on one another:

- `atr`
- `ema`
- `adx`
- `rsi`
- `macd`
- `bb_width`
- `body_ratio`
- `rolling_atr_ratio`
- `volume_features`
- `prev_day_hl`
- `prev_week_hl`
- `round_number_flag`
- `intraday_context`
- `volume_profile` when included

Why:
- this is the cleanest sibling-stage fan-out in the system
- each node is already conceptually independent after `normalize_candles`
- these are the least risky places to harvest extra parallel width

Constraint:
- do this only after cache-hit behavior and replay boundaries remain stable
  under the scheduler

### Keep Serial For Now

#### 4. Structural carried-state chain

Do **not** expect major scheduler wins from:
- `swings`
- `trend_state`
- `bos`
- `choch`

Why:
- these are mostly a dependency chain, not a wide branch
- correctness and replay parity matter more here than concurrency
- if one stage depends directly on the prior stage, the scheduler has little to
  parallelize

#### 5. SMC chain

Keep serial by default:
- `fvg_stack`
- `displacement`
- `order_blocks`
- `ob_mitigation`
- `equal_hl`
- `amd_engine`

Why:
- these stages are also mostly serially dependent
- cache and replay correctness are the bigger wins than parallel threads
- parallel execution here adds complexity with limited width benefit

#### 6. `range_boundaries`

Keep the main compute target serial by default:
- `range_context`
- `range_rung_debug__*`
- `range_selected_debug`

Why:
- the known dominant cost is still the heavy internal compute path
- the graph does not have enough broad sibling width to justify making the
  runtime more complex here first
- scheduler work would not solve the main cold-path bottleneck

What may later use the scheduler:
- chart-pack nodes
- CSV bundle nodes
- memo/report nodes

But only after the heavy compute target is already warm and only if write
throttling keeps disk contention under control.

## Rollout Order

### Phase S1 — Research only

Enable bounded scheduling on the research pipeline first.

Acceptance:
- serial vs bounded-parallel outputs are bit-identical
- cache-hit/miss behavior is unchanged
- profiler records scheduler metrics
- cache writes remain serialized
- no increase in validation drift or replay-window failures

### Phase S2 — Live cross-asset only

Enable bounded scheduling on the live pipeline cross-asset nodes only.

Acceptance:
- no change in live output parity
- scan-cycle wall time improves on multi-partner instruments
- CPU stays bounded by the worker cap
- latest-row availability improves without starving the rest of the process

### Phase S3 — Foundation sibling fan-out

Enable the scheduler for the stateless foundation siblings after
`normalize_candles`.

Acceptance:
- no parity drift
- no unexpected cache-write amplification
- real wall-time improvement on cold or semi-cold runs

### Phase S4 — Validator opt-in, one family at a time

Only after the runtime is stable:
- `validate_structure_context`
- `validate_swings`
- `validate_bos`
- `validate_bos_context`
- `validate_choch`
- `validate_choch_context`

Do **not** enable bounded scheduling for `range_boundaries` main compute by
default in this phase.

## Test Protocol

Every graph family that opts into the scheduler must pass:

1. serial vs bounded-parallel output equality
2. deterministic node-result ordering
3. cache-hit behavior equality
4. one-failure cancellation behavior
5. cache-write throttle behavior
6. profiler emission:
   - `scheduler_mode`
   - `worker_count`
   - runnable-queue wait time
   - node execution time
   - cache-write wait time

No graph should switch to bounded scheduling just because it "sounds faster."
It must show:
- lower wall time on representative runs
- acceptable CPU behavior
- no disk thrash
- unchanged outputs

## Why This Is Not Just A Performance Toy

The scheduler is not only about shaving seconds from validation commands.
Used correctly, it improves the operating characteristics of the whole system:

- more predictable scan-cycle latency
- less time spent blocking on independent partner builds
- better use of available CPU without pegging all cores
- safer, centralized resource policy
- a single place to control concurrency as the platform grows

That makes it relevant to the real product path, not just engineering comfort.

---

# Incremental Processing

## What It Is

Incremental processing means computing only the *delta* — new or changed
data — instead of reprocessing the full dataset. It is the single most
important performance technique for any pipeline that runs repeatedly on
growing data.

## Current Implementation

The system already has a working incremental processing layer in
`src/pipeline_runtime/incremental.py`:

```
resolve_incremental_plan(df, metadata, ...)
    │
    ├─ metadata is None           → mode="full", reason="no-prior-metadata"
    ├─ schema_version mismatch    → mode="full", reason="schema-version-changed"
    ├─ contract_version mismatch  → mode="full", reason="contract-version-changed"
    ├─ force_rebuild=True         → mode="full", reason="force-rebuild"
    ├─ input unchanged + ts covered → mode="noop", is_noop=True
    └─ new data at frontier       → mode="incremental", replay_from_ts=...
```

**How replay works:**
- Each stage declares a `ReplayPolicy` with `replay_bars` and optionally
  `carried_state`.
- The plan computes `replay_from_ts` as the earliest timestamp that
  satisfies the maximum replay window across all stages.
- `slice_frame_for_plan` extracts the working slice from the normalized
  input.
- After computation, `merge_recomputed_frontier` stitches the recomputed
  frontier onto the existing history.

**What's missing:**
- Cross-asset market context is always recomputed from scratch (no
  incremental). See Performance Audit item 8.
- The early noop check (added in research pipeline) is not yet in the
  live pipeline's `run_live_pipeline`. The live pipeline falls through
  to cross-asset resolution before checking for noop.
- No incremental for the correlation audit matrix (1.6M rows rebuilt
  from scratch on every materialization).

## Protocol

### Fingerprint Contract

Every pipeline run produces a `PipelineMetadata` record:

| Field | Purpose |
|-------|---------|
| `input_fingerprint` | Content hash of the normalized raw input. Change = new data arrived. |
| `config_fingerprint` | Content hash of pipeline config (instrument, timeframe, swing_window, cross-asset fingerprints). Change = pipeline parameters changed. |
| `schema_version` | Integer. Bump = output schema changed (columns added/removed). Forces full rebuild. |
| `feature_contract_version` | Integer. Bump = feature semantics changed (same columns, different computation). Forces full rebuild. |
| `last_processed_ts` | ISO timestamp of the last bar processed. Used for noop detection. |
| `engine_version` | String tag identifying the pipeline variant (research-v1, live-v1). |

### Decision Matrix

| input_fp match | config_fp match | ts coverage | → Mode |
|:-:|:-:|:-:|---|
| yes | yes | yes | **noop** — return cached history, zero work |
| yes | no | — | **full** — config changed, recompute everything |
| no | yes | yes | **noop** — data hash changed but no new bars (rounding) |
| no | — | no | **incremental** — new frontier bars, replay from `replay_from_ts` |

### Replay Safety for Stateful Stages

Class B stages (swings, trend_state, bos, choch, fvg, ob, etc.) carry
state across bars. Their `ReplayPolicy` declares how many bars to replay
from the frontier to re-derive correct state:

- `swings`: `max(400, swing_window * 30)` bars
- `trend_state`, `bos`, `choch`: 400 bars
- `fvg_stack`, `displacement`, `ob`, `ob_mitigation`: 300 bars

The maximum across all stages determines the global `replay_from_ts`.
This guarantees that even the most state-sensitive stage has enough
history to warm up correctly.

### Target: Incremental Market Context

When `build_global_market_context` runs incrementally:
1. Load prior market context from `data/features/market_context_{variant}/`.
2. Identify new bars since `last_processed_ts` in each context symbol.
3. Extend the raw context frames by the new bars only.
4. Recompute rolling correlations for the last `max_horizon` bars
   (the frontier).
5. Stitch the recomputed frontier onto the historical market context.
6. Persist the updated market context.

This reduces the O(n_symbols² × n_horizons × n_bars) computation to
O(n_symbols² × n_horizons × max_horizon) on each incremental run.

---

# Storage Tiering

## What It Is

Storage tiering separates data by access pattern, freshness, and
criticality into distinct layers with different performance
characteristics, retention policies, and write semantics.

## Current Layout

```
data/
├── raw/                          # Tier 0: Source of truth
│   ├── XAU_USD_H1.parquet
│   ├── XAU_USD_H4.parquet
│   ├── DXY_H1.parquet
│   └── ...                       # 11 symbols × 2 timeframes
├── features/                     # Tier 1: Computed features (production)
│   ├── research/{symbol}/{tf}/   # Monthly partitioned parquets
│   ├── live/{symbol}/{tf}/
│   ├── market_context_research/
│   ├── market_context_live/
│   ├── research_smt/
│   ├── research_cross_asset_audit/
│   ├── _state/                   # Pipeline metadata JSONs
│   └── _temp/                    # Atomic write staging area
├── validation_cache/             # Tier 1b: Validation-specific materialization
│   └── features/                 # Same structure as features/
├── dag_cache/                    # Tier 2: Node-level DAG cache (currently unused)
└── live/                         # Tier 0b: Live buffer (to be implemented)
    └── {symbol}_{tf}/            # Individual bar files
```

## Protocol

### Tier Definitions

| Tier | Path | Write Semantics | Retention | Backed Up |
|------|------|-----------------|-----------|-----------|
| **0 — Raw** | `data/raw/` | Immutable during operation. Only nightly promotion writes. | Permanent | Yes |
| **0b — Live Buffer** | `data/live/` | Append-only, one file per bar. | 7 days after promotion | No (ephemeral) |
| **1 — Features** | `data/features/` | Atomic writes via `write_parquet_atomic`. Monthly partitioned. Frontier-only writes on incremental runs. | Permanent (rebuildable from raw) | Optional |
| **1b — Validation Cache** | `data/validation_cache/` | Same as Tier 1 but for validation scripts. | Ephemeral (rebuildable) | No |
| **2 — DAG Cache** | `data/dag_cache/` | Per-node parquet files keyed by fingerprint. | Until invalidated | No |

### Write Semantics

All writes below Tier 0 use `write_parquet_atomic`:
1. Write to a temp file in `_temp/` with a UUID suffix.
2. `os.rename()` to the final path (atomic on POSIX).
3. `cleanup_temp_artifacts()` on pipeline start removes any orphaned
   temp files from prior crashes.

This guarantees that readers never see a partial parquet file.

### Partitioning Strategy

Feature datasets are partitioned by calendar month:
- Path: `{dataset}/{symbol}/{tf}/YYYY-MM.parquet`
- On incremental runs, only the frontier partition (and any later ones)
  are rewritten.
- Historical partitions are never touched after the month closes.
- `load_partitioned_dataset` globs all partitions and concatenates.

### Promotion Protocol (Live → Raw)

```
1. Load all live buffer files for (symbol, timeframe).
2. Validate: no duplicate timestamps, monotonic order, valid OHLC.
3. Merge with existing raw: concat + dedup on timestamp, keep="last".
4. Write merged raw via write_parquet_atomic.
5. Delete promoted live buffer files.
6. Touch a promotion log: data/live/_promoted/{symbol}_{tf}/YYYY-MM-DD.json
   with row counts, min/max timestamps, and validation checksums.
```

---

# Data Quality & Contracts

## What It Is

Data quality engineering ensures that every stage of the pipeline produces
output that conforms to a declared contract. When a contract is violated,
the pipeline halts or degrades gracefully rather than silently producing
garbage that corrupts downstream analysis.

## Current Implementation

- **Schema normalization**: `normalize_candle_schema` enforces column
  names, types, and the presence of required fields (timestamp, OHLC,
  volume) at pipeline entry.
- **Feature contract version**: `RESEARCH_FEATURE_CONTRACT_VERSION` and
  `LIVE_FEATURE_CONTRACT_VERSION` are integer constants. A bump forces
  full rebuild, preventing stale features from mixing with new
  computation logic.
- **Schema version**: `RESEARCH_SCHEMA_VERSION` / `LIVE_SCHEMA_VERSION`.
  Bump = output columns changed.
- **Validation scripts**: `scripts/validate_smt.py`,
  `scripts/validate_cross_asset.py` produce numeric summaries and HTML
  charts for human review.

## Protocol

### Contract Levels

| Level | What It Checks | When It Runs | On Failure |
|-------|---------------|--------------|------------|
| **L0 — Schema** | Columns exist, types correct, no nulls in primary keys (timestamp) | Pipeline entry (`normalize_candle_schema`) | Hard error, pipeline aborts |
| **L1 — Referential** | Foreign key integrity (e.g., SMT partner timestamps align with primary) | After cross-asset attachment | Hard error, pipeline aborts |
| **L2 — Statistical** | Value ranges (price > 0, indicators bounded, no inf), distribution checks | After each stage group (foundation, structure, SMC) | Warning + flag in metadata |
| **L3 — Semantic** | Business logic (SMT divergence requires opposite swing directions, BOS requires prior trend state) | Validation scripts | Report-only, human review |

### Schema Contract (L0)

Every pipeline entry point must call `normalize_candle_schema`:

```python
required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
required_types = {
    "open": float, "high": float, "low": float, "close": float,
    "volume": float, "timestamp": "datetime64[ns, UTC]"
}
```

Violations produce a hard `ValueError` with the specific missing columns
or type mismatches listed.

### Feature Contract Versioning

```python
# In build_research.py / build_live.py:
RESEARCH_FEATURE_CONTRACT_VERSION = 3   # Bump when computation logic changes
RESEARCH_SCHEMA_VERSION = 1             # Bump when output columns change
```

The incremental planner checks both:
- Schema version mismatch → full rebuild (columns may have changed).
- Feature contract version mismatch → full rebuild (same columns,
  different values).
- Neither changed + input unchanged → noop.

**Rule**: Every PR that changes indicator computation logic MUST bump the
feature contract version. Every PR that adds or removes output columns
MUST bump the schema version.

### Statistical Quality Gates (L2)

Target implementation for per-stage checks:

```python
def _validate_stage_output(frame: pd.DataFrame, stage_name: str) -> list[str]:
    """Return list of violations (empty = pass)."""
    violations = []
    if frame["close"].le(0).any():
        violations.append(f"{stage_name}: close <= 0 detected")
    if frame.select_dtypes(include="number").apply(lambda s: s.isin([float("inf"), float("-inf")])).any().any():
        violations.append(f"{stage_name}: infinite values detected")
    # Per-indicator bounds
    if "rsi_14" in frame.columns:
        if not frame["rsi_14"].dropna().between(0, 100).all():
            violations.append(f"{stage_name}: RSI outside [0, 100]")
    if "adx_14" in frame.columns:
        if not frame["adx_14"].dropna().ge(0).all():
            violations.append(f"{stage_name}: ADX < 0")
    return violations
```

### Circuit Breaker

If L2 violations exceed a threshold (e.g., >5% of rows have any
violation), the pipeline should:
1. Log the violations with full row-level detail.
2. Skip persistence — do not overwrite good cached features with bad ones.
3. Return the prior cached features with a degraded flag in metadata.
4. Alert (log to stderr, or write to a `_alerts/` directory).

This prevents a bad data ingestion (broker sends 0-price bars during
maintenance) from cascading through the entire feature store.

### Data Lineage

Every `PipelineMetadata` record already captures:
- Input fingerprint (what raw data produced this).
- Config fingerprint (what parameters were used).
- Engine version (which pipeline variant).
- Timestamps (when it was produced, what data it covers).

For full lineage, add:
- `parent_pipeline_run_id` for partner builds (link DXY features back
  to the DXY pipeline run that produced them).
- `dependency_fingerprints`: dict mapping each dependency name to its
  fingerprint (market context, each partner frame).

---

# Batch vs Streaming — Research & Live Parity

## What It Is

The system runs two pipeline variants that must produce identical
structural outputs for the same input data. This is a lambda architecture:
the batch (research) layer produces complete, retrospective features for
backtesting and training; the speed (live) layer produces causal,
point-in-time features for real-time decision making.

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │         Shared Causal Backbone       │
                     │  normalize → atr → ema → ... →      │
                     │  swings (causal) → trend_state →     │
                     │  bos → choch → ... → regime          │
                     └──────────────┬──────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────────┐       ┌─────────────────────┐
          │   Research Pipeline  │       │    Live Pipeline     │
          │   (build_research)   │       │    (build_live)      │
          │                     │       │                     │
          │ + include_avwap     │       │ - no avwap          │
          │ + research_only     │       │ - no research_only  │
          │   volume features   │       │   volume features   │
          │ + research_only     │       │ - no research_only  │
          │   session features  │       │   session features  │
          │ + research_only     │       │ - no research_only  │
          │   regime features   │       │   regime features   │
          │ + amd_labels        │       │ - no labels (ever)  │
          │ + smt_research_table│       │                     │
          │ + correlation_audit │       │                     │
          └─────────────────────┘       └─────────────────────┘
```

## Parity Contract

The two pipelines share the same stage functions, same order, same
`ReplayPolicy` values. The differences are:

| Aspect | Research | Live |
|--------|----------|------|
| Swing engine | `add_swings` (causal) | `add_swings` (causal) — **same** |
| `include_research_only` | True | False |
| AVWAP | Optional (`include_avwap`) | Never |
| AMD labels | `add_labels=False` in both | Same |
| Cross-asset | Full audit + research table | Context attachment only |
| Materialization dataset | `research/` | `live/` |
| Market context variant | `market_context_research/` | `market_context_live/` |

**Invariant**: For any input DataFrame `df`, the columns produced by both
pipelines with `include_research_only=False` and `include_avwap=False`
must be identical in name, order, and value. This is the *parity
invariant* — it guarantees that a model trained on research features
will see the same feature distribution in live.

## Protocol

### Parity Test

Run both pipelines on the same input and assert column-level equality:

```python
def test_research_live_parity(raw_df, instrument, timeframe):
    research = build_research_indicators(
        raw_df, instrument=instrument, include_avwap=False,
        timeframe=timeframe, include_cross_asset=True,
    )
    live = build_live_indicators(
        raw_df, instrument=instrument,
        timeframe=timeframe, include_cross_asset=True,
    )
    # Drop research-only columns
    research_only_cols = [c for c in research.columns if c not in live.columns]
    research_trimmed = research.drop(columns=research_only_cols)

    pd.testing.assert_frame_equal(
        research_trimmed.reset_index(drop=True),
        live.reset_index(drop=True),
        check_dtype=False,
        atol=1e-10,
    )
```

This test must pass in CI on every PR that touches any indicator stage.

### Feature Flag Discipline

- `include_research_only`: Controls whether research-only columns
  (extra session bins, regime sub-scores, volume internals) are computed.
  Never set to True in the live pipeline.
- `include_avwap`: Research-only toggle. AVWAP in live is computed
  per-signal by the scanner, not per-bar by the pipeline.
- `add_labels`: Always False in both pipelines. Labels are added by
  a separate labeling step in research, never baked into features.

### Version Lockstep

`RESEARCH_SCHEMA_VERSION` and `LIVE_SCHEMA_VERSION` should be bumped
together when shared stages change. If research schema is at version N
and live is at version M, a parity test failure is likely.

---

# Feature Store

## What It Is

A feature store is the serving layer between pipeline computation and
model consumption. It provides:
- **Versioned storage**: Features are persisted with metadata that
  identifies how and when they were produced.
- **Point-in-time correctness**: Historical queries return features as
  they existed at a given timestamp, not as currently computed.
- **Incremental updates**: Only the frontier is recomputed and merged.
- **Lineage**: Every feature row can be traced back to its raw input
  and pipeline configuration.

## Current Implementation

The system implements a file-based feature store using the
`pipeline_runtime` module:

### Storage

```
data/features/
├── research/XAU_USD/H1/
│   ├── 2025-01.parquet       # Monthly partition
│   ├── 2025-02.parquet
│   └── ...
├── _state/
│   └── build_research/XAU_USD/H1/metadata.json
│       {
│           "symbol": "XAU_USD",
│           "timeframe": "H1",
│           "pipeline": "build_research",
│           "last_processed_ts": "2026-04-22T13:00:00+00:00",
│           "schema_version": 1,
│           "feature_contract_version": 3,
│           "input_fingerprint": "abc123...",
│           "config_fingerprint": "def456...",
│           "engine_version": "research-v1",
│           "updated_at": "2026-04-22T14:30:00+00:00"
│       }
```

### Read Path

```python
features = load_partitioned_dataset(
    "data/features",
    dataset="research",
    symbol="XAU_USD",
    timeframe="H1",
)
# Returns: pd.DataFrame with all monthly partitions concatenated
```

### Write Path

```python
artifacts = persist_partitioned_dataset(
    frame,
    base_dir="data/features",
    dataset="research",
    symbol="XAU_USD",
    timeframe="H1",
    frontier_from_ts=plan.replay_from_ts,
    full_rebuild=False,  # only rewrite frontier partition
)
```

### Materialization Flow

```
materialize_research_features(raw_df, ...)
    │
    ├─ Read metadata from _state/
    ├─ Load existing history from partitioned dataset
    ├─ Run pipeline (noop / incremental / full)
    ├─ Persist updated partitions (frontier-only on incremental)
    ├─ Persist market context
    ├─ Persist research artifacts (SMT table, correlation audit)
    └─ Write updated metadata atomically
```

## Protocol

### Feature Registration

Every feature set must declare:

| Property | Example | Purpose |
|----------|---------|---------|
| `dataset` name | `"research"`, `"live"` | Namespace in the feature store |
| `schema_version` | `1` | Column set version |
| `feature_contract_version` | `3` | Computation semantics version |
| `engine_version` | `"research-v1"` | Pipeline variant that produced it |

### Serving Contract

Consumers (signal generator, backtester, notebooks) read features via
`load_partitioned_dataset`. They MUST NOT read raw data or compute
indicators directly. The feature store is the single serving interface.

### Backfill Protocol

When a feature contract version bumps (computation logic changed):
1. All existing feature partitions are stale.
2. Run `materialize_*_features` with `force_rebuild=True` for each
   (symbol, timeframe) pair.
3. The metadata records the new contract version.
4. Downstream consumers automatically get the recomputed features on
   their next read.

### Garbage Collection

Old partitions from before a full rebuild are overwritten in place.
There is no explicit GC step — `persist_partitioned_dataset` with
`full_rebuild=True` rewrites all partitions, and the atomic write
replaces the old files.

For the DAG cache (Tier 2), implement TTL-based eviction: delete
cache entries whose fingerprint hasn't been accessed in > 7 days.

---

# Why This Work Matters For The Scanner And Live Retrieval

## Short Answer

Yes. This data-engineering work is directly relevant to the scanner and to live
feature retrieval. It is not just "test infrastructure" work.

## What It Changes For The Scanner

The scanner ultimately cares about one thing: **how quickly and reliably can it
obtain the latest correct feature row and any required context without blowing
up the machine?**

That depends on exactly the systems described in this document:

- incremental planning
- DAG cache invalidation
- market-context reuse
- bounded scheduling
- partition persistence
- replay safety

If those layers are poor, the scanner suffers in several ways even if the
scanner code itself is untouched.

## Concrete Scanner Benefits

### 1. Lower scan-cycle latency

When the scanner ingests a new bar, it needs the newest live-safe feature row.
If the pipeline can:
- noop
- replay only the frontier
- reuse cross-asset context
- reuse warm DAG nodes
- parallelize independent partner builds in a bounded way

then the scanner gets the latest row faster.

This is not cosmetic. It directly affects how quickly a signal can be
evaluated after a bar close.

### 2. Better freshness and lower tail latency

The scanner does not only care about average speed. It cares about the worst
cases too. If one scan cycle occasionally explodes in wall time, the signal
loop becomes unreliable.

Data-engineering work reduces that risk by:
- narrowing recompute scope
- avoiding accidental full rebuilds
- preventing hidden audit work on live paths
- controlling concurrency instead of letting CPU usage spike arbitrarily

### 3. Better machine stability during live operation

If one pipeline run pegs CPU or floods disk writes, the scanner competes with:
- broker ingestion
- dashboard refresh
- notebooks
- validation scripts
- OS file cache

Bounded execution policy matters here because the best system is not the one
that occasionally finishes fastest; it is the one that keeps the whole machine
healthy while consistently delivering live features.

### 4. More reliable signal quality

This work is not just about speed. It also protects correctness:

- replay-window rules protect carried-state indicators from drift
- topological fingerprints prevent stale cached outputs
- source-hash invalidation means logic changes rebuild the correct closure
- graph-owned cross-asset nodes make dependencies explicit and auditable

That matters to the scanner because a fast but stale or drifted feature row is
worse than a slow one.

## Concrete Live Retrieval Benefits

"Live retrieval" here means any path that needs to fetch or serve the latest
usable features or supporting context, for example:
- scanner latest-row load
- live dashboard views
- downstream signal or agent context retrieval

This work helps live retrieval by:

- reducing how often the canonical live dataset must be rewritten
- making frontier persistence narrower
- making market-context reuse explicit and validatable
- keeping output contracts stable while internals become cheaper

In other words, retrieval gets better because production feature generation gets
cheaper, safer, and more predictable.

## What This Work Does *Not* Do By Itself

This work does **not** automatically improve:
- detector logic quality
- signal selection quality
- entry/exit logic
- position sizing
- scanner strategy ranking

Those are downstream consumers of better infrastructure.

So the right way to think about it is:

- data engineering improves **latency, stability, correctness, and operating
  cost**
- strategy logic improves **trading quality**

You need both. One does not replace the other.

## Why It Was Worth Doing Even In MVP1

Even in MVP1, this work has real product value because it:
- keeps iteration speed sane during heavy testing
- prevents the scanner/live path from inheriting obviously wasteful behavior
- hardens the system before more features and strategies are added
- avoids scaling a bad architecture into a worse production problem later

The key point is:

This work is not only useful for development ergonomics. It is foundational
runtime engineering for the actual live system.

---

# Performance Audit Punch List

This section catalogs every caching and performance problem identified in the
indicator pipeline, along with concrete fixes and their priority. It is written
as a punch list — each item has a diagnosis, impact, and remediation.

---

## 1. DAG Node Cache Is Partially Active

**Diagnosis.** Node-level caching is no longer dormant, but it is intentionally
limited to the safe subset. Class A nodes, `swings`, and non-carried
`rsi_divergence` materialize. Carried-state Class B nodes were rolled back out
of the whitelist after a reproduced persistence parity failure.

**Impact.** Direct graph execution and builder reuse now benefit from safe
node-cache hits, but carried-state tails still recompute. This preserves
correctness at the cost of leaving some warm-rerun savings on the table until a
replay-context parity gate exists.

**Remediation.**
- Keep `materialize=True` on Class A nodes, `swings`, and non-carried
  `rsi_divergence`.
- Keep `materialize=False` on carried-state Class B nodes until node
  fingerprints incorporate replay-context safety strongly enough to preserve
  persistence parity.
- Treat `tests/test_pipeline_persistence.py` as a hard gate before any future
  re-expansion of the cache whitelist.

**Priority: P1.**

---

## 2. The DAG Is a Linear Chain, Not a Graph

**Diagnosis.** `build_research_stage_graph` constructs a strict linear chain:
`raw → normalize → atr → ema → adx → ... → regime`. Every node depends on the
single immediately preceding node. There is no branching, no parallelism, no
shared subgraph reuse.

Cross-asset work no longer lives entirely outside the DAG: live and research
pipelines now model peer loading, market-context construction, partner builds,
and attach/merge steps as graph nodes. The remaining gap is safe cache scope,
bounded scheduler rollout, and broader validator/runtime adoption.

**Impact.**
- No caching, fingerprinting, or invalidation for cross-asset work.
- No parallelism for independent partner pipeline runs (DXY, USD_JPY can run
  concurrently but currently run sequentially).
- The profiler only sees the indicator chain; the 60-80% of wall time spent on
  cross-asset is unprofiled.

**Remediation.**
- Phase 1: Add a `market_context` source node that injects a pre-built or
  cached market context frame into the graph. This alone gives fingerprint-based
  cache invalidation for the context.
- Phase 2: Add partner pipeline sub-graphs as independent branches that converge
  at an `smt_divergence` node. This enables parallel execution of partner builds
  and proper fingerprint propagation.
- Phase 3: Move `attach_cross_asset_context` into the graph as a merge node so
  the full pipeline is one graph with cache/profile coverage.

**Priority: P2 — design-heavy, schedule after node cache is proven.**

---

## 3. Partner Pipelines Run the Full Stack Unnecessarily

**Diagnosis.** When building SMT partners (e.g., DXY for XAU_USD), the code
calls `build_research_indicators` on each partner with the full stage chain
(25 stages). But partner frames are only consumed by `add_smt_divergence`, which
needs: `timestamp`, `close`, `swing_*` columns. Stages like `volume_features`,
`fvg_stack`, `displacement`, `order_blocks`, `ob_mitigation`,
`equal_hl`, `amd_engine`, `regime`, etc. are computed and
immediately discarded.

**Impact.** Each unnecessary partner stage adds ~0.5-2s. With 2 partners
(DXY, USD_JPY) and ~15 unnecessary stages each, this wastes 15-60s per pipeline
run.

**Remediation.**
- Create a `build_smt_partner_indicators` function that runs only through
  `swings` (the 9th stage) and stops. It needs: normalize → atr → ema → adx →
  rsi → macd → bb_width → body_ratio → swings.
- Wire this as the `partner_builder` callback instead of the full
  `build_research_indicators`.
- Validate: SMT output must be identical with the shorter partner stack (the
  extra columns are never read by `add_smt_divergence`).

**Priority: P1 — straightforward, large time savings.**

---

## 4. Raw Parquet Files Are Loaded Multiple Times Per Run

**Diagnosis.** `resolve_cross_asset_inputs` is called in two places during a
single `run_research_pipeline` call:

1. At the top of `run_research_pipeline` (line 518) to resolve market context
   and partner frames.
2. Inside `build_research_indicators` → `resolve_cross_asset_inputs` (line 444)
   when cross-asset is enabled.

Both calls hit `load_raw_context_frames`, which reads parquet files for all 11
context symbols from disk. The frames loaded in call (1) are passed into
`build_research_indicators` via the `market_context` and
`processed_cross_asset_frames` parameters, but then `build_research_indicators`
also defines its own `_build_partner_frame` and calls `resolve_cross_asset_inputs`
again internally for any missing partners.

Additionally, `load_raw_context_frames` is called *without* the `instruments`
filter in the context-frame loop (line 546), loading all 11 symbols even when
only 2-3 are needed for partners.

**Impact.** 11 parquet reads × 2 calls = 22 reads per run. Each file is 1-5 MB.
On spinning disk or cold NFS this adds several seconds. On SSD the overhead is
smaller but still wasteful.

**Remediation.**
- Add an LRU cache (or simple dict cache scoped to the run) for
  `load_raw_context_frames`. Key by `(symbol, timeframe)`.
- Pass `instruments=tuple(missing_partners)` to the second
  `load_raw_context_frames` call to avoid loading unnecessary symbols.
- Longer term: the raw data loading should be a source node in the DAG so
  fingerprinting handles invalidation.

**Priority: P2.**

---

## 5. Market Context Outer Merge Creates Sparse Megaframes

**Diagnosis.** `build_global_market_context` merges 11 symbol close-price series
via `how="outer"`. Commodity symbols (XAU_USD, USOIL) trade ~22 hours/day; FX
trades ~24. DXY has different session hours again. The outer merge creates one
row for every unique timestamp across all symbols, producing ~76,000 rows for a
1,200-bar H1 primary input.

While the trimming fix (`_trim_frame_to_range`) reduces this from 76k to ~1.5k,
the outer merge still produces rows where some symbols have data and others have
NaN due to session gaps. Every rolling correlation, z-score, significance, and
stability computation runs over this sparse frame.

**Impact.** Memory bloat (76k rows × hundreds of columns before trimming).
Wasted FLOPs on NaN-dominated windows. This is the single biggest CPU consumer
in the pipeline.

**Remediation (already partially done).**
- Frame trimming to primary time range: **done** (`_trim_frame_to_range`).
- `min_periods` on all rolling operations: **done** (`_corr_min_periods`).
- Remaining: consider an inner-join merge restricted to the primary symbol's
  timestamps (forward-fill peer prices to the nearest primary bar). This
  eliminates sparsity entirely and produces exactly `len(primary)` rows. The
  tradeoff is a design choice on whether you want to preserve inter-symbol
  timestamp resolution.

**Priority: P3 — the high-impact parts are done. Inner-join is a design call.**

---

## 6. No Profiling for Cross-Asset Wall Time

**Diagnosis.** `PipelineRunProfiler` and `GraphRunProfiler` track per-node
timing inside the DAG. But the cross-asset work happens outside the graph:
market context build, partner pipeline runs, SMT divergence, correlation
attachment. None of this is profiled.

**Impact.** You cannot identify or prove bottlenecks in the most expensive part
of the pipeline. If you optimize the DAG stages but cross-asset takes 80% of
wall time, you won't know.

**Remediation.**
- Wrap `build_global_market_context`, each partner build, and
  `attach_cross_asset_context` with `profiler.record_stage(name, elapsed)`.
- Or: move these into the DAG (item 2 above) so the graph profiler covers them
  automatically.

**Priority: P2.**

---

## 7. Pipeline Noop Detection Doesn't Cover Config Changes Inside Cross-Asset

**Diagnosis.** `resolve_incremental_plan` checks `config_fingerprint` to decide
if the pipeline needs to rerun. The config payload includes
`cross_asset_market_context_fp` and `cross_asset_partner_fp`. But when the
market context is loaded from cache (our Fix3), its fingerprint is computed
*after* resolution. If a new symbol is added to `CONTEXT_SYMBOLS` or a
correlation horizon changes, the cached market context is stale but its
fingerprint matches the *old* config.

**Impact.** Subtle staleness bug: changing cross-asset config constants
(e.g., adding a new LAG_SCAN_PAIR) won't invalidate cached market context
unless the user passes `--force-graph-recompute true`.

**Remediation.**
- Include a hash of the cross-asset config constants (`CONTEXT_SYMBOLS`,
  `HORIZONS_BY_TIMEFRAME`, `LAG_SCAN_PAIRS`, `SMT_PARTNERS`,
  `VOL_NORM_HALFLIFE`) in the market context fingerprint.
- When loading cached market context, verify this hash matches before accepting
  the cache.

**Priority: P2.**

---

## 8. No Incremental Market Context Computation

**Diagnosis.** `build_global_market_context` recomputes all correlations from
scratch every time. If 100 new bars arrive on an existing 5,000-bar dataset,
all 5,000 bars' worth of rolling correlations, z-scores, significance, and
stability classifications are recomputed.

**Impact.** On large datasets the correlation computation scales as
O(n_symbols² × n_horizons × n_bars). For 11 symbols, 3 horizons, and 5,000
bars this is ~1.8M rolling window evaluations. With incremental, only the new
frontier needs computation.

**Remediation.**
- Store the market context frame as a persisted dataset (already done via
  `persist_market_context`).
- On incremental runs, load the prior market context, append new rows to the
  raw context frames, and recompute only the frontier (last `max_horizon` bars).
- Merge the recomputed frontier with the historical market context using
  `merge_recomputed_frontier`.

**Priority: P2 — medium effort, high payoff on large datasets.**

---

## 9. Validation Script Rebuilds SMT Research Table Every Run

**Diagnosis.** `validate_smt.py` line 154 calls
`build_smt_research_table(full_df)` on every run. This table is a pure function
of the feature frame and is not persisted.

**Impact.** Minor (research table build is fast), but violates the caching
principle. Repeated validation runs during development waste a few seconds each.

**Remediation.**
- Persist the research table alongside the feature frame.
- Load from cache when the feature frame fingerprint matches.

**Priority: P3.**

---

## 10. No Connection Pooling for Parquet Reads

**Diagnosis.** Every `pd.read_parquet` call opens, reads, and closes the file
independently. `load_raw_context_frames` loops over 11 symbols doing individual
reads. `load_partitioned_dataset` does glob + concat of monthly partitions.

**Impact.** On NFS or cloud-mounted storage, file open latency dominates. Even
on local SSD, pyarrow's parquet reader has per-file overhead (~10-50ms) that
adds up across 20+ reads per pipeline run.

**Remediation.**
- Add a session-scoped frame cache (dict keyed by `(path, mtime)`) that deduplicates
  reads within a single pipeline run.
- For partitioned datasets, consider pre-concatenating partitions into a single
  consolidated file when the partition set is stable.

**Priority: P3.**

---

## 11. `_iter_correlation_pairs` Computes All N² Pairs

**Diagnosis.** Cross-asset correlation is computed for every unordered pair of
available symbols: C(11,2) = 55 pairs × 3 horizons = 165 rolling correlations.
But the downstream consumer (`_attach_named_columns`) only maps a subset of
these to the output frame. For XAU_USD, only pairs involving XAU_USD, DXY,
USD_JPY, USOIL are relevant. The remaining ~120 correlations are computed and
stored in the market context but never read by the primary instrument's feature
frame.

**Impact.** ~70% of correlation computation is wasted for any single-instrument
run. This is pure CPU waste.

**Remediation.**
- Keep the global context as-is for multi-instrument research (it's correct
  to have the full matrix available).
- For single-instrument pipeline runs, add a `relevant_pairs` filter that
  limits correlation computation to pairs involving the primary instrument and
  its SMT partners.
- The global context can remain a superset; the filter only applies when
  building context for a specific instrument run.

**Priority: P2 — moderate effort, significant CPU reduction.**

---

## 12. Fingerprint Uses Full DataFrame Content Hash

**Diagnosis.** `dataframe_fingerprint` (in `pipeline_runtime`) computes a
content hash of the entire DataFrame. For a 1,200-row, 150-column frame, this
involves serializing ~180K cells to produce a hash.

**Impact.** Fingerprint computation adds 50-200ms per frame. It's called
multiple times per pipeline run: on the raw input, on the market context, on
each partner frame.

**Remediation.**
- For raw input frames, use a fast fingerprint: `(row_count, first_ts, last_ts,
  sha256(close[-10:]))`. This catches all practical data changes.
- For market context, use `(row_count, column_count, first_ts, last_ts)`.
- Keep full content hash as a validation-mode option, not the default hot path.

**Priority: P3 — micro-optimization, do after the bigger items.**

---

## Summary Table

| # | Issue | Priority | Status |
|---|-------|----------|--------|
| 1 | DAG node cache only partially active (carried-state Class B rolled back) | P1 | Partial |
| 2 | Cross-asset now DAG-owned, but runtime adoption/correctness hardening remains | P2 | Partial |
| 3 | Partner pipelines run full 25-stage stack | P1 | Open |
| 4 | Raw parquet files loaded multiple times | P2 | Open |
| 5 | Outer merge sparse megaframes | P3 | Partially fixed |
| 6 | No profiling for cross-asset wall time | P2 | Open |
| 7 | Config changes don't invalidate cached context | P2 | Open |
| 8 | No incremental market context computation | P2 | Open |
| 9 | Research table rebuilt every validation run | P3 | Open |
| 10 | No parquet read deduplication | P3 | Open |
| 11 | All N² correlation pairs computed | P2 | Open |
| 12 | Fingerprint uses full content hash | P3 | Open |

### Recommended execution order

1. **P1-3**: Trim partner pipelines to stop at `swings` — biggest bang for the
   least code.
2. **P1-1**: Enable `materialize=True` on Class A nodes + `swings` — unlocks
   sub-second DAG re-runs.
3. **P2-6**: Add profiler coverage for cross-asset — needed to measure remaining
   items.
4. **P2-7**: Fix config fingerprinting for market context cache.
5. **P2-11**: Filter correlation pairs by relevance.
6. **P2-4**: Deduplicate raw parquet reads.
7. **P2-8**: Incremental market context computation.
8. **P2-2**: Refactor DAG to include cross-asset as branches (design-heavy).
9. **P3-5/9/10/12**: Remaining micro-optimizations.
