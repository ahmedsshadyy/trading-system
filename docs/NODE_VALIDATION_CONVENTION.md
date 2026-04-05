# Node Validation Convention

This document defines how DAG nodes must be validated in this codebase. The goal is to preserve exact trading semantics while allowing aggressive refactoring of orchestration, caching, and persistence.

## What Counts As A Node

A node is a single manifest-defined unit of execution with:

- explicit upstream dependencies
- an explicit fingerprint contract
- explicit output artifacts
- an explicit semantic class
- an explicit validation level

Examples:

- causal compute stage
- aggregate analytics stage
- selected-rung decision stage
- canonical materialization stage
- report/render stage

## Node Contract Requirements

Every node must declare:

- graph name
- node name
- node kind
- semantic class
- upstream nodes
- source inputs
- output artifacts
- schema version
- feature contract version
- engine version
- cache policy
- validation policy
- replay/frontier policy if applicable
- mutable scope
- failure recovery policy

No node may rely on hidden upstream state that is not either:

- a declared source input, or
- a declared upstream node artifact

## Validation Levels

### 1. Unit validation

Use when:

- the node is a small pure helper
- no graph-level dependency behavior is involved

Required checks:

- deterministic output for fixed input
- required columns/fields present
- expected empty-input behavior

### 2. Node parity validation

Use when:

- a node wraps existing logic whose semantics must not change

Required checks:

- new node output equals legacy direct-call output
- exact equality for event ids, timestamps, state labels, and discrete flags
- tolerant float equality only where the legacy validator already tolerates float variance

### 3. Graph parity validation

Use when:

- node composition or orchestration changes

Required checks:

- graph target output equals legacy pipeline/validator output
- same row count
- same timestamps/order
- same columns
- same event timing
- same selected-rung / selected-config decisions where applicable

### 4. Incremental/frontier validation

Use when:

- a node participates in live or research incremental execution

Required checks:

- full rebuild vs incremental parity
- no-op rerun behavior
- one-bar append behavior
- multi-bar append behavior
- changed frontier input behavior
- interrupted-write recovery behavior for materialization nodes

### 5. Report/render validation

Use when:

- the node writes HTML/CSV/JSON/MD or similar presentation artifacts

Required checks:

- unchanged inputs skip rewrite
- report invalidation does not trigger upstream compute invalidation
- output path and metadata fingerprint update correctly

## Required Checks On Every Iteration

When any node changes, validate in this order:

1. Node contract still matches documented inputs/outputs
2. Fingerprint inputs are still complete and content-aware
3. Cache invalidation remains dependency-driven
4. Legacy-vs-node parity still passes
5. Graph target parity still passes if the node is on a canonical path
6. Profiling evidence still makes sense

If the node is a live or research materialization node, also validate:

7. immutable historical partitions remain untouched on ordinary runs
8. metadata advances only after successful persistence

## When Full Rebuild Comparison Is Mandatory

Full rebuild comparison is mandatory when:

- replay policy changes
- stage ordering changes
- a node moves between semantic classes
- a node becomes frontier-safe for the first time
- research-only behavior is split from shared live-backbone behavior
- selected-rung / selected-config logic changes

## Approving A Node For Frontier-Safe Incremental Execution

Do not mark a node as frontier-safe until all of the following are true:

- replay policy is explicit
- upstream dependencies are explicit
- incremental vs full rebuild parity passes on representative windows
- changed-frontier-input invalidation behaves correctly
- interrupted-write recovery is tested if the node writes canonical artifacts

If any of those are missing, the node stays:

- `full_history_only`, or
- `explicit_rebuild_only`

## No-Semantic-Drift Merge Gate

Before merge, the implementer must be able to state:

- which node changed
- whether outputs changed intentionally or not
- what parity comparison was run
- whether event timing changed
- whether schema changed
- whether replay/frontier behavior changed

If schema or semantics changed intentionally, the contract doc and node manifest versions must be bumped explicitly.

## Hard Validator Merge Gate

Validator-family work is blocked from acceptance unless all of the following pass:

1. node parity vs legacy helper path
2. graph parity vs legacy validator path
3. command-contract verification
4. profiler sanity check

### Command-contract verification requires

- wrapper command hits the intended DAG target or target bundle
- unchanged rerun reuses cache as expected
- report-only command does not trigger upstream recompute
- printed profiler path and artifact paths match the executed target class

### Profiler sanity check requires

- cache hit/miss behavior matches the intended change
- targeted speedup can be tied to a named node or bundle
- no unexpected report or artifact writes appear in diagnostic-only commands

### Additional hard gate for range boundaries

`range_boundaries` changes are blocked unless all of these remain unchanged relative to the legacy path:

- selected rung
- retune usage
- selected summary
- downstream summary
- diagnostics bundle
- report targets remain terminal only

### Acceptance rule

- incomplete parity means the change is not accepted
- there is no “fast now, verify later” path for validator orchestration work

## Performance Freeze Gate

Declaring a validator family performance-frozen requires all of the following:

1. profiling proof
2. target-closure proof
3. cache/write-discipline proof
4. no-drift proof

### Profiling proof requires

- named sub-stage timings for the known heavy path
- stable machine-readable profiler output
- bottleneck attribution by named node and named sub-stage

### Target-closure proof requires

- each supported target runs only its intended closure plus true upstream dependencies
- report targets do not execute unrelated downstream bundles
- report flags do not invalidate upstream compute nodes

### Cache/write-discipline proof requires

- unchanged rerun is mostly cache hits
- diagnostic-only targets do not write unrelated artifacts
- report-only targets write only report artifacts for the chosen target

### No-drift proof requires

- the existing hard validator merge gate still passes
- no event timing, selected-config, schema, or report-semantic change is accepted

## Profiling Evidence Expected

Every nontrivial DAG refactor should produce:

- node-level timings
- cache hit/miss counts
- bytes written per node
- artifacts written per node
- before/after evidence for the targeted bottleneck

Optimization work is incomplete if it reduces runtime but cannot show which node family got faster or which writes were eliminated.

## Acceptance Template For New Nodes

Every new node should be reviewed against this checklist:

- name and purpose are specific
- upstream nodes are explicit
- source inputs are explicit
- fingerprint inputs are complete
- output artifacts are minimal and reusable
- validation level is correct
- replay/frontier policy is explicit if applicable
- parity test exists
- profiler output includes this node
- report nodes are terminal only

## Practical Iteration Rule

For this codebase, iteration should be:

1. add or refactor one node family
2. run node parity
3. run graph parity
4. run incremental/frontier checks if relevant
5. inspect profiler output
6. only then proceed to the next node family

Do not refactor multiple heavy node families at once without validating each layer separately.
