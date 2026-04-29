

Short answer to your product question: yes, this work (the docs/DATA_ENGINEERING.md plan that has been done ) is directly relevant to the scanner and live system, not just to testing.

Why it matters:

it reduces scan-cycle latency by making live feature generation narrower and more reusable
it improves worst-case latency, not just average speed
it reduces CPU/disk contention so the scanner is less likely to get starved by pipeline work
it makes live outputs safer through replay correctness, source-hash invalidation, and explicit dependency tracking
it gives you a cleaner platform for later strategy work, instead of scaling a wasteful runtime into production
The important nuance is:

data engineering improves latency, stability, correctness, and operating cost
it does not by itself improve detector quality or trading edge
For the scheduler specifically, the doc now recommends:

make it a global DAG-runtime capability
keep it serial by default
enable it first on:
research cross-asset partner branches
live cross-asset partner branches
later, stateless foundation sibling stages
keep it serial for now on:
carried-state structural/SMC chains
range_boundaries main heavy compute path