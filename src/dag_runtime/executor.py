from __future__ import annotations

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from src.dag_runtime.cache_store import (
    invalidate_node_cache,
    load_cached_node,
    save_cached_node,
)
from src.dag_runtime.fingerprints import node_fingerprint
from src.dag_runtime.graph import GraphManifest, topo_nodes
from src.dag_runtime.node import GraphRunContext, NodeExecutionResult, NodeOutput
from src.dag_runtime.profiling import GraphRunProfiler


@dataclass(slots=True)
class GraphRunResult:
    graph: GraphManifest
    target: str
    node_results: dict[str, NodeExecutionResult]
    profiler: GraphRunProfiler

    def output(self) -> NodeOutput:
        return self.node_results[self.target].output

    def primary_frame(self):
        return self.node_results[self.target].primary_frame()

    @property
    def executed_nodes(self) -> list[str]:
        return [
            name for name, result in self.node_results.items() if not result.cache_hit
        ]

    @property
    def closure_nodes(self) -> list[str]:
        return list(self.node_results)


@dataclass(slots=True)
class _NodeArtifacts:
    path: str
    bytes_written: int | None = None
    rows: int | None = None
    kind: str | None = None


@dataclass(slots=True)
class _ComputedNode:
    node_name: str
    result: NodeExecutionResult
    seconds: float
    queue_wait_seconds: float
    rows_out: int | None
    estimated_memory_bytes: int | None
    details: dict[str, Any]
    artifacts: list[_NodeArtifacts]


def _ordered_node_results(
    manifests: list, node_results: dict[str, NodeExecutionResult]
) -> dict[str, NodeExecutionResult]:
    return {
        manifest.node_name: node_results[manifest.node_name]
        for manifest in manifests
        if manifest.node_name in node_results
    }


def _record_materialized_artifacts(
    profiler: GraphRunProfiler, node_name: str, artifacts: list[_NodeArtifacts]
) -> None:
    for artifact in artifacts:
        profiler.record_artifact(
            node_name=node_name,
            path=artifact.path,
            bytes_written=artifact.bytes_written,
            rows=artifact.rows,
            kind=artifact.kind,
        )


def _compute_node(
    manifest,
    *,
    context: GraphRunContext,
    dep_results: dict[str, NodeExecutionResult],
    fingerprint: str,
    submit_time: float,
    cache_write_semaphore: threading.Semaphore,
) -> _ComputedNode:
    started_at = time.perf_counter()
    output = manifest.compute_fn(context, dep_results)
    profile_details = dict(output.profile_details)
    artifacts: list[_NodeArtifacts] = []
    if manifest.cache_policy.materialize:
        wait_started_at = time.perf_counter()
        with cache_write_semaphore:
            cache_write_wait_seconds = time.perf_counter() - wait_started_at
            cache_write_started_at = time.perf_counter()
            result = save_cached_node(
                context.cache_root,
                manifest=manifest,
                symbol=context.symbol,
                timeframe=context.timeframe,
                fingerprint=fingerprint,
                output=output,
                metadata_extra={"node_kind": manifest.node_kind},
            )
            cache_write_seconds = time.perf_counter() - cache_write_started_at
        profile_details["cache_write_seconds"] = cache_write_seconds
        profile_details["cache_write_wait_seconds"] = cache_write_wait_seconds
        if manifest.node_name == "range_selected_debug":
            profile_details["selected_debug_cache_write"] = cache_write_seconds
        output.profile_details = profile_details
        for name, frame in output.frames.items():
            frame_path = result.cache_path.parent / f"{fingerprint}.{name}.parquet"
            artifacts.append(
                _NodeArtifacts(
                    path=str(frame_path),
                    bytes_written=(
                        frame_path.stat().st_size if frame_path.exists() else None
                    ),
                    rows=len(frame),
                    kind=f"{manifest.node_kind}:frame",
                )
            )
        if result.cache_path is not None and result.cache_path.exists():
            artifacts.append(
                _NodeArtifacts(
                    path=str(result.cache_path),
                    bytes_written=result.cache_path.stat().st_size,
                    kind=f"{manifest.node_kind}:payload",
                )
            )
    else:
        result = NodeExecutionResult(
            manifest=manifest,
            output=output,
            fingerprint=fingerprint,
            cache_hit=False,
        )
    primary_frame = result.primary_frame()
    return _ComputedNode(
        node_name=manifest.node_name,
        result=result,
        seconds=time.perf_counter() - started_at,
        queue_wait_seconds=started_at - submit_time,
        rows_out=len(primary_frame) if primary_frame is not None else None,
        estimated_memory_bytes=(
            int(primary_frame.memory_usage(deep=True).sum())
            if primary_frame is not None
            else None
        ),
        details=profile_details,
        artifacts=artifacts,
    )


def _integrate_computed_node(
    *,
    profiler: GraphRunProfiler,
    node_results: dict[str, NodeExecutionResult],
    computed: _ComputedNode,
) -> None:
    details = dict(computed.details)
    details["runnable_queue_wait_seconds"] = computed.queue_wait_seconds
    node_results[computed.node_name] = computed.result
    _record_materialized_artifacts(profiler, computed.node_name, computed.artifacts)
    profiler.record_node(
        node_name=computed.node_name,
        node_kind=computed.result.manifest.node_kind,
        cache_hit=False,
        seconds=computed.seconds,
        rows_out=computed.rows_out,
        estimated_memory_bytes=computed.estimated_memory_bytes,
        fingerprint=computed.result.fingerprint,
        details=details,
    )


def _resolve_cached_or_explain(
    manifest,
    *,
    graph: GraphManifest,
    context: GraphRunContext,
    dep_results: dict[str, NodeExecutionResult],
    fingerprint: str,
    invalidated: set[str],
    profiler: GraphRunProfiler,
) -> NodeExecutionResult | None:
    cache_lookup_started_at = time.perf_counter()
    if manifest.node_name in invalidated:
        invalidate_node_cache(
            context.cache_root,
            graph_name=graph.graph_name,
            symbol=context.symbol,
            timeframe=context.timeframe,
            node_name=manifest.node_name,
        )
    cache_candidate = None
    if (
        manifest.cache_policy.materialize
        and not context.force
        and not context.invalidate_cache
        and manifest.node_name not in invalidated
    ):
        cache_candidate = load_cached_node(
            context.cache_root,
            manifest=manifest,
            symbol=context.symbol,
            timeframe=context.timeframe,
            fingerprint=fingerprint,
        )
    cache_lookup_seconds = time.perf_counter() - cache_lookup_started_at
    if cache_candidate is not None:
        profiler.record_node(
            node_name=manifest.node_name,
            node_kind=manifest.node_kind,
            cache_hit=True,
            seconds=cache_lookup_seconds,
            rows_out=(
                len(cache_candidate.primary_frame())
                if cache_candidate.primary_frame() is not None
                else None
            ),
            estimated_memory_bytes=(
                int(cache_candidate.primary_frame().memory_usage(deep=True).sum())
                if cache_candidate.primary_frame() is not None
                else None
            ),
            fingerprint=fingerprint,
            details={
                **cache_candidate.output.profile_details,
                "runnable_queue_wait_seconds": 0.0,
            },
        )
        return cache_candidate

    if context.explain_only:
        explain_result = NodeExecutionResult(
            manifest=manifest,
            output=NodeOutput(payload={"explain": "would_execute"}),
            fingerprint=fingerprint,
            cache_hit=False,
        )
        profiler.record_node(
            node_name=manifest.node_name,
            node_kind=manifest.node_kind,
            cache_hit=False,
            seconds=cache_lookup_seconds,
            fingerprint=fingerprint,
            details={"explain_only": True, "runnable_queue_wait_seconds": 0.0},
        )
        return explain_result
    return None


def _execute_serial(
    *,
    graph: GraphManifest,
    target: str,
    context: GraphRunContext,
    profiler: GraphRunProfiler,
    invalidated: set[str],
) -> GraphRunResult:
    node_results: dict[str, NodeExecutionResult] = {}
    manifests = topo_nodes(graph, target)
    cache_write_semaphore = threading.Semaphore(1)
    for manifest in manifests:
        dep_results = {name: node_results[name] for name in manifest.upstream_nodes}
        fingerprint = node_fingerprint(manifest, context, dep_results)
        resolved = _resolve_cached_or_explain(
            manifest,
            graph=graph,
            context=context,
            dep_results=dep_results,
            fingerprint=fingerprint,
            invalidated=invalidated,
            profiler=profiler,
        )
        if resolved is not None:
            node_results[manifest.node_name] = resolved
            continue
        computed = _compute_node(
            manifest,
            context=context,
            dep_results=dep_results,
            fingerprint=fingerprint,
            submit_time=time.perf_counter(),
            cache_write_semaphore=cache_write_semaphore,
        )
        _integrate_computed_node(
            profiler=profiler, node_results=node_results, computed=computed
        )
    return GraphRunResult(
        graph=graph,
        target=target,
        node_results=_ordered_node_results(manifests, node_results),
        profiler=profiler,
    )


def _execute_bounded_parallel(
    *,
    graph: GraphManifest,
    target: str,
    context: GraphRunContext,
    profiler: GraphRunProfiler,
    invalidated: set[str],
) -> GraphRunResult:
    manifests = topo_nodes(graph, target)
    order_index = {
        manifest.node_name: index for index, manifest in enumerate(manifests)
    }
    completed: dict[str, NodeExecutionResult] = {}
    in_flight: dict[Future[_ComputedNode], str] = {}
    pending = {manifest.node_name for manifest in manifests}
    cache_write_semaphore = threading.Semaphore(
        max(1, context.execution_policy.max_concurrent_cache_writes)
    )
    max_workers = max(1, context.execution_policy.max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or in_flight:
            progressed = False
            for manifest in manifests:
                if manifest.node_name not in pending:
                    continue
                if any(dep not in completed for dep in manifest.upstream_nodes):
                    continue
                dep_results = {
                    name: completed[name] for name in manifest.upstream_nodes
                }
                fingerprint = node_fingerprint(manifest, context, dep_results)
                resolved = _resolve_cached_or_explain(
                    manifest,
                    graph=graph,
                    context=context,
                    dep_results=dep_results,
                    fingerprint=fingerprint,
                    invalidated=invalidated,
                    profiler=profiler,
                )
                if resolved is not None:
                    completed[manifest.node_name] = resolved
                    pending.remove(manifest.node_name)
                    progressed = True
                    continue
                if len(in_flight) >= max_workers:
                    continue
                submit_time = time.perf_counter()
                future = executor.submit(
                    _compute_node,
                    manifest,
                    context=context,
                    dep_results=dep_results,
                    fingerprint=fingerprint,
                    submit_time=submit_time,
                    cache_write_semaphore=cache_write_semaphore,
                )
                in_flight[future] = manifest.node_name
                pending.remove(manifest.node_name)
                progressed = True
            if progressed:
                continue
            if not in_flight:
                break
            done, _pending = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            completed_futures = sorted(
                ((order_index[in_flight[future]], future) for future in done),
                key=lambda item: item[0],
            )
            for _index, future in completed_futures:
                in_flight.pop(future)
                try:
                    computed = future.result()
                except Exception:
                    for outstanding in in_flight:
                        outstanding.cancel()
                    raise
                _integrate_computed_node(
                    profiler=profiler, node_results=completed, computed=computed
                )
        if pending:
            raise RuntimeError(
                f"bounded_parallel scheduler could not resolve pending nodes: {sorted(pending)}"
            )
    return GraphRunResult(
        graph=graph,
        target=target,
        node_results=_ordered_node_results(manifests, completed),
        profiler=profiler,
    )


def execute_graph(
    graph: GraphManifest,
    *,
    target: str | None = None,
    context: GraphRunContext,
    profiler: GraphRunProfiler | None = None,
    invalidate_nodes: set[str] | None = None,
) -> GraphRunResult:
    actual_target = target or graph.default_target
    runtime_profiler = profiler or GraphRunProfiler(
        graph_name=graph.graph_name,
        symbol=context.symbol,
        timeframe=context.timeframe,
    )
    invalidated = invalidate_nodes or set()
    policy = context.execution_policy
    scheduler_mode = policy.scheduler_mode
    worker_count = 1
    if (
        scheduler_mode == "bounded_parallel"
        and not context.explain_only
        and policy.max_workers > 1
    ):
        worker_count = min(max(1, policy.max_workers), max(1, os.cpu_count() or 1))
    else:
        scheduler_mode = "serial"
    runtime_profiler.set_scheduler(
        scheduler_mode=scheduler_mode, worker_count=worker_count
    )
    runtime_profiler.set_metric(
        "max_concurrent_cache_writes",
        max(1, context.execution_policy.max_concurrent_cache_writes),
    )
    if scheduler_mode == "bounded_parallel":
        return _execute_bounded_parallel(
            graph=graph,
            target=actual_target,
            context=context,
            profiler=runtime_profiler,
            invalidated=invalidated,
        )
    return _execute_serial(
        graph=graph,
        target=actual_target,
        context=context,
        profiler=runtime_profiler,
        invalidated=invalidated,
    )


def explain_graph_run(
    graph: GraphManifest,
    *,
    target: str | None = None,
    context: GraphRunContext,
    invalidate_nodes: set[str] | None = None,
) -> dict[str, Any]:
    explain_context = GraphRunContext(
        graph_name=context.graph_name,
        symbol=context.symbol,
        timeframe=context.timeframe,
        inputs=context.inputs,
        config=context.config,
        cache_root=context.cache_root,
        state_root=context.state_root,
        features_root=context.features_root,
        force=context.force,
        invalidate_cache=context.invalidate_cache,
        explain_only=True,
        execution_policy=context.execution_policy,
    )
    invalidated = invalidate_nodes or set()
    result = execute_graph(
        graph, target=target, context=explain_context, invalidate_nodes=invalidated
    )
    return {
        "graph_name": graph.graph_name,
        "target": result.target,
        "nodes": [
            {
                "node_name": name,
                "node_kind": execution.manifest.node_kind,
                "upstream_nodes": list(execution.manifest.upstream_nodes),
                "fingerprint": execution.fingerprint,
                "cache_hit": execution.cache_hit,
                "would_execute": not execution.cache_hit,
                "reason": (
                    "cache-hit"
                    if execution.cache_hit
                    else (
                        "invalidated-node"
                        if name in invalidated
                        else (
                            "force"
                            if context.force
                            else (
                                "invalidate-cache"
                                if context.invalidate_cache
                                else (
                                    "source-input"
                                    if execution.manifest.node_kind == "source"
                                    else "cache-miss"
                                )
                            )
                        )
                    )
                ),
            }
            for name, execution in result.node_results.items()
        ],
    }
