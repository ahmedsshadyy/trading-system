from __future__ import annotations

import time
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
    node_results: dict[str, NodeExecutionResult] = {}
    invalidated = invalidate_nodes or set()

    for manifest in topo_nodes(graph, actual_target):
        dep_results = {name: node_results[name] for name in manifest.upstream_nodes}
        fingerprint = node_fingerprint(manifest, context, dep_results)
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
        if cache_candidate is not None:
            node_results[manifest.node_name] = cache_candidate
            runtime_profiler.record_node(
                node_name=manifest.node_name,
                node_kind=manifest.node_kind,
                started_at=time.perf_counter(),
                cache_hit=True,
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
                details=cache_candidate.output.profile_details,
            )
            continue

        if context.explain_only:
            explain_result = NodeExecutionResult(
                manifest=manifest,
                output=NodeOutput(payload={"explain": "would_execute"}),
                fingerprint=fingerprint,
                cache_hit=False,
            )
            node_results[manifest.node_name] = explain_result
            runtime_profiler.record_node(
                node_name=manifest.node_name,
                node_kind=manifest.node_kind,
                started_at=time.perf_counter(),
                cache_hit=False,
                fingerprint=fingerprint,
                details={"explain_only": True, **explain_result.output.profile_details},
            )
            continue

        started_at = time.perf_counter()
        output = manifest.compute_fn(context, dep_results)
        profile_details = dict(output.profile_details)
        if manifest.cache_policy.materialize:
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
            profile_details["cache_write_seconds"] = (
                time.perf_counter() - cache_write_started_at
            )
            if manifest.node_name == "range_selected_debug":
                profile_details["selected_debug_cache_write"] = profile_details[
                    "cache_write_seconds"
                ]
            output.profile_details = profile_details
            for name, frame in output.frames.items():
                frame_path = result.cache_path.parent / f"{fingerprint}.{name}.parquet"
                runtime_profiler.record_artifact(
                    node_name=manifest.node_name,
                    path=frame_path,
                    bytes_written=(
                        frame_path.stat().st_size if frame_path.exists() else None
                    ),
                    rows=len(frame),
                    kind=f"{manifest.node_kind}:frame",
                )
            if result.cache_path is not None and result.cache_path.exists():
                runtime_profiler.record_artifact(
                    node_name=manifest.node_name,
                    path=result.cache_path,
                    bytes_written=result.cache_path.stat().st_size,
                    kind=f"{manifest.node_kind}:payload",
                )
        else:
            result = NodeExecutionResult(
                manifest=manifest,
                output=output,
                fingerprint=fingerprint,
                cache_hit=False,
            )
        node_results[manifest.node_name] = result
        primary_frame = result.primary_frame()
        runtime_profiler.record_node(
            node_name=manifest.node_name,
            node_kind=manifest.node_kind,
            started_at=started_at,
            cache_hit=False,
            rows_out=len(primary_frame) if primary_frame is not None else None,
            estimated_memory_bytes=(
                int(primary_frame.memory_usage(deep=True).sum())
                if primary_frame is not None
                else None
            ),
            fingerprint=fingerprint,
            details=profile_details,
        )

    return GraphRunResult(
        graph=graph,
        target=actual_target,
        node_results=node_results,
        profiler=runtime_profiler,
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
