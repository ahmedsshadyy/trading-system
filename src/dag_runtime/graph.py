from __future__ import annotations

from dataclasses import dataclass

from src.dag_runtime.node import NodeManifest


@dataclass(frozen=True, slots=True)
class GraphManifest:
    graph_name: str
    nodes: tuple[NodeManifest, ...]
    default_target: str

    def node_map(self) -> dict[str, NodeManifest]:
        return {node.node_name: node for node in self.nodes}


def dependency_closure(graph: GraphManifest, target: str) -> list[str]:
    nodes = graph.node_map()
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name not in nodes:
            raise KeyError(f"Unknown node {name!r} for graph {graph.graph_name!r}")
        visited.add(name)
        for dep in nodes[name].upstream_nodes:
            visit(dep)
        ordered.append(name)

    visit(target)
    return ordered


def topo_nodes(graph: GraphManifest, target: str) -> list[NodeManifest]:
    node_map = graph.node_map()
    return [node_map[name] for name in dependency_closure(graph, target)]
