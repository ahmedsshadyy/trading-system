from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pandas as pd

from src.dag_runtime.node import NodeOutput
from scripts import validate_range_boundaries as vrb


def _sample_raw() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=20, freq="4h", tz="UTC")
    close = pd.Series(range(20), dtype=float).mul(0.5).add(2000.0)
    open_ = close.shift(1).fillna(close.iloc[0] - 0.2)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.3
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.3
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000 + idx for idx in range(len(ts))],
        }
    )


def _args(**overrides) -> argparse.Namespace:
    base = {
        "instrument": "UNIT",
        "timeframe": "H4",
        "target": "selection",
        "date_from": "2026-01-01",
        "plot_rows": 300,
        "tail_rows": None,
        "full": False,
        "html": False,
        "write_csv": False,
        "explain": False,
        "force": False,
        "invalidate_cache": False,
        "cleanup_stale": False,
        "max_artifact_age_days": 30,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class _FakeGraphResult:
    def __init__(self, payload: dict, *, executed_nodes: list[str]) -> None:
        self._payload = payload
        self.node_results = {
            name: SimpleNamespace(cache_hit=False) for name in executed_nodes
        }
        self.profiler = SimpleNamespace(write_json=lambda _path: None)

    def output(self) -> NodeOutput:
        return NodeOutput(payload=self._payload)


def test_selection_target_resolves_to_selection_bundle(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        argparse.ArgumentParser, "parse_args", lambda self: _args(target="selection")
    )
    monkeypatch.setattr(vrb.Path, "exists", lambda self: True)
    monkeypatch.setattr(vrb.pd, "read_parquet", lambda _path: _sample_raw())
    monkeypatch.setattr(vrb, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(vrb, "CACHE_ROOT", tmp_path / "cache")

    captured: dict[str, object] = {}

    def fake_execute_graph(graph, *, context, target):
        captured["target"] = target
        return _FakeGraphResult(
            {
                "selected_rung": {
                    "selected_label": "step8e_a/mid_a",
                    "reporting_label": "step8e_a/mid_a",
                    "used_retune": False,
                    "selected_params": {"min_confirm_bars": 3},
                    "has_valid_rung": True,
                }
            },
            executed_nodes=[
                "range_context",
                "range_selected_rung",
                "range_selection_bundle",
            ],
        )

    monkeypatch.setattr(
        vrb,
        "get_builtin_graph",
        lambda *args, **kwargs: SimpleNamespace(graph_name="validate_range_boundaries"),
    )
    monkeypatch.setattr(vrb, "execute_graph", fake_execute_graph)

    vrb.main()

    out = capsys.readouterr().out
    assert captured["target"] == "range_selection_bundle"
    assert "wrapper_target: selection" in out
    assert "resolved_target: range_selection_bundle" in out
    assert "selected_rung: step8e_a/mid_a" in out


def test_geometry_target_resolves_to_geometry_node(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        argparse.ArgumentParser, "parse_args", lambda self: _args(target="geometry")
    )
    monkeypatch.setattr(vrb.Path, "exists", lambda self: True)
    monkeypatch.setattr(vrb.pd, "read_parquet", lambda _path: _sample_raw())
    monkeypatch.setattr(vrb, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(vrb, "CACHE_ROOT", tmp_path / "cache")

    captured: dict[str, object] = {}

    def fake_execute_graph(graph, *, context, target):
        captured["target"] = target
        return _FakeGraphResult(
            {"summary": {}},
            executed_nodes=[
                "range_context",
                "range_selected_debug",
                "range_geometry_audit",
            ],
        )

    result = _FakeGraphResult(
        {"summary": {}},
        executed_nodes=[
            "range_context",
            "range_selected_debug",
            "range_geometry_audit",
        ],
    )
    result.output = lambda: NodeOutput(
        frames={
            "geometry_audit": pd.DataFrame(
                {"geometry_review_bucket_suggested": ["ok", "ok"]}
            )
        }
    )

    monkeypatch.setattr(
        vrb,
        "get_builtin_graph",
        lambda *args, **kwargs: SimpleNamespace(graph_name="validate_range_boundaries"),
    )
    monkeypatch.setattr(
        vrb,
        "execute_graph",
        lambda graph, *, context, target: (
            captured.update({"target": target}) or result
        ),
    )

    vrb.main()

    out = capsys.readouterr().out
    assert captured["target"] == "range_geometry_audit"
    assert "resolved_target: range_geometry_audit" in out
    assert "geometry_rows: 2" in out


def test_charts_target_resolves_to_chart_bundle(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: _args(target="charts", html=True),
    )
    monkeypatch.setattr(vrb.Path, "exists", lambda self: True)
    monkeypatch.setattr(vrb.pd, "read_parquet", lambda _path: _sample_raw())
    monkeypatch.setattr(vrb, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(vrb, "CACHE_ROOT", tmp_path / "cache")

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        vrb,
        "get_builtin_graph",
        lambda *args, **kwargs: SimpleNamespace(graph_name="validate_range_boundaries"),
    )
    monkeypatch.setattr(
        vrb,
        "execute_graph",
        lambda graph, *, context, target: (
            captured.update({"target": target})
            or _FakeGraphResult(
                {
                    "artifacts": {
                        "range_main_chart": {
                            "html_path": "main.html",
                            "cache_hit": False,
                        },
                        "range_geometry_chart_pack": {
                            "html_path": "geometry.html",
                            "cache_hit": True,
                        },
                    }
                },
                executed_nodes=["range_selected_debug", "range_chart_bundle"],
            )
        ),
    )

    vrb.main()

    out = capsys.readouterr().out
    assert captured["target"] == "range_chart_bundle"
    assert "resolved_target: range_chart_bundle" in out
    assert "range_main_chart: main.html" in out


def test_csv_target_resolves_to_csv_bundle(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: _args(target="csv", write_csv=True),
    )
    monkeypatch.setattr(vrb.Path, "exists", lambda self: True)
    monkeypatch.setattr(vrb.pd, "read_parquet", lambda _path: _sample_raw())
    monkeypatch.setattr(vrb, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(vrb, "CACHE_ROOT", tmp_path / "cache")

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        vrb,
        "get_builtin_graph",
        lambda *args, **kwargs: SimpleNamespace(graph_name="validate_range_boundaries"),
    )
    monkeypatch.setattr(
        vrb,
        "execute_graph",
        lambda graph, *, context, target: (
            captured.update({"target": target})
            or _FakeGraphResult(
                {
                    "artifact_paths": {
                        "events": "events.csv",
                        "ranking_memo": "ranking.md",
                    }
                },
                executed_nodes=["range_selected_debug", "range_csv_bundle"],
            )
        ),
    )

    vrb.main()

    out = capsys.readouterr().out
    assert captured["target"] == "range_csv_bundle"
    assert "resolved_target: range_csv_bundle" in out
    assert "events: events.csv" in out


def test_explain_prints_resolved_target_and_reason(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: _args(target="selection", explain=True),
    )
    monkeypatch.setattr(vrb.Path, "exists", lambda self: True)
    monkeypatch.setattr(vrb.pd, "read_parquet", lambda _path: _sample_raw())
    monkeypatch.setattr(vrb, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(vrb, "CACHE_ROOT", tmp_path / "cache")

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        vrb,
        "get_builtin_graph",
        lambda *args, **kwargs: SimpleNamespace(graph_name="validate_range_boundaries"),
    )
    monkeypatch.setattr(
        vrb,
        "explain_graph_run",
        lambda graph, *, context, target: (
            captured.update({"target": target})
            or {
                "nodes": [
                    {
                        "node_name": "range_selection_bundle",
                        "node_kind": "aggregate",
                        "upstream_nodes": ["range_selected_rung"],
                        "fingerprint": "abc",
                        "cache_hit": False,
                        "would_execute": True,
                        "reason": "cache-miss",
                    }
                ]
            }
        ),
    )

    vrb.main()

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert captured["target"] == "range_selection_bundle"
    assert payload["wrapper_target"] == "selection"
    assert payload["resolved_target"] == "range_selection_bundle"
    assert payload["nodes"][0]["reason"] == "cache-miss"
