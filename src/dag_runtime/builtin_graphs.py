from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.dag_runtime.artifacts import write_csv_atomic, write_text_atomic
from src.dag_runtime.contracts import (
    CachePolicy,
    MutableScope,
    ReplayPolicyContract,
    ValidationPolicy,
    WindowPolicy,
)
from src.dag_runtime.fingerprints import default_node_fingerprint_payload
from src.dag_runtime.graph import GraphManifest
from src.dag_runtime.node import (
    GraphRunContext,
    NodeExecutionResult,
    NodeManifest,
    NodeOutput,
)
from src.validation.common.chart_core import save_figure_html
from src.validation.indicators.range_boundaries import validate_range_boundaries
from src.validation.indicators.regime import validate_regime
from src.validation.indicators.trend_state import validate_trend_state
from src.validation.indicators.sr_levels import summarize_sr_levels


def _source_node(graph_name: str, name: str, input_key: str) -> NodeManifest:
    def _fingerprint(
        manifest: NodeManifest,
        context: GraphRunContext,
        dependency_results: dict[str, NodeExecutionResult],
    ) -> dict[str, Any]:
        payload = default_node_fingerprint_payload(
            manifest, context, dependency_results
        )
        payload["runtime_config"] = {}
        return payload

    return NodeManifest(
        graph_name=graph_name,
        node_name=name,
        node_kind="source",
        semantic_class="A",
        inputs=(input_key,),
        upstream_nodes=(),
        output_artifacts=("frame",),
        fingerprint_fn=_fingerprint,
        cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
        validation_policy=ValidationPolicy(level="unit"),
        mutable_scope=MutableScope(scope="immutable"),
        compute_fn=lambda context, _deps, input_key=input_key: NodeOutput(
            frames={"frame": context.inputs[input_key].copy()}
        ),
    )


def build_live_stage_graph(
    *, instrument: str, swing_window: int, include_vp: bool
) -> GraphManifest:
    from src.indicators.pipelines import build_live as live

    graph_name = "live_pipeline"
    nodes: list[NodeManifest] = [_source_node(graph_name, "raw_input", "raw_input")]
    upstream = "raw_input"
    for stage in live._live_stages(
        instrument=instrument, swing_window=swing_window, include_vp=include_vp
    ):
        prev = upstream
        nodes.append(
            NodeManifest(
                graph_name=graph_name,
                node_name=stage.name,
                node_kind="compute",
                semantic_class=stage.policy.classification,
                inputs=(),
                upstream_nodes=(prev,),
                output_artifacts=("frame",),
                cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
                replay_policy=ReplayPolicyContract(
                    mode=(
                        "carried_state"
                        if stage.policy.carried_state
                        else (
                            "bounded_replay"
                            if stage.policy.replay_bars > 0
                            else "append_only_safe"
                        )
                    ),
                    replay_bars=stage.policy.replay_bars,
                    notes=stage.policy.notes,
                ),
                window_policy=WindowPolicy(mode="full"),
                validation_policy=ValidationPolicy(level="node_parity"),
                mutable_scope=MutableScope(scope="frontier_only"),
                compute_fn=lambda context, deps, stage=stage, prev=prev: NodeOutput(
                    frames={"frame": stage.fn(deps[prev].primary_frame().copy())}
                ),
            )
        )
        upstream = stage.name
    return GraphManifest(
        graph_name=graph_name, nodes=tuple(nodes), default_target=upstream
    )


def build_research_stage_graph(
    *, instrument: str, swing_window: int, include_vp: bool, include_avwap: bool
) -> GraphManifest:
    from src.indicators.pipelines import build_research as research

    graph_name = "research_pipeline"
    nodes: list[NodeManifest] = [_source_node(graph_name, "raw_input", "raw_input")]
    upstream = "raw_input"
    for stage in research._research_stages(
        instrument=instrument,
        swing_window=swing_window,
        include_vp=include_vp,
        include_avwap=include_avwap,
    ):
        prev = upstream
        nodes.append(
            NodeManifest(
                graph_name=graph_name,
                node_name=stage.name,
                node_kind="compute",
                semantic_class=stage.policy.classification,
                inputs=(),
                upstream_nodes=(prev,),
                output_artifacts=("frame",),
                cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
                replay_policy=ReplayPolicyContract(
                    mode=(
                        "carried_state"
                        if stage.policy.carried_state
                        else (
                            "bounded_replay"
                            if stage.policy.replay_bars > 0
                            else "append_only_safe"
                        )
                    ),
                    replay_bars=stage.policy.replay_bars,
                    notes=stage.policy.notes,
                ),
                validation_policy=ValidationPolicy(level="node_parity"),
                mutable_scope=MutableScope(
                    scope=(
                        "explicit_rebuild_only"
                        if stage.policy.classification == "C"
                        else "frontier_only"
                    )
                ),
                compute_fn=lambda context, deps, stage=stage, prev=prev: NodeOutput(
                    frames={"frame": stage.fn(deps[prev].primary_frame().copy())}
                ),
            )
        )
        upstream = stage.name
    return GraphManifest(
        graph_name=graph_name, nodes=tuple(nodes), default_target=upstream
    )


def build_range_boundaries_graph(*, instrument: str, timeframe: str) -> GraphManifest:
    from scripts import validate_range_boundaries as vrb

    graph_name = "validate_range_boundaries"
    nodes: list[NodeManifest] = [_source_node(graph_name, "raw_input", "raw_input")]

    def _runtime_config_fingerprint(*keys: str):
        def _fingerprint(
            manifest: NodeManifest,
            context: GraphRunContext,
            dependency_results: dict[str, NodeExecutionResult],
        ) -> dict[str, Any]:
            payload = default_node_fingerprint_payload(
                manifest, context, dependency_results
            )
            payload["runtime_config"] = {
                key: payload["runtime_config"][key]
                for key in keys
                if key in payload["runtime_config"]
            }
            return payload

        return _fingerprint

    no_runtime_config_fingerprint = _runtime_config_fingerprint()
    chart_runtime_config_fingerprint = _runtime_config_fingerprint(
        "html", "date_from", "plot_rows", "full", "out_dir"
    )
    csv_runtime_config_fingerprint = _runtime_config_fingerprint("write_csv", "out_dir")

    def context_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        raw = deps["raw_input"].primary_frame().copy()
        canonical = vrb._load_canonical_live_context(
            raw, instrument=instrument, timeframe=timeframe
        )
        frame = canonical.copy() if canonical is not None else vrb._build_context(raw)
        return NodeOutput(
            payload={
                "source": "canonical_live" if canonical is not None else "raw_rebuild"
            },
            frames={"frame": frame},
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            fingerprint_fn=no_runtime_config_fingerprint,
            replay_policy=ReplayPolicyContract(mode="bounded_replay", replay_bars=400),
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=context_compute,
        )
    )

    rung_names: list[str] = []
    step8e_a_names: list[str] = []
    step8e_b_names: list[str] = []
    for phase, extra in (("step8e_a", {}), ("step8e_b", vrb.STEP8E_B_RETUNE_PARAMS)):
        ladder = vrb._build_recovery_ladder()
        for label, overrides in ladder:
            node_name = f"range_rung_debug__{phase}__{label}"
            params = {**vrb.BASE_RECOVERY_PARAMS, **overrides, **extra}
            rung_names.append(node_name)
            if phase == "step8e_a":
                step8e_a_names.append(node_name)
            else:
                step8e_b_names.append(node_name)
            assessment_label = f"{phase}/{label}"

            def rung_compute(
                context: GraphRunContext,
                deps: dict[str, NodeExecutionResult],
                params: dict[str, Any] = params,
                assessment_label: str = assessment_label,
                phase: str = phase,
            ) -> NodeOutput:
                if (
                    phase == "step8e_b"
                    and not deps["range_retune_gate"].output.payload["retune_needed"]
                ):
                    return NodeOutput(
                        payload={
                            "params": params,
                            "summary": {},
                            "label": assessment_label,
                            "skipped": True,
                        },
                        frames={
                            "event_table": pd.DataFrame(),
                            "candidate_table": pd.DataFrame(),
                        },
                        profile_details={
                            "label": assessment_label,
                            "skipped": True,
                            "substage_seconds": {
                                "debug_collect": 0.0,
                                "pressure_imbalance_legacy": 0.0,
                                "pressure_imbalance_v2": 0.0,
                                "contract_scores": 0.0,
                                "summary_build": 0.0,
                            },
                        },
                    )
                result = vrb._run_debug_with_params(
                    deps["range_context"].primary_frame().copy(),
                    params,
                )
                return NodeOutput(
                    payload={
                        "params": params,
                        "summary": result["summary"],
                        "label": assessment_label,
                        "skipped": False,
                    },
                    frames={
                        "event_table": result["event_table"],
                        "candidate_table": result["candidate_table"],
                    },
                    profile_details={
                        "label": assessment_label,
                        "skipped": False,
                        **result.get("profile_details", {}),
                    },
                )

            nodes.append(
                NodeManifest(
                    graph_name=graph_name,
                    node_name=node_name,
                    node_kind="compute",
                    semantic_class="B",
                    inputs=(),
                    upstream_nodes=(
                        ("range_context",)
                        if phase == "step8e_a"
                        else ("range_context", "range_retune_gate")
                    ),
                    output_artifacts=("event_table", "candidate_table"),
                    fingerprint_fn=no_runtime_config_fingerprint,
                    replay_policy=ReplayPolicyContract(
                        mode="bounded_replay", replay_bars=400
                    ),
                    validation_policy=ValidationPolicy(level="node_parity"),
                    mutable_scope=MutableScope(scope="frontier_only"),
                    compute_fn=rung_compute,
                )
            )

    def retune_gate_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        assessments: list[dict[str, Any]] = []
        params_by_label: dict[str, dict[str, Any]] = {}
        for name in step8e_a_names:
            dep = deps[name]
            label = dep.output.payload["label"]
            summary = dep.output.payload["summary"]
            event_table = dep.output.frames["event_table"]
            assessment = vrb._assess_rung(
                label, {"summary": summary, "event_table": event_table}
            )
            assessments.append(assessment)
            params_by_label[label] = dep.output.payload["params"]
        best, has_valid = vrb._select_best_assessment(assessments)
        return NodeOutput(
            payload={
                "step8e_a_assessments": assessments,
                "step8e_a_best": best,
                "step8e_a_has_valid_rung": has_valid,
                "retune_needed": not has_valid,
                "params_by_label": params_by_label,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_retune_gate",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=tuple(step8e_a_names),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            validation_policy=ValidationPolicy(level="node_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=retune_gate_compute,
        )
    )

    def select_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        gate = deps["range_retune_gate"].output.payload
        assessments: list[dict[str, Any]] = list(gate["step8e_a_assessments"])
        params_by_label: dict[str, dict[str, Any]] = dict(gate["params_by_label"])
        has_valid = bool(gate["step8e_a_has_valid_rung"])
        best = dict(gate["step8e_a_best"])
        used_retune = not has_valid
        if used_retune:
            retune_assessments: list[dict[str, Any]] = []
            for name in step8e_b_names:
                dep = deps[name]
                if dep.output.payload.get("skipped"):
                    continue
                label = dep.output.payload["label"]
                summary = dep.output.payload["summary"]
                event_table = dep.output.frames["event_table"]
                assessment = vrb._assess_rung(
                    label, {"summary": summary, "event_table": event_table}
                )
                retune_assessments.append(assessment)
                params_by_label[label] = dep.output.payload["params"]
            if retune_assessments:
                assessments.extend(retune_assessments)
                best, has_valid = vrb._select_best_assessment(assessments)
        selected_label = str(best["label"])
        return NodeOutput(
            payload={
                "rung_assessments": assessments,
                "selected_label": (
                    selected_label if has_valid else "no_valid_contract_rung"
                ),
                "reporting_label": selected_label,
                "selected_params": params_by_label[selected_label],
                "has_valid_rung": has_valid,
                "used_retune": used_retune,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_selected_rung",
            node_kind="selection",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_retune_gate",) + tuple(step8e_b_names),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            validation_policy=ValidationPolicy(level="graph_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=select_compute,
        )
    )

    def selection_bundle_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        return NodeOutput(
            payload={"selected_rung": deps["range_selected_rung"].output.payload}
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_selection_bundle",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_selected_rung",),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=selection_bundle_compute,
        )
    )

    def selected_debug_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        selected_params = deps["range_selected_rung"].output.payload["selected_params"]
        result = vrb._run_debug_with_params(
            deps["range_context"].primary_frame().copy(),
            selected_params,
        )
        return NodeOutput(
            payload={"params": selected_params, "summary": result["summary"]},
            frames={
                "frame": result["frame"],
                "event_table": result["event_table"],
                "candidate_table": result["candidate_table"],
            },
            profile_details={
                "label": "selected_debug",
                "skipped": False,
                **result.get("profile_details", {}),
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_selected_debug",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_context", "range_selected_rung"),
            output_artifacts=("frame", "event_table", "candidate_table"),
            fingerprint_fn=no_runtime_config_fingerprint,
            replay_policy=ReplayPolicyContract(mode="bounded_replay", replay_bars=400),
            validation_policy=ValidationPolicy(level="graph_parity"),
            mutable_scope=MutableScope(scope="frontier_only"),
            compute_fn=selected_debug_compute,
        )
    )

    def forensics_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        forensics, short_high, long_medium = vrb._build_forensics_tables(event_table)
        scored = vrb._add_path_c2_candidate_scores(forensics)
        tagged = vrb._assign_contract_bucket_labels(scored)
        return NodeOutput(
            frames={
                "forensics": tagged,
                "short_high": short_high,
                "long_medium": long_medium,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_forensics",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_selected_debug",),
            output_artifacts=("forensics",),
            fingerprint_fn=no_runtime_config_fingerprint,
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=forensics_compute,
        )
    )

    def geometry_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        candidate_table = deps["range_selected_debug"].output.frames["candidate_table"]
        geometry = vrb._build_geometry_audit(full_df, event_table, candidate_table)
        geometry_candidate_comparison, geometry_candidate_summary = (
            vrb._build_geometry_candidate_comparison(
                full_df,
                geometry,
            )
        )
        return NodeOutput(
            frames={
                "geometry_audit": geometry,
                "geometry_candidate_comparison": geometry_candidate_comparison,
                "geometry_candidate_summary": geometry_candidate_summary,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_geometry_audit",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_selected_debug",),
            output_artifacts=(
                "geometry_audit",
                "geometry_candidate_comparison",
                "geometry_candidate_summary",
            ),
            fingerprint_fn=no_runtime_config_fingerprint,
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=geometry_compute,
        )
    )

    def active_truth_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        full_df = deps["range_selected_debug"].output.frames["frame"]
        geometry = deps["range_geometry_audit"].output.frames["geometry_audit"]
        active_truth, doctrine = vrb._build_active_truth_audit(full_df, geometry)
        geometry_candidate_truth = vrb._build_geometry_candidate_active_truth_summary(
            full_df,
            geometry,
            active_truth,
        )
        return NodeOutput(
            frames={
                "active_truth_audit": active_truth,
                "doctrine_report": doctrine,
                "geometry_candidate_truth_summary": geometry_candidate_truth,
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_active_truth_audit",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_selected_debug", "range_geometry_audit"),
            output_artifacts=(
                "active_truth_audit",
                "doctrine_report",
                "geometry_candidate_truth_summary",
            ),
            fingerprint_fn=no_runtime_config_fingerprint,
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=active_truth_compute,
        )
    )

    def coverage_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        rung_results = []
        assessments = deps["range_selected_rung"].output.payload["rung_assessments"]
        assessment_map = {str(item["label"]): item for item in assessments}
        for name in rung_names:
            dep = deps[name]
            label = dep.output.payload["label"]
            rung_results.append(
                (
                    label,
                    {
                        "summary": dep.output.payload["summary"],
                        "event_table": dep.output.frames["event_table"],
                    },
                )
            )
        coverage = vrb._build_coverage_regime_report(
            rung_results, list(assessment_map.values())
        )
        return NodeOutput(frames={"coverage_regime_report": coverage})

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_coverage_regime_report",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=tuple(rung_names) + ("range_selected_rung",),
            output_artifacts=("coverage_regime_report",),
            fingerprint_fn=no_runtime_config_fingerprint,
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=coverage_compute,
        )
    )

    def ranking_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        forensics = deps["range_forensics"].output.frames["forensics"]
        repair_gates, recommendation = vrb._evaluate_path_c2_candidates(forensics)
        geometry_candidate_comparison = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_comparison"
        ]
        geometry_ranking_preservation = vrb._build_geometry_ranking_preservation_report(
            forensics,
            geometry_candidate_comparison,
        )
        return NodeOutput(
            payload={"ranking_repair_recommendation": recommendation},
            frames={
                "ranking_report": vrb._build_ranking_disagreement_report(forensics),
                "ranking_rebase_report": vrb._build_ranking_rebase_comparison_report(
                    forensics
                ),
                "ranking_repair_report": vrb._build_path_c2_candidate_report(forensics),
                "ranking_repair_gates": repair_gates,
                "agreement_report": vrb._build_agreement_matrix(forensics),
                "bucket_lift_report": vrb._build_bucket_lift_report(forensics),
                "family_report": vrb._build_family_comparison_report(forensics),
                "path_c2_archetype_report": vrb._build_path_c2_archetype_report(
                    forensics
                ),
                "geometry_ranking_preservation": geometry_ranking_preservation,
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_ranking_bundle",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("range_forensics", "range_geometry_audit"),
            output_artifacts=("ranking_bundle",),
            fingerprint_fn=no_runtime_config_fingerprint,
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=ranking_compute,
        )
    )

    def diagnostics_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        forensics = deps["range_forensics"].output.frames["forensics"]
        short_high = deps["range_forensics"].output.frames["short_high"]
        long_medium = deps["range_forensics"].output.frames["long_medium"]
        geometry_audit = deps["range_geometry_audit"].output.frames["geometry_audit"]
        geometry_candidate_comparison = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_comparison"
        ]
        geometry_candidate_summary = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_summary"
        ]
        active_truth_audit = deps["range_active_truth_audit"].output.frames[
            "active_truth_audit"
        ]
        doctrine_report = deps["range_active_truth_audit"].output.frames[
            "doctrine_report"
        ]
        geometry_candidate_truth_summary = deps[
            "range_active_truth_audit"
        ].output.frames["geometry_candidate_truth_summary"]
        ranking_report = deps["range_ranking_bundle"].output.frames["ranking_report"]
        geometry_ranking_preservation = deps["range_ranking_bundle"].output.frames[
            "geometry_ranking_preservation"
        ]
        downstream_summary = deps["range_downstream_usefulness"].output.payload[
            "summary"
        ]
        geometry_candidate_downstream_summary = deps[
            "range_downstream_usefulness"
        ].output.frames["geometry_candidate_downstream_summary"]
        path_summary = vrb._primary_path_from_reports(
            active_truth_audit,
            doctrine_report,
            ranking_report,
            downstream_summary,
            geometry_audit,
        )
        geometry_candidate_gate_report, geometry_candidate_recommendation = (
            vrb._build_geometry_candidate_gate_report(
                geometry_candidate_summary,
                geometry_candidate_truth_summary,
                geometry_candidate_downstream_summary,
                geometry_ranking_preservation,
            )
        )

        recommended_improvement_ids: list[Any] = []
        if (
            geometry_candidate_recommendation != "no_candidate_passed"
            and not geometry_candidate_comparison.empty
        ):
            recommended_rows = geometry_candidate_comparison[
                geometry_candidate_comparison["candidate_family"]
                == geometry_candidate_recommendation
            ][["range_id", "geometry_chart_fit_score"]].copy()
            legacy_rows = geometry_candidate_comparison[
                geometry_candidate_comparison["candidate_family"] == "g1_legacy"
            ][["range_id", "geometry_chart_fit_score"]].copy()
            legacy_rows.rename(
                columns={"geometry_chart_fit_score": "legacy_geometry_chart_fit_score"},
                inplace=True,
            )
            recommended_improvement_ids = (
                recommended_rows.merge(legacy_rows, on="range_id", how="left")
                .assign(
                    improvement_score=(
                        pd.to_numeric(
                            recommended_rows["geometry_chart_fit_score"],
                            errors="coerce",
                        )
                        - pd.to_numeric(
                            recommended_rows.merge(
                                legacy_rows, on="range_id", how="left"
                            )["legacy_geometry_chart_fit_score"],
                            errors="coerce",
                        ).fillna(0.0)
                    )
                )
                .sort_values("improvement_score", ascending=False)["range_id"]
                .head(4)
                .tolist()
            )
        geometry_samples = geometry_audit[
            pd.to_numeric(geometry_audit["range_id"], errors="coerce").isin(
                recommended_improvement_ids
            )
        ].copy()
        if geometry_samples.empty:
            geometry_samples = (
                pd.concat(
                    [
                        forensics.nlargest(3, "rb_plausibility_score"),
                        forensics[
                            forensics["contract_bucket"] == "strong_false_positive"
                        ].head(3),
                    ],
                    ignore_index=True,
                )
                .drop_duplicates(subset=["range_id"])
                .head(6)
            )
        refresh_ids: list[Any] = []
        if not active_truth_audit.empty:
            refresh_rank = active_truth_audit.assign(
                refresh_gain=(
                    pd.to_numeric(
                        active_truth_audit["bounded_refresh_visual_plausibility_score"],
                        errors="coerce",
                    )
                    - pd.to_numeric(
                        active_truth_audit["frozen_visual_plausibility_score"],
                        errors="coerce",
                    )
                )
            ).sort_values("refresh_gain", ascending=False)
            refresh_ids = refresh_rank["range_id"].head(6).tolist()
        refresh_samples = geometry_audit[
            pd.to_numeric(geometry_audit["range_id"], errors="coerce").isin(refresh_ids)
        ].copy()
        downstream_usefulness = deps["range_downstream_usefulness"].output.frames[
            "downstream_usefulness"
        ]
        helpful = downstream_usefulness[
            pd.to_numeric(
                downstream_usefulness.get("interpretive_value_flag"), errors="coerce"
            )
            .fillna(0)
            .eq(1)
        ].nlargest(2, "rb_plausibility_score")
        little = downstream_usefulness[
            pd.to_numeric(
                downstream_usefulness.get("interpretive_value_flag"), errors="coerce"
            )
            .fillna(0)
            .eq(0)
        ].head(2)
        misled = downstream_usefulness[
            downstream_usefulness.get("contract_bucket", pd.Series(dtype="object")).eq(
                "strong_false_positive"
            )
        ].head(2)
        downstream_samples = (
            pd.concat([helpful, little, misled], ignore_index=True)
            .drop_duplicates(subset=["range_id"])
            .head(6)
        )

        if not forensics.empty:
            shortest_df = forensics.nsmallest(20, "duration_bars")
            longest_df = forensics.nlargest(20, "duration_bars")
            strongest_df = forensics.nlargest(20, "strength")
            weakest_df = forensics.nsmallest(20, "strength")
            ranging_short_lived_df = (
                forensics[
                    pd.to_numeric(forensics["confirm_regime"], errors="coerce").eq(0)
                    & pd.to_numeric(forensics["duration_bars"], errors="coerce").le(2)
                ]
                .sort_values(["duration_bars", "strength"], ascending=[True, False])
                .head(20)
            )
            plausibility_df = forensics.nlargest(20, "rb_plausibility_score")
            monitor_df = forensics.nlargest(20, "rb_monitor_worthiness_score")
            micro_box_df = forensics.nlargest(20, "rb_micro_box_risk_score")
            late_fragility_df = forensics.nlargest(
                20, "rb_late_confirm_fragility_score"
            )
        else:
            shortest_df = pd.DataFrame()
            longest_df = pd.DataFrame()
            strongest_df = pd.DataFrame()
            weakest_df = pd.DataFrame()
            ranging_short_lived_df = pd.DataFrame()
            plausibility_df = pd.DataFrame()
            monitor_df = pd.DataFrame()
            micro_box_df = pd.DataFrame()
            late_fragility_df = pd.DataFrame()

        forensics_bundle = vrb._bundle_named_reports(
            {
                "shortest_lived": shortest_df,
                "longest_lived": longest_df,
                "strongest": strongest_df,
                "weakest": weakest_df,
                "ranging_short_lived": ranging_short_lived_df,
                "short_lived_high_strength": short_high,
                "long_lived_medium_strength": long_medium,
                "highest_plausibility": plausibility_df,
                "highest_monitor_worthiness": monitor_df,
                "highest_micro_box_risk": micro_box_df,
                "highest_late_confirm_fragility": late_fragility_df,
            }
        )
        ranking_bundle = vrb._bundle_named_reports(
            {
                "ranking_disagreement": deps["range_ranking_bundle"].output.frames[
                    "ranking_report"
                ],
                "ranking_rebase": deps["range_ranking_bundle"].output.frames[
                    "ranking_rebase_report"
                ],
                "ranking_repair_candidates": deps["range_ranking_bundle"].output.frames[
                    "ranking_repair_report"
                ],
                "ranking_repair_gates": deps["range_ranking_bundle"].output.frames[
                    "ranking_repair_gates"
                ],
                "ranking_agreement": deps["range_ranking_bundle"].output.frames[
                    "agreement_report"
                ],
                "bucket_lift": deps["range_ranking_bundle"].output.frames[
                    "bucket_lift_report"
                ],
                "family_comparison": deps["range_ranking_bundle"].output.frames[
                    "family_report"
                ],
                "path_c2_archetypes": deps["range_ranking_bundle"].output.frames[
                    "path_c2_archetype_report"
                ],
                "geometry_ranking_preservation": deps[
                    "range_ranking_bundle"
                ].output.frames["geometry_ranking_preservation"],
                "geometry_candidate_gate_report": geometry_candidate_gate_report,
                "geometry_candidate_summary": geometry_candidate_summary,
                "geometry_candidate_truth_summary": geometry_candidate_truth_summary,
                "geometry_candidate_downstream_summary": geometry_candidate_downstream_summary,
            }
        )
        return NodeOutput(
            payload={
                "interpretability_summary": vrb._build_interpretability_metrics_summary(
                    forensics
                ),
                "contract_bucket_summary": vrb._build_contract_bucket_summary(
                    forensics
                ),
                "archetype_summary": vrb._build_archetype_summary(
                    short_high, long_medium
                ),
                "alignment_audit": vrb._build_viability_alignment_audit(
                    short_high, long_medium
                ),
                "pressure_audit": vrb._build_pressure_alignment_audit(
                    short_high, long_medium
                ),
                "path_summary": path_summary,
                "geometry_candidate_recommendation": geometry_candidate_recommendation,
            },
            frames={
                "geometry_samples": geometry_samples,
                "refresh_samples": refresh_samples,
                "downstream_samples": downstream_samples,
                "forensics_bundle": forensics_bundle,
                "ranking_bundle": ranking_bundle,
                "geometry_candidate_comparison": geometry_candidate_comparison,
                "geometry_candidate_summary": geometry_candidate_summary,
                "geometry_candidate_gate_report": geometry_candidate_gate_report,
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_diagnostics_bundle",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=(
                "range_forensics",
                "range_geometry_audit",
                "range_active_truth_audit",
                "range_ranking_bundle",
                "range_downstream_usefulness",
            ),
            output_artifacts=(
                "payload",
                "geometry_samples",
                "refresh_samples",
                "downstream_samples",
                "forensics_bundle",
                "ranking_bundle",
                "geometry_candidate_comparison",
                "geometry_candidate_summary",
                "geometry_candidate_gate_report",
            ),
            fingerprint_fn=no_runtime_config_fingerprint,
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=diagnostics_compute,
        )
    )

    def downstream_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        full_df = deps["range_selected_debug"].output.frames["frame"]
        forensics = deps["range_forensics"].output.frames["forensics"]
        downstream_usefulness, summary = vrb._build_downstream_usefulness_report(
            full_df, forensics
        )
        geometry_audit = deps["range_geometry_audit"].output.frames["geometry_audit"]
        geometry_candidate_downstream_summary = (
            vrb._build_geometry_candidate_downstream_summary(full_df, geometry_audit)
        )
        return NodeOutput(
            payload={"summary": summary},
            frames={
                "downstream_usefulness": downstream_usefulness,
                "geometry_candidate_downstream_summary": geometry_candidate_downstream_summary,
            },
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_downstream_usefulness",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_forensics",
                "range_geometry_audit",
            ),
            output_artifacts=(
                "downstream_usefulness",
                "geometry_candidate_downstream_summary",
            ),
            fingerprint_fn=no_runtime_config_fingerprint,
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=downstream_compute,
        )
    )

    def analysis_bundle_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        return NodeOutput(
            payload={
                "selected_summary": deps["range_selected_debug"].output.payload[
                    "summary"
                ],
                "downstream_summary": deps[
                    "range_downstream_usefulness"
                ].output.payload["summary"],
                "diagnostics": deps["range_diagnostics_bundle"].output.payload,
                "ranking_repair_recommendation": deps[
                    "range_ranking_bundle"
                ].output.payload["ranking_repair_recommendation"],
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_analysis_bundle",
            node_kind="aggregate",
            semantic_class="B",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_geometry_audit",
                "range_active_truth_audit",
                "range_coverage_regime_report",
                "range_ranking_bundle",
                "range_downstream_usefulness",
                "range_diagnostics_bundle",
            ),
            output_artifacts=("payload",),
            fingerprint_fn=no_runtime_config_fingerprint,
            cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=analysis_bundle_compute,
        )
    )

    def chart_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("html", False):
            return NodeOutput(
                payload={
                    "html_path": None,
                    "summary": deps["range_selected_debug"].output.payload["summary"],
                }
            )
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        html_path = (
            out_dir / f"range_boundaries_validation_{instrument}_{timeframe}.html"
        )
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        candidate_table = deps["range_selected_debug"].output.frames["candidate_table"]
        selected = deps["range_selected_rung"].output.payload["reporting_label"]
        plot_rows = int(context.config.get("plot_rows", 300))
        date_from = pd.Timestamp(
            context.config.get("date_from", "2026-01-01"), tz="UTC"
        )
        plot_df = full_df[full_df["timestamp"] >= date_from].copy()
        if plot_df.empty:
            plot_df = full_df.tail(plot_rows).copy()
        elif not context.config.get("full", False):
            plot_df = plot_df.tail(plot_rows).copy()
        result = validate_range_boundaries(
            plot_df,
            outpath=html_path,
            title=f"Range Boundary Validation — {instrument} {timeframe} [{selected}]",
            summary_df=full_df,
            event_table=event_table,
            candidate_table=candidate_table,
        )
        return NodeOutput(
            payload={
                "html_path": str(result["html_path"]),
                "summary": result["summary"],
            },
            artifacts={"html": result["html_path"]},
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_main_chart",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("range_selected_debug", "range_selected_rung"),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=CachePolicy(materialize=True, artifact_kind="report"),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=chart_compute,
        )
    )

    def geometry_chart_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("html", False):
            return NodeOutput(payload={"html_path": None})
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        html_path = (
            out_dir
            / f"range_boundaries_geometry_chart_pack_{instrument}_{timeframe}.html"
        )
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        sample_df = deps["range_diagnostics_bundle"].output.frames["geometry_samples"]
        written = vrb._plot_audit_chart_pack(
            full_df,
            sample_df,
            event_table=event_table,
            outpath=html_path,
            title=f"Range Boundary Geometry Audit — {instrument} {timeframe}",
            active_truth_audit=deps["range_active_truth_audit"].output.frames[
                "active_truth_audit"
            ],
            show_geometry_candidates=True,
            highlight_candidate_family=deps[
                "range_diagnostics_bundle"
            ].output.payload.get("geometry_candidate_recommendation"),
        )
        return NodeOutput(
            payload={"html_path": str(written)}, artifacts={"html": written}
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_geometry_chart_pack",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_diagnostics_bundle",
                "range_active_truth_audit",
            ),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=CachePolicy(materialize=True, artifact_kind="report"),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=geometry_chart_compute,
        )
    )

    def refresh_chart_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("html", False):
            return NodeOutput(payload={"html_path": None})
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        html_path = (
            out_dir
            / f"range_boundaries_refresh_chart_pack_{instrument}_{timeframe}.html"
        )
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        geometry_samples = deps["range_diagnostics_bundle"].output.frames[
            "geometry_samples"
        ]
        refresh_samples = deps["range_diagnostics_bundle"].output.frames[
            "refresh_samples"
        ]
        sample_df = (
            refresh_samples if not refresh_samples.empty else geometry_samples.head(4)
        )
        written = vrb._plot_audit_chart_pack(
            full_df,
            sample_df,
            event_table=event_table,
            outpath=html_path,
            title=f"Range Boundary Frozen vs Refresh Audit — {instrument} {timeframe}",
            active_truth_audit=deps["range_active_truth_audit"].output.frames[
                "active_truth_audit"
            ],
        )
        return NodeOutput(
            payload={"html_path": str(written)}, artifacts={"html": written}
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_refresh_chart_pack",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_diagnostics_bundle",
                "range_active_truth_audit",
            ),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=CachePolicy(materialize=True, artifact_kind="report"),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=refresh_chart_compute,
        )
    )

    def downstream_chart_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("html", False):
            return NodeOutput(payload={"html_path": None})
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        html_path = (
            out_dir
            / f"range_boundaries_downstream_chart_pack_{instrument}_{timeframe}.html"
        )
        full_df = deps["range_selected_debug"].output.frames["frame"]
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        geometry_samples = deps["range_diagnostics_bundle"].output.frames[
            "geometry_samples"
        ]
        downstream_samples = deps["range_diagnostics_bundle"].output.frames[
            "downstream_samples"
        ]
        sample_df = (
            downstream_samples
            if not downstream_samples.empty
            else geometry_samples.head(4)
        )
        written = vrb._plot_audit_chart_pack(
            full_df,
            sample_df,
            event_table=event_table,
            outpath=html_path,
            title=f"Range Boundary Downstream Operator Pack — {instrument} {timeframe}",
            active_truth_audit=deps["range_active_truth_audit"].output.frames[
                "active_truth_audit"
            ],
        )
        return NodeOutput(
            payload={"html_path": str(written)}, artifacts={"html": written}
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_downstream_chart_pack",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_diagnostics_bundle",
                "range_active_truth_audit",
            ),
            output_artifacts=("html",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=CachePolicy(materialize=True, artifact_kind="report"),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=downstream_chart_compute,
        )
    )

    def chart_bundle_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        artifacts: dict[str, Any] = {}
        for node_name in (
            "range_main_chart",
            "range_geometry_chart_pack",
            "range_refresh_chart_pack",
            "range_downstream_chart_pack",
        ):
            artifacts[node_name] = {
                **deps[node_name].output.payload,
                "cache_hit": deps[node_name].cache_hit,
            }
        return NodeOutput(payload={"artifacts": artifacts})

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_chart_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_main_chart",
                "range_geometry_chart_pack",
                "range_refresh_chart_pack",
                "range_downstream_chart_pack",
            ),
            output_artifacts=("payload",),
            fingerprint_fn=chart_runtime_config_fingerprint,
            cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
            validation_policy=ValidationPolicy(level="report"),
            compute_fn=chart_bundle_compute,
        )
    )

    def csv_bundle_compute(
        context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        if not context.config.get("write_csv", False):
            return NodeOutput(payload={"artifact_paths": {}})
        out_dir = Path(context.config.get("out_dir", "notebooks/foundation"))
        event_table = deps["range_selected_debug"].output.frames["event_table"]
        candidate_table = deps["range_selected_debug"].output.frames["candidate_table"]
        geometry_audit = deps["range_geometry_audit"].output.frames["geometry_audit"]
        geometry_candidate_comparison = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_comparison"
        ]
        geometry_candidate_summary = deps["range_geometry_audit"].output.frames[
            "geometry_candidate_summary"
        ]
        active_truth_audit = deps["range_active_truth_audit"].output.frames[
            "active_truth_audit"
        ]
        doctrine_report = deps["range_active_truth_audit"].output.frames[
            "doctrine_report"
        ]
        geometry_candidate_truth_summary = deps[
            "range_active_truth_audit"
        ].output.frames["geometry_candidate_truth_summary"]
        coverage_regime_report = deps["range_coverage_regime_report"].output.frames[
            "coverage_regime_report"
        ]
        forensics_bundle = deps["range_diagnostics_bundle"].output.frames[
            "forensics_bundle"
        ]
        ranking_bundle = deps["range_diagnostics_bundle"].output.frames[
            "ranking_bundle"
        ]
        geometry_candidate_gate_report = deps["range_diagnostics_bundle"].output.frames[
            "geometry_candidate_gate_report"
        ]
        downstream_usefulness = deps["range_downstream_usefulness"].output.frames[
            "downstream_usefulness"
        ]
        geometry_candidate_downstream_summary = deps[
            "range_downstream_usefulness"
        ].output.frames["geometry_candidate_downstream_summary"]
        reporting_label = deps["range_selected_rung"].output.payload["reporting_label"]
        selected_label = deps["range_selected_rung"].output.payload["selected_label"]
        best_assessment = next(
            item
            for item in deps["range_selected_rung"].output.payload["rung_assessments"]
            if str(item["label"]) == str(reporting_label)
        )
        diagnosis_text = vrb._build_diagnosis_memo_text(
            geometry_audit=geometry_audit,
            active_truth_audit=active_truth_audit,
            doctrine_report=doctrine_report,
            coverage_regime_report=coverage_regime_report,
            ranking_report=deps["range_ranking_bundle"].output.frames["ranking_report"],
            family_report=deps["range_ranking_bundle"].output.frames["family_report"],
            downstream_summary=deps["range_downstream_usefulness"].output.payload[
                "summary"
            ],
            path_summary=deps["range_diagnostics_bundle"].output.payload[
                "path_summary"
            ],
            selected_label=reporting_label,
        )
        ranking_text = "\n".join(
            [
                "# Step 8F Path C Ranking Rebase",
                "",
                "## Score Architecture",
                "- `range_strength_structure`: retained structure-side quality block.",
                "- `range_strength_monitorability`: monitor-worthiness proxy rebased toward interpretability truth.",
                "- `range_strength_semantic`: semantic keep-watching block favoring plausible low-risk ranges.",
                "- `range_strength_viability`: rebased visible viability score.",
                "- `range_strength`: rebased production ranking score.",
                "",
                "## Mapping Doctrine",
                "- Positive: plausibility-adjacent monitorability, boundary relevance, low pressure imbalance, stable two-sided structure.",
                "- Negative: micro-box risk proxy built from low confirm latency, minimal touches, narrow/weak width, and hyper-fresh tidy behavior.",
                "- Legacy formation/viability remain exported as `_legacy` comparison fields.",
                "",
                "## Acceptance Snapshot",
                f"- Selected/reporting rung: `{reporting_label}`",
                f"- Contract-selected rung: `{selected_label}`",
                f"- Long-lived vs short-lived final strength gate passed: `{bool(best_assessment.get('strength_not_badly_inverted'))}`",
                f"- Coverage stable: confirmed=`{best_assessment.get('confirmed_ranges')}`, active_rows=`{best_assessment.get('active_rows')}`",
            ]
        )
        ranking_repair_text = "\n".join(
            [
                "# Step 8F.2 Path C2 Ranking Repair",
                "",
                f"- Recommendation: `{deps['range_ranking_bundle'].output.payload['ranking_repair_recommendation']}`",
                "",
                "## Candidate Gate Results",
                (
                    deps["range_ranking_bundle"]
                    .output.frames["ranking_repair_gates"]
                    .to_string(index=False)
                    if not deps["range_ranking_bundle"]
                    .output.frames["ranking_repair_gates"]
                    .empty
                    else "No gate results."
                ),
                "",
                "## Candidate Top-Rank Report",
                (
                    deps["range_ranking_bundle"]
                    .output.frames["ranking_repair_report"]
                    .to_string(index=False)
                    if not deps["range_ranking_bundle"]
                    .output.frames["ranking_repair_report"]
                    .empty
                    else "No repair report."
                ),
                "",
                "## Candidate Archetype Report",
                (
                    deps["range_ranking_bundle"]
                    .output.frames["path_c2_archetype_report"]
                    .to_string(index=False)
                    if not deps["range_ranking_bundle"]
                    .output.frames["path_c2_archetype_report"]
                    .empty
                    else "No archetype report."
                ),
            ]
        )
        geometry_candidate_bundle = vrb._bundle_named_reports(
            {
                "geometry_candidate_comparison": geometry_candidate_comparison,
                "geometry_candidate_summary": geometry_candidate_summary,
                "geometry_candidate_truth_summary": geometry_candidate_truth_summary,
                "geometry_candidate_downstream_summary": geometry_candidate_downstream_summary,
                "geometry_ranking_preservation": deps[
                    "range_ranking_bundle"
                ].output.frames["geometry_ranking_preservation"],
                "geometry_candidate_gate_report": geometry_candidate_gate_report,
            },
            group_col="geometry_report_group",
        )
        artifacts = {
            "events": out_dir / f"range_boundaries_events_{instrument}_{timeframe}.csv",
            "candidates": out_dir
            / f"range_boundaries_candidates_{instrument}_{timeframe}.csv",
            "forensics_bundle": out_dir
            / f"range_boundaries_forensics_bundle_{instrument}_{timeframe}.csv",
            "geometry_audit": out_dir
            / f"range_boundaries_geometry_audit_{instrument}_{timeframe}.csv",
            "geometry_candidates": out_dir
            / f"range_boundaries_geometry_candidates_{instrument}_{timeframe}.csv",
            "active_truth": out_dir
            / f"range_boundaries_active_truth_audit_{instrument}_{timeframe}.csv",
            "doctrine": out_dir
            / f"range_boundaries_frozen_vs_refresh_{instrument}_{timeframe}.csv",
            "coverage": out_dir
            / f"range_boundaries_coverage_regimes_{instrument}_{timeframe}.csv",
            "ranking_bundle": out_dir
            / f"range_boundaries_ranking_bundle_{instrument}_{timeframe}.csv",
            "downstream": out_dir
            / f"range_boundaries_downstream_usefulness_{instrument}_{timeframe}.csv",
            "diagnosis_memo": out_dir
            / f"range_boundaries_step8x_diagnosis_{instrument}_{timeframe}.md",
            "ranking_memo": out_dir
            / f"range_boundaries_path_c_ranking_rebase_{instrument}_{timeframe}.md",
            "ranking_repair_memo": out_dir
            / f"range_boundaries_path_c2_ranking_repair_{instrument}_{timeframe}.md",
        }
        write_csv_atomic(event_table, artifacts["events"])
        write_csv_atomic(candidate_table, artifacts["candidates"])
        write_csv_atomic(forensics_bundle, artifacts["forensics_bundle"])
        write_csv_atomic(geometry_audit, artifacts["geometry_audit"])
        write_csv_atomic(geometry_candidate_bundle, artifacts["geometry_candidates"])
        write_csv_atomic(active_truth_audit, artifacts["active_truth"])
        write_csv_atomic(doctrine_report, artifacts["doctrine"])
        write_csv_atomic(coverage_regime_report, artifacts["coverage"])
        write_csv_atomic(ranking_bundle, artifacts["ranking_bundle"])
        write_csv_atomic(downstream_usefulness, artifacts["downstream"])
        write_text_atomic(diagnosis_text, artifacts["diagnosis_memo"])
        write_text_atomic(ranking_text, artifacts["ranking_memo"])
        write_text_atomic(ranking_repair_text, artifacts["ranking_repair_memo"])
        return NodeOutput(
            payload={
                "artifact_paths": {name: str(path) for name, path in artifacts.items()}
            },
            artifacts=artifacts,
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_csv_bundle",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selected_debug",
                "range_selected_rung",
                "range_geometry_audit",
                "range_active_truth_audit",
                "range_coverage_regime_report",
                "range_ranking_bundle",
                "range_downstream_usefulness",
                "range_diagnostics_bundle",
            ),
            output_artifacts=("artifacts",),
            fingerprint_fn=csv_runtime_config_fingerprint,
            cache_policy=CachePolicy(materialize=True, artifact_kind="report"),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=csv_bundle_compute,
        )
    )

    def validation_bundle_compute(
        _context: GraphRunContext, deps: dict[str, NodeExecutionResult]
    ) -> NodeOutput:
        return NodeOutput(
            payload={
                "selected_rung": deps["range_selection_bundle"].output.payload[
                    "selected_rung"
                ],
                "selected_summary": deps["range_analysis_bundle"].output.payload[
                    "selected_summary"
                ],
                "downstream_summary": deps["range_analysis_bundle"].output.payload[
                    "downstream_summary"
                ],
                "diagnostics": deps["range_analysis_bundle"].output.payload[
                    "diagnostics"
                ],
                "artifacts": {
                    **deps["range_chart_bundle"].output.payload["artifacts"],
                    "range_csv_bundle": {
                        **deps["range_csv_bundle"].output.payload,
                        "cache_hit": deps["range_csv_bundle"].cache_hit,
                    },
                },
            }
        )

    nodes.append(
        NodeManifest(
            graph_name=graph_name,
            node_name="range_validation_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "range_selection_bundle",
                "range_analysis_bundle",
                "range_chart_bundle",
                "range_csv_bundle",
            ),
            output_artifacts=("payload",),
            fingerprint_fn=_runtime_config_fingerprint(
                "html", "write_csv", "date_from", "plot_rows", "full", "out_dir"
            ),
            cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=validation_bundle_compute,
        )
    )

    return GraphManifest(
        graph_name=graph_name,
        nodes=tuple(nodes),
        default_target="range_validation_bundle",
    )


def build_regime_validation_graph(*, instrument: str, timeframe: str) -> GraphManifest:
    from scripts import validate_regime as vr

    graph_name = "validate_regime"
    nodes: list[NodeManifest] = [_source_node(graph_name, "raw_input", "raw_input")]

    def make_context(include_research_only: bool, node_name: str) -> NodeManifest:
        return NodeManifest(
            graph_name=graph_name,
            node_name=node_name,
            node_kind="compute",
            semantic_class="C" if include_research_only else "B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps, include_research_only=include_research_only: NodeOutput(
                frames={
                    "frame": (
                        canonical.copy()
                        if (
                            canonical := vr._load_canonical_context(
                                deps["raw_input"].primary_frame().copy(),
                                instrument=instrument,
                                timeframe=timeframe,
                                dataset="research" if include_research_only else "live",
                            )
                        )
                        is not None
                        else vr._build_context(
                            deps["raw_input"].primary_frame().copy(),
                            include_research_only=include_research_only,
                        )
                    )
                }
            ),
        )

    nodes.extend(
        [
            make_context(False, "regime_live_context"),
            make_context(True, "regime_research_context"),
            NodeManifest(
                graph_name=graph_name,
                node_name="regime_summary",
                node_kind="aggregate",
                semantic_class="C",
                inputs=(),
                upstream_nodes=("regime_live_context", "regime_research_context"),
                output_artifacts=("payload",),
                cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
                validation_policy=ValidationPolicy(level="graph_parity"),
                compute_fn=lambda context, deps: NodeOutput(
                    payload=validate_regime(
                        (
                            deps["regime_research_context"].primary_frame()
                            if context.config.get("full", False)
                            else deps["regime_research_context"]
                            .primary_frame()
                            .tail(int(context.config.get("plot_rows", vr.PLOT_ROWS)))
                        ),
                        summary_df=deps["regime_research_context"].primary_frame(),
                        live_df=deps["regime_live_context"].primary_frame(),
                        research_df=deps["regime_research_context"].primary_frame(),
                        outpath=None,
                        title=f"Regime Validation — {instrument} {timeframe}",
                        synthetic_summary=vr._synthetic_fixture_summary(),
                    )
                ),
            ),
            NodeManifest(
                graph_name=graph_name,
                node_name="regime_main_chart",
                node_kind="report",
                semantic_class="C",
                inputs=(),
                upstream_nodes=(
                    "regime_summary",
                    "regime_live_context",
                    "regime_research_context",
                ),
                output_artifacts=("html",),
                cache_policy=CachePolicy(materialize=True, artifact_kind="report"),
                validation_policy=ValidationPolicy(level="report"),
                mutable_scope=MutableScope(scope="immutable"),
                compute_fn=lambda context, deps: (
                    NodeOutput(payload={"html_path": None})
                    if not context.config.get("html", False)
                    else NodeOutput(
                        payload={
                            "html_path": str(
                                validate_regime(
                                    (
                                        deps["regime_research_context"].primary_frame()
                                        if context.config.get("full", False)
                                        else deps["regime_research_context"]
                                        .primary_frame()
                                        .tail(
                                            int(
                                                context.config.get(
                                                    "plot_rows", vr.PLOT_ROWS
                                                )
                                            )
                                        )
                                    ),
                                    summary_df=deps[
                                        "regime_research_context"
                                    ].primary_frame(),
                                    live_df=deps["regime_live_context"].primary_frame(),
                                    research_df=deps[
                                        "regime_research_context"
                                    ].primary_frame(),
                                    outpath=Path(
                                        context.config.get(
                                            "out_dir", "notebooks/foundation"
                                        )
                                    )
                                    / f"regime_validation_{instrument}_{timeframe}.html",
                                    title=f"Regime Validation — {instrument} {timeframe}",
                                    synthetic_summary=vr._synthetic_fixture_summary(),
                                )["html_path"]
                            )
                        },
                        artifacts={
                            "html": Path(
                                context.config.get("out_dir", "notebooks/foundation")
                            )
                            / f"regime_validation_{instrument}_{timeframe}.html"
                        },
                    )
                ),
            ),
            NodeManifest(
                graph_name=graph_name,
                node_name="regime_validation_bundle",
                node_kind="aggregate",
                semantic_class="C",
                inputs=(),
                upstream_nodes=("regime_summary", "regime_main_chart"),
                output_artifacts=("payload",),
                cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
                validation_policy=ValidationPolicy(level="graph_parity"),
                compute_fn=lambda context, deps: NodeOutput(
                    payload={
                        "summary": deps["regime_summary"].output.payload["summary"],
                        "html_path": deps["regime_main_chart"].output.payload.get(
                            "html_path"
                        ),
                    }
                ),
            ),
        ]
    )
    return GraphManifest(
        graph_name=graph_name,
        nodes=tuple(nodes),
        default_target="regime_validation_bundle",
    )


def build_trend_state_validation_graph(
    *, instrument: str, timeframe: str
) -> GraphManifest:
    from scripts import validate_trend_state as vt

    graph_name = "validate_trend_state"
    nodes = [
        _source_node(graph_name, "raw_input", "raw_input"),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_minimal_overlay_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": (
                        canonical.copy()
                        if (
                            canonical := vt._load_canonical_live_context(
                                deps["raw_input"].primary_frame().copy(),
                                instrument=instrument,
                                timeframe=timeframe,
                            )
                        )
                        is not None
                        else vt._build_trend_state_context(
                            deps["raw_input"].primary_frame().copy()
                        )
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": (
                        canonical.copy()
                        if (
                            canonical := vt._load_canonical_live_context(
                                deps["raw_input"].primary_frame().copy(),
                                instrument=instrument,
                                timeframe=timeframe,
                            )
                        )
                        is not None
                        else vt._build_context(deps["raw_input"].primary_frame().copy())
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_summary",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("trend_state_context",),
            output_artifacts=("payload",),
            cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload=validate_trend_state(
                    (
                        deps["trend_state_context"].primary_frame()
                        if context.config.get("full", False)
                        else deps["trend_state_context"]
                        .primary_frame()
                        .tail(int(context.config.get("plot_rows", vt.PLOT_ROWS)))
                    ),
                    summary_df=deps["trend_state_context"].primary_frame(),
                    outpath=None,
                    title=f"Trend State Validation — {instrument} {timeframe}",
                    n_windows=5,
                )
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_main_chart",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("trend_state_summary", "trend_state_context"),
            output_artifacts=("html",),
            cache_policy=CachePolicy(materialize=True, artifact_kind="report"),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=lambda context, deps: (
                NodeOutput(payload={"html_path": None})
                if not context.config.get("html", False)
                else NodeOutput(
                    payload={
                        "html_path": str(
                            validate_trend_state(
                                (
                                    deps["trend_state_context"].primary_frame()
                                    if context.config.get("full", False)
                                    else deps["trend_state_context"]
                                    .primary_frame()
                                    .tail(
                                        int(
                                            context.config.get(
                                                "plot_rows", vt.PLOT_ROWS
                                            )
                                        )
                                    )
                                ),
                                summary_df=deps["trend_state_context"].primary_frame(),
                                outpath=Path(
                                    context.config.get("out_dir", "notebooks/structure")
                                )
                                / f"trend_state_validation_{instrument}_{timeframe}.html",
                                title=f"Trend State Validation — {instrument} {timeframe}",
                                n_windows=5,
                            )["html_path"]
                        )
                    },
                    artifacts={
                        "html": Path(
                            context.config.get("out_dir", "notebooks/structure")
                        )
                        / f"trend_state_validation_{instrument}_{timeframe}.html"
                    },
                )
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="trend_state_validation_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("trend_state_summary", "trend_state_main_chart"),
            output_artifacts=("payload",),
            cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": deps["trend_state_summary"].output.payload["summary"],
                    "html_path": deps["trend_state_main_chart"].output.payload.get(
                        "html_path"
                    ),
                }
            ),
        ),
    ]
    return GraphManifest(
        graph_name=graph_name,
        nodes=tuple(nodes),
        default_target="trend_state_validation_bundle",
    )


def build_sr_levels_validation_graph(
    *, instrument: str, timeframe: str
) -> GraphManifest:
    from scripts import validate_sr_levels as vsr

    graph_name = "validate_sr_levels"
    nodes = [
        _source_node(graph_name, "raw_input", "raw_input"),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_enriched_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("raw_input",),
            output_artifacts=("frame",),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": deps["raw_input"]
                    .primary_frame()
                    .pipe(vsr.normalize_candle_schema, require_volume=True)
                    .pipe(vsr.add_atr)
                    .pipe(vsr.add_swings)
                    .pipe(vsr.add_equal_hl)
                    .pipe(vsr.add_prev_day_hl)
                    .pipe(vsr.add_prev_week_hl)
                    .pipe(vsr.add_session_features)
                    .pipe(vsr.add_volume_profile)
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_registry",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("sr_enriched_context",),
            output_artifacts=("payload",),
            cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "registry": vsr.build_sr_level_registry(
                        deps["sr_enriched_context"].primary_frame()
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_projected_context",
            node_kind="compute",
            semantic_class="B",
            inputs=(),
            upstream_nodes=("sr_enriched_context", "sr_registry"),
            output_artifacts=("frame",),
            validation_policy=ValidationPolicy(level="node_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                frames={
                    "frame": vsr.project_sr_context(
                        deps["sr_enriched_context"].primary_frame().copy(),
                        deps["sr_registry"].output.payload["registry"],
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_summary",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("sr_projected_context", "sr_registry"),
            output_artifacts=("payload",),
            cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": summarize_sr_levels(
                        deps["sr_projected_context"].primary_frame(),
                        deps["sr_registry"].output.payload["registry"],
                        live_df=deps["sr_projected_context"].primary_frame(),
                    )
                }
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_main_chart",
            node_kind="report",
            semantic_class="C",
            inputs=(),
            upstream_nodes=(
                "sr_enriched_context",
                "sr_projected_context",
                "sr_registry",
            ),
            output_artifacts=("html",),
            cache_policy=CachePolicy(materialize=True, artifact_kind="report"),
            validation_policy=ValidationPolicy(level="report"),
            mutable_scope=MutableScope(scope="immutable"),
            compute_fn=lambda context, deps: (
                NodeOutput(payload={"html_path": None})
                if not context.config.get("html", False)
                else NodeOutput(
                    payload={
                        "html_path": str(
                            save_figure_html(
                                vsr._build_sr_chart(
                                    deps["sr_enriched_context"].primary_frame(),
                                    deps["sr_projected_context"].primary_frame(),
                                    deps["sr_registry"].output.payload["registry"],
                                    title=f"S/R Levels — {instrument} {timeframe}  |  {vsr.DATE_FROM} → end",
                                ),
                                Path(
                                    context.config.get(
                                        "out_dir", "notebooks/foundation"
                                    )
                                )
                                / f"sr_levels_validation_{instrument}_{timeframe}.html",
                            )
                        )
                    },
                    artifacts={
                        "html": Path(
                            context.config.get("out_dir", "notebooks/foundation")
                        )
                        / f"sr_levels_validation_{instrument}_{timeframe}.html"
                    },
                )
            ),
        ),
        NodeManifest(
            graph_name=graph_name,
            node_name="sr_validation_bundle",
            node_kind="aggregate",
            semantic_class="C",
            inputs=(),
            upstream_nodes=("sr_summary", "sr_main_chart", "sr_projected_context"),
            output_artifacts=("payload",),
            cache_policy=CachePolicy(materialize=False, artifact_kind="ephemeral"),
            validation_policy=ValidationPolicy(level="graph_parity"),
            compute_fn=lambda context, deps: NodeOutput(
                payload={
                    "summary": deps["sr_summary"].output.payload["summary"],
                    "row_count": int(len(deps["sr_projected_context"].primary_frame())),
                    "html_path": deps["sr_main_chart"].output.payload.get("html_path"),
                }
            ),
        ),
    ]
    return GraphManifest(
        graph_name=graph_name, nodes=tuple(nodes), default_target="sr_validation_bundle"
    )


def get_builtin_graph(graph_name: str, **kwargs: Any) -> GraphManifest:
    if graph_name == "live_pipeline":
        return build_live_stage_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            swing_window=int(kwargs.get("swing_window", 6)),
            include_vp=bool(kwargs.get("include_vp", True)),
        )
    if graph_name == "research_pipeline":
        return build_research_stage_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            swing_window=int(kwargs.get("swing_window", 6)),
            include_vp=bool(kwargs.get("include_vp", True)),
            include_avwap=bool(kwargs.get("include_avwap", False)),
        )
    if graph_name == "validate_range_boundaries":
        return build_range_boundaries_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    if graph_name == "validate_regime":
        return build_regime_validation_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    if graph_name == "validate_trend_state":
        return build_trend_state_validation_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    if graph_name == "validate_sr_levels":
        return build_sr_levels_validation_graph(
            instrument=kwargs.get("instrument", "XAU_USD"),
            timeframe=kwargs.get("timeframe", "H4"),
        )
    raise KeyError(f"Unknown builtin graph {graph_name!r}")
