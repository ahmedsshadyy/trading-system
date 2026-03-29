"""
pipelines/ — Build orchestration controllers.

Separate pipelines for research (full stack + diagnostics),
live (causal only, no labels), and future plotting builds.
"""

from src.indicators.pipelines.build_research import (
    build_research_indicators,
    build_all_indicators,
    materialize_research_features,
    run_research_pipeline,
)
from src.indicators.pipelines.build_live import (
    build_live_indicators,
    materialize_live_features,
    run_live_pipeline,
)

__all__ = [
    "build_research_indicators",
    "build_all_indicators",
    "build_live_indicators",
    "materialize_live_features",
    "materialize_research_features",
    "run_live_pipeline",
    "run_research_pipeline",
]
