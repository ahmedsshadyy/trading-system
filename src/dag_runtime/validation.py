from __future__ import annotations

import pandas as pd

from src.dag_runtime.executor import GraphRunResult


def assert_frame_parity(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    normalize_missing_non_numeric: bool = False,
    atol: float | None = None,
) -> None:
    lhs = left.copy()
    rhs = right.copy()
    if normalize_missing_non_numeric:
        for frame in (lhs, rhs):
            for column in frame.columns:
                if pd.api.types.is_numeric_dtype(
                    frame[column]
                ) or pd.api.types.is_datetime64_any_dtype(frame[column]):
                    continue
                frame[column] = (
                    frame[column].astype("object").where(frame[column].notna(), None)
                )
    kwargs = {"check_dtype": False}
    if atol is not None:
        kwargs["atol"] = atol
        kwargs["rtol"] = 0
    pd.testing.assert_frame_equal(
        lhs.reset_index(drop=True),
        rhs.reset_index(drop=True),
        **kwargs,
    )


def validate_graph_parity(
    baseline: GraphRunResult,
    candidate: GraphRunResult,
    *,
    atol: float | None = None,
) -> None:
    left = baseline.primary_frame()
    right = candidate.primary_frame()
    if left is None or right is None:
        raise AssertionError(
            "Graph parity requires both targets to produce a primary frame"
        )
    assert_frame_parity(left, right, normalize_missing_non_numeric=True, atol=atol)
