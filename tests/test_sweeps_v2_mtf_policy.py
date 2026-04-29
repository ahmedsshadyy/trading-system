"""Tests for Step 9 — same-timeframe MTF policy freeze."""

from __future__ import annotations

import pandas as pd
import pytest

from src.indicators.sweeps_v2.mtf_policy import (
    HTF_LIQUIDITY_PROJECTION_ENABLED,
    SWEEP_MTF_POLICY,
    assert_known_timeframe,
    assert_same_timeframe_sources,
    known_timeframes,
    mtf_policy_summary,
)


def test_v1_freeze_constants() -> None:
    """The v1 freeze must keep HTF projection disabled and the policy
    label pinned. Flipping either is a doctrine change, not a typo fix.
    """

    assert SWEEP_MTF_POLICY == "same_timeframe_only"
    assert HTF_LIQUIDITY_PROJECTION_ENABLED is False


def test_mtf_policy_summary_has_required_keys() -> None:
    summary = mtf_policy_summary("H4")
    for key in (
        "mtf_policy",
        "htf_projection_enabled",
        "scan_timeframe",
        "source_timeframe_matches_scan_timeframe",
    ):
        assert key in summary
    assert summary["mtf_policy"] == "same_timeframe_only"
    assert summary["htf_projection_enabled"] is False
    assert summary["scan_timeframe"] == "H4"


def test_assert_known_timeframe_normalizes_case() -> None:
    assert assert_known_timeframe("h4") == "H4"
    assert assert_known_timeframe("H1") == "H1"


def test_assert_known_timeframe_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        assert_known_timeframe("h2")


def test_assert_known_timeframe_rejects_none() -> None:
    with pytest.raises(ValueError):
        assert_known_timeframe(None)


def test_known_timeframes_includes_canonical_set() -> None:
    tfs = set(known_timeframes())
    for required in ("M1", "M15", "H1", "H4", "D", "W"):
        assert required in tfs


def test_assert_same_timeframe_sources_passes_when_all_match() -> None:
    sources = pd.DataFrame(
        {
            "source_timeframe": ["H4", "H4", "H4"],
            "level": [1.0, 2.0, 3.0],
        }
    )
    # Should not raise
    assert_same_timeframe_sources(sources, scan_timeframe="H4")


def test_assert_same_timeframe_sources_fails_on_foreign_timeframe() -> None:
    sources = pd.DataFrame(
        {
            "source_timeframe": ["H4", "D", "H4"],
            "level": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="MTF policy violation"):
        assert_same_timeframe_sources(sources, scan_timeframe="H4")


def test_assert_same_timeframe_sources_skips_empty() -> None:
    """Empty source frames should not trigger the guard — there is nothing
    to validate. The guard is meant to catch projection leaks, not require
    that every bar has sources."""

    assert_same_timeframe_sources(pd.DataFrame(), scan_timeframe="H4")


def test_assert_same_timeframe_sources_requires_column() -> None:
    sources = pd.DataFrame({"level": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        assert_same_timeframe_sources(sources, scan_timeframe="H4")
