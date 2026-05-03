"""Step 9 — Multi-timeframe (MTF) policy freeze for final sweeps v1.

The final sweeps v1 contract is **same-timeframe only**:

* A sweep on H4 consumes only H4 sources.
* A sweep on H1 consumes only H1 sources.
* No source may be projected from D / H4 / H1 / M15 into another timeframe.

Why this matters
----------------
HTF projection introduces three classes of complexity that we are deferring:

1. *Timestamp inheritance* — the projected source has an HTF origin and confirm
   timestamp; consumers must be told whether the source is "live" on every
   intra-HTF bar of the lower timeframe.
2. *Age inheritance* — bars-since-origin can be measured in HTF bars, LTF bars,
   or wall-clock time. Picking one freezes the strength curve.
3. *Strength inheritance vs. recomputation* — HTF source strength is calibrated
   on HTF candles; if we just copy it down we contaminate LTF strength
   calibration; if we recompute we lose HTF context.

We freeze the cleaner contract first, ship sweeps, then revisit projection.

Validator contract
------------------
Every validation script that touches the unified source framework or final
sweeps must print these three lines (see :func:`mtf_policy_summary`):

    mtf_policy: same_timeframe_only
    htf_projection_enabled: False
    source_timeframe_matches_scan_timeframe: True

If a future caller flips :data:`HTF_LIQUIDITY_PROJECTION_ENABLED` to ``True``
without first updating the unified source framework to label
``source_timeframe`` correctly, :func:`assert_same_timeframe_sources` will
raise loudly so the violation cannot land silently in production.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Frozen policy constants
# ---------------------------------------------------------------------------

#: The single supported MTF policy in v1. Any other value is a hard error.
SWEEP_MTF_POLICY = "same_timeframe_only"

#: Master kill-switch for HTF projection. Must remain ``False`` until the
#: projection contract (timestamp/age/strength inheritance) is frozen and the
#: unified source framework grows the corresponding columns.
HTF_LIQUIDITY_PROJECTION_ENABLED: bool = False

#: Set of timeframe labels we consider valid identity strings.
_KNOWN_TIMEFRAMES: frozenset[str] = frozenset(
    {"M1", "M5", "M15", "M30", "H1", "H4", "D", "W", "MN"}
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def mtf_policy_summary(scan_timeframe: str | None = None) -> dict[str, object]:
    """Return the validator-facing MTF policy summary.

    Parameters
    ----------
    scan_timeframe
        The timeframe the sweeps are being scanned on. When provided, the
        returned dict pins the active scan timeframe for downstream display.

    The returned dict is the canonical format that every sweeps validation
    script prints. Adding new keys is allowed; removing or renaming keys is a
    breaking change to the validator contract.
    """

    return {
        "mtf_policy": SWEEP_MTF_POLICY,
        "htf_projection_enabled": bool(HTF_LIQUIDITY_PROJECTION_ENABLED),
        "scan_timeframe": scan_timeframe,
        "source_timeframe_matches_scan_timeframe": True,
    }


def assert_same_timeframe_sources(
    sources: pd.DataFrame,
    *,
    scan_timeframe: str,
    timeframe_column: str = "source_timeframe",
) -> None:
    """Hard-fail if any source row's timeframe differs from the scan timeframe.

    This is the runtime guard that makes the v1 freeze enforceable. The
    :mod:`unified_sources` builder always stamps every emitted source row with
    ``source_timeframe = scan_timeframe`` (which the builder already knows from
    the pipeline config). If a future contributor adds an HTF projection path
    they must either:

    1. flip :data:`HTF_LIQUIDITY_PROJECTION_ENABLED` to ``True`` and update the
       MTF policy contract, or
    2. tag projected sources with ``source_timeframe`` set to the projected
       timeframe. The next call to this guard will then raise — making the
       leak visible in CI.
    """

    if sources is None or len(sources) == 0:
        return

    if HTF_LIQUIDITY_PROJECTION_ENABLED:
        # Future path; v1 keeps this disabled. We still enforce that
        # ``source_timeframe`` is set to a known value when projection is on.
        unknown = sources.loc[
            ~sources[timeframe_column].astype(str).isin(_KNOWN_TIMEFRAMES),
            timeframe_column,
        ]
        if len(unknown) > 0:
            raise ValueError(
                "Unknown source_timeframe values present while HTF projection "
                f"is enabled: {sorted(set(unknown.astype(str).tolist()))}"
            )
        return

    if timeframe_column not in sources.columns:
        raise ValueError(
            f"assert_same_timeframe_sources: source frame missing "
            f"'{timeframe_column}' column. The unified source builder must "
            f"stamp every row with the scan timeframe."
        )

    distinct = sources[timeframe_column].dropna().astype(str).unique().tolist()
    bad = [tf for tf in distinct if tf and tf != scan_timeframe]
    if bad:
        raise ValueError(
            "MTF policy violation: same_timeframe_only is active but the "
            f"unified source frame contains foreign timeframes {sorted(bad)} "
            f"while the scan timeframe is {scan_timeframe!r}. Flip "
            "HTF_LIQUIDITY_PROJECTION_ENABLED only after the projection "
            "contract is fully implemented."
        )


def assert_known_timeframe(timeframe: str | None) -> str:
    """Normalize and validate a timeframe label, returning the canonical form.

    The unified source builder uses this to stamp every source row, so a typo
    in the pipeline wiring fails fast instead of silently shipping unlabeled
    sources.
    """

    if timeframe is None:
        raise ValueError(
            "assert_known_timeframe: timeframe must be provided. The sweeps "
            "stages cannot be wired without an explicit scan timeframe."
        )
    tf = str(timeframe).upper()
    if tf not in _KNOWN_TIMEFRAMES:
        raise ValueError(
            f"assert_known_timeframe: {timeframe!r} is not a recognised "
            f"timeframe. Known: {sorted(_KNOWN_TIMEFRAMES)}"
        )
    return tf


def known_timeframes() -> Iterable[str]:
    """Return the canonical set of recognised timeframe labels."""

    return tuple(sorted(_KNOWN_TIMEFRAMES))


__all__ = [
    "SWEEP_MTF_POLICY",
    "HTF_LIQUIDITY_PROJECTION_ENABLED",
    "mtf_policy_summary",
    "assert_same_timeframe_sources",
    "assert_known_timeframe",
    "known_timeframes",
]
