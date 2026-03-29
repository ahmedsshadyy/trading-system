"""
_helpers/event_ids.py

Deterministic sequential event-id allocation helpers.
"""

from __future__ import annotations


class SequentialEventIdAllocator:
    """Allocate stable sequential integer ids starting at 1."""

    def __init__(self, start: int = 1) -> None:
        if start < 1:
            raise ValueError("start must be >= 1")
        self._next = int(start)

    def allocate(self) -> int:
        event_id = self._next
        self._next += 1
        return event_id
