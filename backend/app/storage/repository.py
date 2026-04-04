"""
Repository abstraction for future scan and grading history.

Phase 1: no-op implementations. Replace with SQLite (aiosqlite/sqlalchemy) when needed.
"""

from __future__ import annotations

from typing import Any, Protocol


class ScanRepository(Protocol):
    """Contract for storing analyze results and optional image references."""

    def save_analyze_result(self, payload: dict[str, Any], image_ref: str | None) -> str | None:
        """Persist analysis; return record id or None if disabled."""
        ...


class NullScanRepository:
    """Default repository that does not persist anything."""

    def save_analyze_result(self, payload: dict[str, Any], image_ref: str | None) -> str | None:
        return None


def get_repository() -> ScanRepository:
    """Factory for DI — swap implementation when SQLite is wired up."""
    return NullScanRepository()
