"""Shared pytest configuration for TransformerEngine-FL plugin tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repository_root() -> Path:
    """Return the TransformerEngine-FL repository root."""
    return Path(__file__).resolve().parents[2]
