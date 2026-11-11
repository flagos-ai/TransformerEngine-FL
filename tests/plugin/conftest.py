"""Shared pytest configuration for TransformerEngine-FL plugin tests."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repository_root() -> Path:
    """Return the TransformerEngine-FL repository root."""
    return Path(__file__).resolve().parents[2]


def _platform_ignore_rules(platform: str) -> dict[str, str]:
    """Read literal node rules from the owning platform config, without executing it."""
    if not re.fullmatch(r"[a-z0-9_]+", platform):
        return {}
    path = Path(__file__).resolve().parent / "backend" / platform / "config.sh"
    if not path.is_file():
        return {}
    source = path.read_text()
    declaration = "readonly PLUGIN_IGNORE_TEST_NODES=("
    if declaration not in source:
        return {}
    block = source.split(declaration, 1)[1]
    lines = block.splitlines()
    rules = {}
    for line in lines:
        if line.strip() == ")":
            return rules
        try:
            entries = shlex.split(line, comments=True)
        except ValueError as error:
            raise pytest.UsageError(f"Invalid plugin ignore rule in {path}: {error}") from error
        if not entries:
            continue
        if len(entries) != 1:
            raise pytest.UsageError(f"Expected one quoted node|reason per line in {path}")
        node, separator, reason = entries[0].partition("|")
        if not separator or "::" not in node or not reason.strip() or node in rules:
            raise pytest.UsageError(f"Invalid or duplicate plugin ignore rule in {path}: {node}")
        rules[node] = reason
    raise pytest.UsageError(f"Unterminated PLUGIN_IGNORE_TEST_NODES array in {path}")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Filter inner subprocess cases too, without hiding the entire driver."""
    platform = os.environ.get("PLATFORM", "")
    rules = _platform_ignore_rules(platform)
    root = Path(__file__).resolve().parent
    selected, ignored = [], []
    for item in items:
        path = Path(item.path).resolve()
        if not path.is_relative_to(root):
            selected.append(item)
            continue
        node = path.relative_to(root).as_posix() + "::" + item.nodeid.split("::", 1)[1]
        reason = rules.get(node)
        if reason is None:
            selected.append(item)
        else:
            print(f"[IGNORE][{platform}] {node}: {reason}")
            ignored.append(item)
    if ignored:
        items[:] = selected
        config.hook.pytest_deselected(items=ignored)
