"""Shared pytest configuration for TransformerEngine-FL plugin tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="session")
def repository_root() -> Path:
    """Return the TransformerEngine-FL repository root."""
    return Path(__file__).resolve().parents[2]


def _platform_ignore_rules(platform: str) -> dict[str, str]:
    """Read plugin node rules from the owning YAML platform configuration."""
    if not platform or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in platform
    ):
        return {}
    root = Path(
        os.environ.get("GITHUB_WORKSPACE")
        or os.environ.get("TE_PATH")
        or Path(__file__).resolve().parents[2]
    )
    path = root / ".github" / "configs" / f"{platform}.yml"
    if not path.is_file():
        return {}
    try:
        config = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        raise pytest.UsageError(f"Invalid platform config {path}: {error}") from error
    rules = {}
    for target in config.get("ignored_tests", []) or []:
        if (
            not isinstance(target, str)
            or not target.startswith("tests/plugin/")
            or "::" not in target
        ):
            continue
        path_part, node = target.split("::", 1)
        rules[f"{path_part.removeprefix('tests/plugin/')}::{node}"] = target
    return rules


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
        target = rules.get(node)
        if target is None:
            selected.append(item)
        else:
            print(f"[IGNORE][{platform}] {target}")
            ignored.append(item)
    if ignored:
        items[:] = selected
        config.hook.pytest_deselected(items=ignored)
