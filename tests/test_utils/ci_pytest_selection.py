# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Apply the selected test groups' file scope inside existing platform QA scripts.

The plugin is enabled by run_ci_test_group.py only for groups with test_targets.
It narrows pytest's original arguments before test modules are imported, leaving
platform-specific node ids, -k expressions, deselection and other options intact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from flagcicd_contract import split_test_target, test_target_matches


class _Selection:
    def __init__(self, raw: str, *, ignored: bool = False):
        setting = "TE_CI_IGNORE_TEST_TARGETS_JSON" if ignored else "TE_CI_TEST_TARGETS_JSON"
        try:
            targets = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise pytest.UsageError(f"{setting} must contain valid JSON") from error
        if (
            not isinstance(targets, list)
            or (not targets and not ignored)
            or any(not isinstance(target, str) or not target.strip() for target in targets)
        ):
            raise pytest.UsageError("test_targets must be a non-empty list of relative paths")

        self.root = Path(
            os.environ.get("GITHUB_WORKSPACE") or os.environ.get("TE_PATH") or os.getcwd()
        ).resolve()
        self.paths: dict[Path, bool] = {}
        self.targets = targets
        for target in targets:
            try:
                target_path, target_node = split_test_target(target)
            except ValueError as error:
                raise pytest.UsageError(str(error)) from error
            # A node-level ignore must not hide the entire file before import.
            if ignored and target_node:
                continue
            for path in self.root.glob(target_path):
                resolved = path.resolve()
                if not resolved.is_relative_to(self.root):
                    raise pytest.UsageError(f"Test target escapes the repository: {target!r}")
                self.paths[resolved] = path.is_dir()

    def selected(self, path: Path, *, ancestors: bool = False) -> bool:
        resolved = path.resolve()
        for target, directory in self.paths.items():
            if (
                resolved == target
                or directory
                and resolved.is_relative_to(target)
                or ancestors
                and target.is_relative_to(resolved)
            ):
                return True
        return False


def pytest_configure(config):
    raw = os.environ.get("TE_CI_TEST_TARGETS_JSON")
    config._flagcicd_selection = None if raw is None else _Selection(raw)
    config._flagcicd_ignored = _Selection(
        os.environ.get("TE_CI_IGNORE_TEST_TARGETS_JSON", "[]"), ignored=True
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection(session):
    selection = session.config._flagcicd_selection
    if selection is None:
        return
    selected = []
    for argument in session.config.args:
        path = Path(argument.split("::", 1)[0])
        if not path.is_absolute():
            path = Path(session.config.invocation_params.dir) / path
        if session.config._flagcicd_ignored.selected(path):
            print(f"[IGNORE] {argument}")
        elif selection.selected(path, ancestors=path.is_dir()):
            selected.append(argument)
    if not selected:
        # Platform launchers invoke pytest once per test file. An unrelated
        # invocation is not a failure and must not manufacture passing evidence.
        pytest.exit("No test files in this invocation match the FlagCICD plan", returncode=0)
    session.config.args = selected


def pytest_ignore_collect(collection_path: Path, config):
    selection = config._flagcicd_selection
    if config._flagcicd_ignored.selected(collection_path):
        print(f"[IGNORE] {collection_path}")
        return True
    if selection is not None and not selection.selected(
        collection_path, ancestors=collection_path.is_dir()
    ):
        return True
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    selection = config._flagcicd_selection
    if selection is None:
        return
    config._flagcicd_had_items = bool(items)
    selected, deselected = [], []
    for item in items:
        path = Path(item.path).resolve()
        if not path.is_relative_to(selection.root):
            raise pytest.UsageError(f"Collected test is outside the repository: {path}")
        relative_path = path.relative_to(selection.root).as_posix()
        nodeid = relative_path + "::" + item.nodeid.split("::", 1)[1]
        if any(test_target_matches(nodeid, target) for target in config._flagcicd_ignored.targets):
            print(f"[IGNORE] {nodeid}")
            deselected.append(item)
            continue
        if not any(test_target_matches(nodeid, target) for target in selection.targets):
            deselected.append(item)
            continue
        selected.append(item)
        item.user_properties.extend(
            [
                ("flagcicd_test_path", relative_path),
                ("flagcicd_test_nodeid", nodeid),
            ]
        )
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def pytest_collection_finish(session):
    if (
        session.config._flagcicd_selection is not None
        and getattr(session.config, "_flagcicd_had_items", False)
        and not session.items
        and not session.testsfailed
    ):
        # An adapter may invoke one whitelisted node at a time. An invocation
        # outside this plan's node scope produces no passing evidence.
        pytest.exit("No selected FlagCICD test nodes remain", returncode=0)
