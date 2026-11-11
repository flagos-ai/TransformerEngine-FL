#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select affected PR platforms; run all platforms for other events.

paths_watch entries are exact repository paths or directory prefixes ending in
"/". Unmapped code, unavailable diffs, and missing watch rules run all platforms.
Only explicitly recognized documentation changes can produce an empty matrix.
"""

from __future__ import annotations

import pathlib
import subprocess

import yaml


_NON_PLATFORM_STEMS = frozenset(("template", "image_sources", "platforms"))


def _changed_files(base_ref: str, cwd: pathlib.Path | None = None) -> list[str]:
    # Disabling rename detection includes both old and new paths for moves.
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            f"origin/{base_ref}...HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd,
    )
    return [path for path in result.stdout.split("\0") if path]


def _matches(path: str, watch: str) -> bool:
    return path.startswith(watch) if watch.endswith("/") else path == watch


def _is_documentation(path: str) -> bool:
    return path.startswith("docs/") or pathlib.PurePosixPath(path).suffix in (
        ".md",
        ".rst",
    )


def _load_watch_rules(configs_dir: pathlib.Path) -> dict[str, list[str]]:
    rules = {}
    for config in sorted(configs_dir.glob("*.yml")):
        if config.stem in _NON_PLATFORM_STEMS:
            continue
        data = yaml.safe_load(config.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Invalid platform configuration: {config}")
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"enabled must be a boolean: {config}")
        if not enabled:
            continue
        watch = data.get("paths_watch", [])
        if not isinstance(watch, list) or any(
            not isinstance(path, str) or not path for path in watch
        ):
            raise ValueError(f"paths_watch must be a list of nonempty paths: {config}")
        rules[config.stem] = watch
    if not rules:
        raise ValueError(f"No platform configurations found: {configs_dir}")
    return rules


def select_platforms(
    rules: dict[str, list[str]],
    changed: list[str],
    event_name: str,
    labels: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    all_platforms = sorted(rules)
    if event_name != "pull_request":
        return all_platforms, [f"{event_name}: full matrix"]
    requested = {
        label.removeprefix("platform:") for label in labels if label.startswith("platform:")
    }
    unknown = requested - set(rules) - {"all"}
    if unknown:
        raise ValueError(f"Unknown platform labels: {', '.join(sorted(unknown))}")
    if "all" in requested:
        return all_platforms, ["platform:all label: full matrix"]
    if requested:
        return sorted(requested), ["Explicit platform labels override changed-file routing"]
    if not changed:
        return all_platforms, ["No changed-file information: full matrix"]

    code_paths = [path for path in changed if not _is_documentation(path)]
    if not code_paths:
        return [], ["Documentation only: no hardware tests required"]
    if any(not watch for watch in rules.values()):
        return all_platforms, ["Missing platform watch rules: full matrix"]

    selected = set()
    reasons = []
    for path in code_paths:
        affected = {
            platform
            for platform, watch in rules.items()
            if any(_matches(path, entry) for entry in watch)
        }
        # A mixed PR must not hide an unmapped/common change behind a matched
        # vendor path. Check each path before taking the union.
        if not affected:
            return all_platforms, [f"Shared or unmapped path {path}: full matrix"]
        selected.update(affected)
        reasons.append(f"{path}: {', '.join(sorted(affected))}")
    return sorted(selected), reasons
