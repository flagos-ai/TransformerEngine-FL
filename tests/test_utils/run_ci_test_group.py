#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Execute a CI unit-test group described by the platform configuration."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(os.environ.get("GITHUB_WORKSPACE", Path(__file__).resolve().parents[2]))


def _expand(value: str) -> str:
    return value.replace("{workspace}", str(REPO_ROOT))


def _load_group() -> dict[str, Any]:
    raw_group = os.environ.get("TE_TEST_GROUP_JSON")
    if not raw_group:
        raise SystemExit("TE_TEST_GROUP_JSON is required")
    group = json.loads(raw_group)
    if not isinstance(group, dict) or not group.get("name"):
        raise SystemExit("The test group must be an object with a name")
    return group


def _run_script(group: dict[str, Any]) -> int:
    script = group.get("path")
    if not script:
        raise SystemExit(f"Script test group {group['name']} has no path")
    script_path = REPO_ROOT / _expand(str(script))
    if not script_path.is_file():
        raise SystemExit(f"Test script does not exist: {script_path}")
    command = ["bash", str(script_path)]
    command.extend(_expand(str(arg)) for arg in group.get("args", []))
    script_env = os.environ.copy()
    script_env.update(
        {str(key): _expand(str(value)) for key, value in group.get("env", {}).items()}
    )
    print(f"[RUN] {shlex.join(command)}", flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, env=script_env, check=False).returncode


def _pytest_command(use_platform_runner: bool) -> list[str]:
    platform_command = os.environ.get("TE_TEST_PYTEST_COMMAND")
    if use_platform_runner and platform_command:
        return [_expand(part) for part in shlex.split(platform_command)]
    return [sys.executable, "-m", "pytest"]


def _run_pytest(group: dict[str, Any]) -> int:
    steps = group.get("steps")
    if not isinstance(steps, list) or not steps:
        raise SystemExit(f"Pytest group {group['name']} has no steps")

    log_dir = REPO_ROOT / _expand(str(group.get("log_dir", "logs")))
    log_dir.mkdir(parents=True, exist_ok=True)
    group_args = [_expand(str(arg)) for arg in group.get("pytest_args", [])]
    group_env = {str(key): _expand(str(value)) for key, value in group.get("env", {}).items()}
    failed = False

    for step in steps:
        if not isinstance(step, dict) or not step.get("name"):
            raise SystemExit(f"Invalid pytest step in group {group['name']}")
        targets = [_expand(str(target)) for target in step.get("targets", [])]
        if not targets:
            raise SystemExit(f"Pytest step {step['name']} has no targets")

        missing_modules = []
        for module_name in step.get("requires_modules", []):
            try:
                importlib.import_module(str(module_name))
            except ModuleNotFoundError:
                missing_modules.append(str(module_name))
        if missing_modules:
            print(
                f"[FAIL] {step['name']}: missing modules: {', '.join(missing_modules)}",
                flush=True,
            )
            failed = True
            continue

        command = _pytest_command(bool(step.get("use_platform_runner", True)))
        command.extend(group_args)
        command.extend(_expand(str(arg)) for arg in step.get("args", []))
        if step.get("junit"):
            command.append(f"--junitxml={log_dir / str(step['junit'])}")
        command.extend(targets)

        step_env = os.environ.copy()
        step_env.update(group_env)
        step_env.update(
            {str(key): _expand(str(value)) for key, value in step.get("env", {}).items()}
        )
        print(f"[RUN] {step['name']}: {shlex.join(command)}", flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT, env=step_env, check=False)
        failed = failed or result.returncode != 0

    return 1 if failed else 0


def main() -> int:
    group = _load_group()
    runner = group.get("runner", "script")
    if runner == "script":
        return _run_script(group)
    if runner == "pytest":
        return _run_pytest(group)
    raise SystemExit(f"Unsupported test runner: {runner}")


if __name__ == "__main__":
    raise SystemExit(main())
