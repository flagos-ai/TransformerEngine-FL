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
import time
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


def _selection_env(env: dict[str, str], group: dict[str, Any]) -> None:
    """Carry the plan's file selection into every child pytest invocation."""
    if "test_targets" not in group:
        return
    targets = group["test_targets"]
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) or not target.strip() for target in targets)
    ):
        raise SystemExit("test_targets must be a non-empty list of repository-relative paths")
    env["TE_CI_TEST_TARGETS_JSON"] = json.dumps(targets)
    plugin_dir = str(Path(__file__).resolve().parent)
    contract_dir = str(Path(__file__).resolve().parents[2] / ".github/scripts/ci")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (plugin_dir, contract_dir, env.get("PYTHONPATH", "")) if part
    )
    plugins = [name.strip() for name in env.get("PYTEST_PLUGINS", "").split(",") if name.strip()]
    if "ci_pytest_selection" not in plugins:
        plugins.append("ci_pytest_selection")
    env["PYTEST_PLUGINS"] = ",".join(plugins)


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
    _selection_env(script_env, group)
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
        _selection_env(step_env, group)
        print(f"[RUN] {step['name']}: {shlex.join(command)}", flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT, env=step_env, check=False)
        if result.returncode < 0 or result.returncode >= 128:
            return result.returncode
        failed = failed or result.returncode != 0

    return 1 if failed else 0


def main() -> int:
    group = _load_group()
    runner = group.get("runner", "script")
    runners = {"script": _run_script, "pytest": _run_pytest}
    if runner not in runners:
        raise SystemExit(f"Unsupported test runner: {runner}")
    raw_retries = group.get("max_retries", os.environ.get("TE_TEST_MAX_RETRIES", "0"))
    if isinstance(raw_retries, bool) or str(raw_retries) not in ("0", "1", "2"):
        raise SystemExit("max_retries / TE_TEST_MAX_RETRIES must be 0, 1, or 2")
    retries = int(raw_retries)
    for attempt in range(retries + 1):
        print(f"[ATTEMPT] {group['name']}: {attempt + 1}/{retries + 1}", flush=True)
        result = runners[runner](group)
        # Do not retry interrupted/killed commands, including shell signal exits.
        if result == 0 or result < 0 or result >= 128 or attempt == retries:
            return result
        print(
            f"[RETRY] Test group failed with exit code {result}; retrying in 5s",
            flush=True,
        )
        time.sleep(5)
    raise AssertionError("Unreachable retry state")


if __name__ == "__main__":
    raise SystemExit(main())
