#!/usr/bin/env python3
"""Shared FlagCICD contract constants and validation (stdlib only)."""

from __future__ import annotations

import hashlib
import fnmatch
import json
from pathlib import PurePosixPath
from typing import Any

# Schema version constants
PLAN_SCHEMA = "flagcicd.plan/v1"
GROUP_REPORT_SCHEMA = "flagcicd.group-report/v1"
PLATFORM_REPORT_SCHEMA = "flagcicd.platform-report/v1"
GATE_SCHEMA = "flagcicd.gate/v1"
IDENTITY_SCHEMA = "flagcicd.artifact-identity/v1"

KINDS = frozenset(("unit", "integration", "preflight", "compat", "benchmark", "release"))

STATUSES = frozenset(
    (
        "passed",
        "test_failed",
        "build_failed",
        "infra_error",
        "unsupported",
        "skipped",
        "cancelled",
        "report_error",
    )
)

LEVELS = ("L0", "L1", "L2", "L3", "L4")
LANES = ("preflight", "parallel", "serial")


def split_test_target(target: str) -> tuple[str, str]:
    """Validate a repository path/glob with an optional pytest node selector."""
    if not isinstance(target, str) or not target.strip() or "\\" in target:
        raise ValueError(f"Invalid test target: {target!r}")
    path, separator, node = target.partition("::")
    parts = PurePosixPath(path).parts
    if not parts or PurePosixPath(path).is_absolute() or ".." in parts:
        raise ValueError(f"Invalid repository-relative test target: {target!r}")
    if separator and (not node or path.endswith("/") or any(not part for part in node.split("::"))):
        raise ValueError(f"Invalid pytest node selector: {target!r}")
    return path, node


def test_target_matches(nodeid: str, target: str) -> bool:
    """Match a canonical pytest node; bare functions/classes include their parameters."""
    target_path, target_node = split_test_target(target)
    path, separator, node = nodeid.partition("::")
    path_matches = (
        path.startswith(target_path)
        if target_path.endswith("/")
        else fnmatch.fnmatchcase(path, target_path)
    )
    if not path_matches or not target_node:
        return path_matches
    if not separator:
        return False
    if "[" in target_node:
        return node == target_node  # Parameter IDs are literal, not glob character classes.
    base_node = node.split("[", 1)[0]
    parts = base_node.split("::")
    return any(
        fnmatch.fnmatchcase("::".join(parts[:length]), target_node)
        for length in range(1, len(parts) + 1)
    )


def canonical_json(obj: Any) -> str:
    """Serialize to deterministic single-line JSON for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_plan_hash(plan: dict[str, Any]) -> str:
    """Compute SHA256 hash of plan excluding plan_hash itself."""
    plan_copy = dict(plan)
    plan_copy.pop("plan_hash", None)
    return hashlib.sha256(canonical_json(plan_copy).encode("utf-8")).hexdigest()


def validate_plan(plan: dict[str, Any]) -> list[str]:
    """Validate plan structure and return list of errors."""
    errors = []
    if plan.get("schema_version") != PLAN_SCHEMA:
        errors.append(f"schema_version must be {PLAN_SCHEMA}")
    if not isinstance(plan.get("platforms"), list):
        errors.append("platforms must be a list")
    if not isinstance(plan.get("test_groups"), list):
        errors.append("test_groups must be a list")
    if not isinstance(plan.get("execution_groups"), list):
        errors.append("execution_groups must be a list")
    if isinstance(plan.get("test_groups"), list) and isinstance(plan.get("execution_groups"), list):
        try:
            executions = {}
            for job in plan["execution_groups"]:
                key = (job["platform"], job["device"], job["group"])
                if key in executions:
                    errors.append(f"Duplicate execution group: {key}")
                names = job["test_groups"]
                if (
                    not isinstance(names, list)
                    or not names
                    or any(not isinstance(name, str) or not name for name in names)
                    or len(names) != len(set(names))
                ):
                    errors.append(f"Invalid execution test_groups: {key}")
                executions[key] = set(names)
            actual = {}
            logical_keys = set()
            for group in plan["test_groups"]:
                key = (group["platform"], group["device"], group["group"])
                if key in logical_keys:
                    errors.append(f"Duplicate logical test group: {key}")
                logical_keys.add(key)
                execution = (group["platform"], group["device"], group["execution_group"])
                actual.setdefault(execution, set()).add(group["group"])
                job = next(
                    (
                        job
                        for job in plan["execution_groups"]
                        if (job["platform"], job["device"], job["group"]) == execution
                    ),
                    None,
                )
                if job and group.get("evidence") != job.get("evidence"):
                    errors.append(f"Test-group evidence does not match execution: {key}")
                if (
                    job
                    and group.get("test_targets")
                    and (job.get("exec") or {}).get("group_test_targets", {}).get(group["group"])
                    != group["test_targets"]
                ):
                    errors.append(f"Test targets do not match execution: {key}")
            if executions != actual:
                errors.append("Logical test groups do not match execution groups")
        except (KeyError, TypeError):
            errors.append("Malformed test-group execution mapping")
    for collection in ("execution_groups", "test_groups"):
        for group in plan.get(collection, []) if isinstance(plan.get(collection), list) else []:
            if not isinstance(group, dict) or group.get("kind") not in KINDS:
                errors.append(f"Invalid or missing kind in {collection}")
    if plan.get("test_level") not in LEVELS:
        errors.append(f"test_level must be one of {LEVELS}")
    expected_hash = compute_plan_hash(plan)
    if plan.get("plan_hash") != expected_hash:
        errors.append(f"plan_hash mismatch: expected {expected_hash[:12]}...")
    return errors


def load_plan(plan_json: str) -> dict[str, Any]:
    """Load and validate plan from JSON string."""
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid plan JSON: {exc}") from exc
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")
    errors = validate_plan(plan)
    if errors:
        raise ValueError(f"Invalid plan: {'; '.join(errors)}")
    return plan
