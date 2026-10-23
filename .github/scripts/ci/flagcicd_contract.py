#!/usr/bin/env python3
"""Shared FlagCICD contract constants and validation (stdlib only)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Schema version constants
PLAN_SCHEMA = "flagcicd.plan/v1"
GROUP_REPORT_SCHEMA = "flagcicd.group-report/v1"
PLATFORM_REPORT_SCHEMA = "flagcicd.platform-report/v1"
GATE_SCHEMA = "flagcicd.gate/v1"
IDENTITY_SCHEMA = "flagcicd.artifact-identity/v1"
EVIDENCE_SCHEMA = "flagcicd.runtime-evidence/v1"

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
RISKS = ("R1", "R2", "R3", "R4")
LANES = ("preflight", "parallel", "serial")

# FastCI selection modes. "static" preserves the existing path-match behaviour
# and is the default when the fastci block is absent or disabled.
SELECTION_MODES = ("static", "hybrid")
EVIDENCE_MODES = ("off", "shadow", "enforced")
FALLBACK_MODES = ("static", "fail")

# Context dimensions used by equivalence pruning (P3). Two executions are
# equivalent only when every dimension listed here matches.
CONTEXT_DIMENSIONS = (
    "platform",
    "device",
    "image",
    "torch",
    "cuda",
    "vendor_sdk",
    "precision",
    "world_size",
    "model_family",
    "sequence_length",
    "test_kind",
    "communication_mode",
)

# Priority waves for risk-ordered scheduling (P4). The planner assigns each
# group to a wave based on its risk and resource class; the workflow consumes
# waves via needs relationships, not JSON ordering.
PRIORITY_WAVES = ("priority-wave-0", "priority-wave-1", "serial-wave")


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
    if plan.get("test_level") not in LEVELS:
        errors.append(f"test_level must be one of {LEVELS}")
    if plan.get("risk_level") not in RISKS:
        errors.append(f"risk_level must be one of {RISKS}")
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
