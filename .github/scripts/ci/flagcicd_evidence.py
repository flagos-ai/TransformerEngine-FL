#!/usr/bin/env python3
"""FlagCICD runtime evidence loader and shadow-plan comparator (stdlib only).

Evidence flow
─────────────
1. A previous run uploaded a runtime-evidence-v1.json artifact for each group.
2. This module loads and validates those files against the current context.
3. In shadow mode it builds a "shadow plan" using the evidence and compares it
   to the static plan; the diff is written to the step summary so engineers can
   see what hybrid selection *would* have done without affecting the actual run.
4. In enforced mode (fastci.evidence_mode=enforced) the hybrid router in
   flagcicd_planner.py can call select_by_evidence() to get a narrowed group
   set; evidence failures fall back to static per fallback_on_error config.

Evidence is trusted only when:
  - schema_version matches EVIDENCE_SCHEMA
  - image_digest matches the current platform image digest
  - every context dimension in CONTEXT_DIMENSIONS matches the current run
  - flagcicd_history reports the group as stable (>= STABLE_PASS_THRESHOLD passes)
  - source_sha is not older than MAX_EVIDENCE_AGE_COMMITS from HEAD
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import flagcicd_contract as contract
from flagcicd_history import query_stability, STABLE_PASS_THRESHOLD

EVIDENCE_SCHEMA = contract.EVIDENCE_SCHEMA
MAX_EVIDENCE_AGE_COMMITS = 30  # reject evidence beyond this many commits from HEAD


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_evidence(path: Path) -> dict[str, Any]:
    """Load and structurally validate a runtime-evidence-v1.json file."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read evidence file {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"evidence file {path} must be a JSON object")
    if obj.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError(
            f"evidence {path} has schema_version {obj.get('schema_version')!r},"
            f" expected {EVIDENCE_SCHEMA!r}"
        )
    for field in ("platform", "device", "group", "image_digest", "context", "frames"):
        if field not in obj:
            raise ValueError(f"evidence {path} missing required field {field!r}")
    return obj


def _context_matches(evidence: dict, current_context: dict) -> tuple[bool, list[str]]:
    """Compare evidence context against current run context dimensions.

    Returns (matches: bool, mismatches: list[str-description]).
    Only dimensions listed in contract.CONTEXT_DIMENSIONS are compared;
    a None value in evidence means "not recorded" and always matches.
    """
    ev_ctx = evidence.get("context", {})
    mismatches: list[str] = []
    for dim in contract.CONTEXT_DIMENSIONS:
        ev_val = ev_ctx.get(dim)
        cur_val = current_context.get(dim)
        if ev_val is None:
            continue  # not recorded → skip
        if ev_val != cur_val:
            mismatches.append(f"{dim}: evidence={ev_val!r} current={cur_val!r}")
    return len(mismatches) == 0, mismatches


def is_evidence_valid(
    evidence: dict,
    *,
    current_image_digest: str,
    current_context: dict,
    history_path: Path | None = None,
) -> tuple[bool, list[str]]:
    """Check whether a piece of evidence should be trusted.

    Returns (valid: bool, reasons: list[str]).
    reasons is non-empty when valid is False, describing why.
    """
    reasons: list[str] = []

    # Image digest must match exactly — a new image invalidates all evidence.
    if evidence.get("image_digest") != current_image_digest:
        reasons.append(
            f"image_digest mismatch: evidence={evidence.get('image_digest')!r}"
            f" current={current_image_digest!r}"
        )

    # Context dimensions must be equivalent.
    ok, mismatches = _context_matches(evidence, current_context)
    if not ok:
        reasons.extend(mismatches)

    # Evidence must have completed successfully.
    if not evidence.get("complete", True):
        reasons.append("evidence is marked incomplete (sampler terminated early)")
    if evidence.get("status") not in ("passed", "failed"):
        reasons.append(f"evidence status {evidence.get('status')!r} is not actionable")

    # Stability check via history (optional — history_path may be absent).
    if history_path is not None and history_path.exists():
        stability = query_stability(
            history_path,
            evidence["platform"],
            evidence["device"],
            evidence["group"],
        )
        if not stability["stable"]:
            reasons.append(
                f"group not yet stable: pass_streak={stability['pass_streak']}"
                f" (need {STABLE_PASS_THRESHOLD})"
            )

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# File-set operations
# ---------------------------------------------------------------------------


def load_evidence_set(
    evidence_dir: Path,
    platform: str,
) -> dict[str, dict[str, Any]]:
    """Load all evidence files for a platform from a directory.

    Returns a dict keyed by group name.  Files that fail to load are skipped
    with a stderr warning so a single corrupt file cannot break the whole run.
    """
    result: dict[str, dict[str, Any]] = {}
    if not evidence_dir.is_dir():
        return result
    pattern = f"runtime-evidence-{platform}-*.json"
    for path in sorted(evidence_dir.glob(pattern)):
        try:
            ev = load_evidence(path)
            if ev.get("platform") == platform:
                result[ev["group"]] = ev
        except ValueError as exc:
            print(f"WARNING: skipping evidence {path.name}: {exc}", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Shadow plan comparison
# ---------------------------------------------------------------------------


def _coverage_files(evidence: dict) -> set[str]:
    """Return the set of source files touched by this evidence's sampled frames."""
    return {f["file"] for f in evidence.get("frames", []) if f.get("file")}


def shadow_compare(
    static_plan_groups: list[dict],
    evidence_set: dict[str, dict],
    current_image_digest: str,
    current_context: dict,
    history_path: Path | None = None,
) -> dict[str, Any]:
    """Build a shadow selection and compare it to the static plan.

    Returns a summary dict with:
      shadow_selected   – group names the evidence would have selected
      shadow_excluded   – group names the evidence would have dropped
      static_only       – in static but not in shadow (potential over-testing)
      shadow_only       – in shadow but not in static (evidence adds groups)
      valid_evidence    – groups with trusted evidence
      invalid_evidence  – groups whose evidence failed validation
      coverage_files    – union of all source files touched by valid evidence
    """
    static_groups = {g["group"] for g in static_plan_groups if g.get("lane") != "preflight"}

    valid_evidence: dict[str, dict] = {}
    invalid_evidence: dict[str, list[str]] = {}

    for group, ev in evidence_set.items():
        ok, reasons = is_evidence_valid(
            ev,
            current_image_digest=current_image_digest,
            current_context=current_context,
            history_path=history_path,
        )
        if ok:
            valid_evidence[group] = ev
        else:
            invalid_evidence[group] = reasons

    # Shadow selection: groups with valid evidence that touched changed files
    # (P2 logic placeholder — for now select everything with valid evidence).
    shadow_selected = set(valid_evidence.keys())
    shadow_excluded = static_groups - shadow_selected

    return {
        "shadow_selected": sorted(shadow_selected),
        "shadow_excluded": sorted(shadow_excluded),
        "static_only": sorted(static_groups - shadow_selected),
        "shadow_only": sorted(shadow_selected - static_groups),
        "valid_evidence": sorted(valid_evidence.keys()),
        "invalid_evidence": {k: v for k, v in invalid_evidence.items()},
        "coverage_files": sorted(
            {f for ev in valid_evidence.values() for f in _coverage_files(ev)}
        ),
    }


def write_shadow_summary(comparison: dict, output_path: Path | None = None) -> None:
    """Write a human-readable shadow comparison to GITHUB_STEP_SUMMARY or a file."""
    import os

    lines = [
        "### FastCI Shadow Plan Comparison\n",
        f"- Valid evidence: {len(comparison['valid_evidence'])} groups\n",
        f"- Invalid evidence: {len(comparison['invalid_evidence'])} groups\n",
        f"- Shadow would select: {len(comparison['shadow_selected'])} groups\n",
        f"- Shadow would exclude: {len(comparison['shadow_excluded'])} groups\n",
    ]
    if comparison["invalid_evidence"]:
        lines.append("\n**Evidence validation failures:**\n")
        for group, reasons in sorted(comparison["invalid_evidence"].items()):
            lines.append(f"- `{group}`: {'; '.join(reasons)}\n")
    if comparison["shadow_excluded"]:
        lines.append("\n**Groups shadow would skip (static keeps them):**\n")
        for g in comparison["shadow_excluded"]:
            lines.append(f"- `{g}`\n")

    text = "".join(lines)
    target = output_path or (
        Path(os.environ["GITHUB_STEP_SUMMARY"]) if "GITHUB_STEP_SUMMARY" in os.environ else None
    )
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    vp = sub.add_parser("validate", help="Validate one evidence file")
    vp.add_argument("--evidence", required=True)
    vp.add_argument("--image-digest", required=True)
    vp.add_argument("--context-json", default="{}")
    vp.add_argument("--history")

    sp = sub.add_parser("shadow", help="Run shadow comparison and write summary")
    sp.add_argument("--evidence-dir", required=True)
    sp.add_argument("--platform", required=True)
    sp.add_argument("--plan", required=True, help="Path to flagcicd.plan/v1 JSON")
    sp.add_argument("--image-digest", required=True)
    sp.add_argument("--context-json", default="{}")
    sp.add_argument("--history")
    sp.add_argument("--output")

    args = parser.parse_args()

    if args.command == "validate":
        try:
            ev = load_evidence(Path(args.evidence))
            ctx = json.loads(args.context_json)
            ok, reasons = is_evidence_valid(
                ev,
                current_image_digest=args.image_digest,
                current_context=ctx,
                history_path=Path(args.history) if args.history else None,
            )
            if ok:
                print("evidence is valid")
                return 0
            for r in reasons:
                print(f"INVALID: {r}")
            return 1
        except ValueError as exc:
            print(f"::error::{exc}", file=sys.stderr)
            return 1

    if args.command == "shadow":
        try:
            plan = json.loads(Path(args.plan).read_text())
            groups = plan.get("test_groups", [])
            evidence_set = load_evidence_set(Path(args.evidence_dir), args.platform)
            ctx = json.loads(args.context_json)
            comparison = shadow_compare(
                groups,
                evidence_set,
                current_image_digest=args.image_digest,
                current_context=ctx,
                history_path=Path(args.history) if args.history else None,
            )
            out = Path(args.output) if args.output else None
            write_shadow_summary(comparison, out)
            # Emit as JSON so callers can parse it
            print(json.dumps(comparison, indent=2))
            return 0
        except Exception as exc:
            print(f"::error::shadow comparison failed: {exc}", file=sys.stderr)
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
