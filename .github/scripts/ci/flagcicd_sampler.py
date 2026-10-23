#!/usr/bin/env python3
"""FlagCICD runtime evidence collector (stdlib + coverage.py optional).

Converts a completed test run's coverage data and/or JUnit XML into a
runtime-evidence-v1.json file that the evidence module can validate and
the shadow comparator can consume.

Two evidence sources are supported (in priority order):
  1. coverage.py .coverage file  → file-level frames extracted via json export
  2. JUnit XML directory         → test class paths used as coarse frame set

This runs *after* the test group finishes, so it never adds latency to the
test execution itself.  The resulting file is uploaded as a CI artifact and
downloaded by the next run before the planner decides which groups to run.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "flagcicd.runtime-evidence/v1"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


def _frames_from_coverage(coverage_file: Path) -> list[dict[str, Any]]:
    """Extract file-level frames from a coverage.py data file via JSON export."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                "--data-file",
                str(coverage_file),
                "--pretty-print",
                "-o",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
    except Exception:
        return []

    frames: list[dict[str, Any]] = []
    for filename in data.get("files", {}):
        frames.append({"file": filename, "lineno": None, "qualname": None})
    return frames


def _frames_from_junit(junit_dir: Path) -> list[dict[str, Any]]:
    """Extract coarse file paths from JUnit XML test class names."""
    frames: list[dict[str, Any]] = []
    seen: set[str] = set()
    for xml_path in sorted(junit_dir.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
            for tc in tree.iter("testcase"):
                classname = tc.get("classname", "")
                if classname and classname not in seen:
                    seen.add(classname)
                    # Convert dotted classname to a plausible file path
                    file_hint = classname.replace(".", "/") + ".py"
                    frames.append({"file": file_hint, "lineno": None, "qualname": None})
        except ET.ParseError:
            continue
    return frames


def _compute_evidence_digest(frames: list[dict], context: dict) -> str:
    payload = json.dumps(
        {"frames": frames, "context": context},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_evidence(
    *,
    platform: str,
    device: str,
    group: str,
    source_sha: str,
    base_sha: str | None,
    plan_hash: str,
    image_digest: str,
    context: dict,
    status: str,
    coverage_file: Path | None = None,
    junit_dir: Path | None = None,
    sampling_rate_hz: float = 0.0,
) -> dict[str, Any]:
    """Build a runtime-evidence-v1.json payload."""
    frames: list[dict[str, Any]] = []

    if coverage_file and coverage_file.exists():
        frames = _frames_from_coverage(coverage_file)

    if not frames and junit_dir and junit_dir.is_dir():
        frames = _frames_from_junit(junit_dir)

    evidence_digest = _compute_evidence_digest(frames, context) if frames else None

    return {
        "schema_version": EVIDENCE_SCHEMA,
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "source_sha": source_sha,
        "base_sha": base_sha,
        "plan_hash": plan_hash,
        "platform": platform,
        "device": device,
        "group": group,
        "image_digest": image_digest,
        "context": context,
        "frames": frames,
        "edges": [],
        "sampling_rate_hz": sampling_rate_hz,
        "complete": True,
        "status": status,
        "evidence_digest": evidence_digest,
        "created_at": _utcnow(),
        "sampler_overhead_ms": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--base-sha", default=None)
    parser.add_argument("--plan-hash", required=True)
    parser.add_argument("--image-digest", required=True)
    # Context dimensions as KEY=VALUE pairs
    parser.add_argument(
        "--context",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Context dimensions, e.g. --context torch=2.4 cuda=12.4",
    )
    parser.add_argument(
        "--status", required=True, choices=["passed", "failed", "cancelled", "unknown"]
    )
    parser.add_argument("--coverage-file", default=None)
    parser.add_argument("--junit-dir", default=None)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    context: dict[str, Any] = {}
    for item in args.context or []:
        if "=" not in item:
            print(f"::error::context item {item!r} must be KEY=VALUE", file=sys.stderr)
            return 1
        k, v = item.split("=", 1)
        context[k] = v

    evidence = build_evidence(
        platform=args.platform,
        device=args.device,
        group=args.group,
        source_sha=args.source_sha,
        base_sha=args.base_sha,
        plan_hash=args.plan_hash,
        image_digest=args.image_digest,
        context=context,
        status=args.status,
        coverage_file=Path(args.coverage_file) if args.coverage_file else None,
        junit_dir=Path(args.junit_dir) if args.junit_dir else None,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"evidence written: {out} ({len(evidence['frames'])} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
