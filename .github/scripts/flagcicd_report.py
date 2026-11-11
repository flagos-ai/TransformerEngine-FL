#!/usr/bin/env python3
"""Create and validate the repository-facing FlagCICD report contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "flagcicd.report/v1"
STATUSES = {
    "passed",
    "test_failed",
    "build_failed",
    "infra_error",
    "unsupported",
    "skipped",
    "cancelled",
    "report_error",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def coverage_metrics(path: Path | None, source_root: str, scope: str) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "language": "python",
            "tool": "coverage.py",
            "source_root": source_root,
            "scope": scope,
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    totals = payload.get("totals", {})
    files = payload.get("files", {})

    result: dict[str, Any] = {
        "available": True,
        "language": "python",
        "tool": "coverage.py",
        "source_root": source_root,
        "scope": scope,
        "metric_type": "line",
        "covered": totals.get("covered_lines", 0),
        "total": totals.get("num_statements", 0),
        "percent": totals.get("percent_covered", 0.0),
        "file_count": len(files),
    }
    if "covered_branches" in totals or "num_branches" in totals:
        result["branch"] = {
            "metric_type": "branch",
            "covered": totals.get("covered_branches", 0),
            "total": totals.get("num_branches", 0),
            "percent": totals.get("percent_covered_branches", 0.0),
        }
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    status = args.status
    if status not in STATUSES:
        raise ValueError(f"unsupported status: {status}")

    artifacts = [{"path": value, "kind": "coverage"} for value in args.artifact]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "task_id": args.task_id,
        "parent_run_id": args.parent_run_id,
        "attempt": args.attempt,
        "repository": args.repository,
        "source_sha": args.source_sha,
        "test_level": args.test_level,
        "platform": {
            "name": args.platform,
            "device": args.device,
            "image_ref": args.image_ref,
            "image_digest": args.image_digest,
            "config_hash": args.config_hash,
        },
        "status": status,
        "failure_reason": args.failure_reason,
        "tool": {
            "name": "coverage.py",
            "version": args.coverage_version,
        },
        "metrics": {"coverage": coverage_metrics(args.coverage, args.source_root, args.scope)},
        "child_jobs": args.child_job,
        "artifacts": artifacts,
        "created_at": utc_now(),
    }
    return report


def validate_report(report: dict[str, Any]) -> list[str]:
    required = {
        "schema_version",
        "run_id",
        "task_id",
        "repository",
        "source_sha",
        "test_level",
        "platform",
        "status",
        "metrics",
        "artifacts",
    }
    errors = [f"missing field: {key}" for key in sorted(required - report.keys())]
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must be flagcicd.report/v1")
    if report.get("status") not in STATUSES:
        errors.append(f"invalid status: {report.get('status')!r}")
    platform = report.get("platform")
    if not isinstance(platform, dict):
        errors.append("platform must be an object")
    coverage = report.get("metrics", {}).get("coverage")
    if not isinstance(coverage, dict):
        errors.append("metrics.coverage must be an object")
    elif coverage.get("available"):
        for key in ("metric_type", "covered", "total", "percent"):
            if key not in coverage:
                errors.append(f"coverage missing field: {key}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, help="Path to coverage.py JSON output")
    parser.add_argument("--validate", type=Path, help="Path to existing report to validate only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument("--task-id", default=os.environ.get("FLAGCICD_TASK_ID", ""))
    parser.add_argument("--parent-run-id", default=os.environ.get("FLAGCICD_PARENT_RUN_ID", ""))
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", "TransformerEngine-FL"),
    )
    parser.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--test-level", default=os.environ.get("FLAGCICD_TEST_LEVEL", "L1"))
    parser.add_argument("--platform", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--image-ref", default=os.environ.get("FLAGCICD_IMAGE_REF", ""))
    parser.add_argument("--image-digest", default=os.environ.get("FLAGCICD_IMAGE_DIGEST", ""))
    parser.add_argument("--config-hash", default=os.environ.get("FLAGCICD_CONFIG_HASH", ""))
    parser.add_argument("--status", default="passed")
    parser.add_argument("--failure-reason", default="")
    parser.add_argument("--coverage-version", default="")
    parser.add_argument("--source-root", default="transformer_engine")
    parser.add_argument("--scope", default="unit")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--child-job", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate:
        report = json.loads(args.validate.read_text(encoding="utf-8"))
        errors = validate_report(report)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print(f"Valid FlagCICD report: {args.validate}")
        return 0

    if args.output is None:
        raise SystemExit("--output is required when creating a report")
    report = build_report(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    errors = validate_report(report)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Created FlagCICD report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
