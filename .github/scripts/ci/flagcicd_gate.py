#!/usr/bin/env python3
"""FlagCICD gate: slice, group-report, platform, gate subcommands (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flagcicd_contract as contract


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cmd_slice(args: argparse.Namespace) -> int:
    """Slice plan for one platform, output matrices."""
    plan_json = os.environ.get("FLAGCICD_PLAN", "")
    if not plan_json:
        print("ERROR: FLAGCICD_PLAN environment variable required", file=sys.stderr)
        return 1

    try:
        plan = contract.load_plan(plan_json)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Verify HEAD matches plan
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    current_sha = result.stdout.strip()
    if current_sha != plan["source_sha"]:
        print(
            f"ERROR: HEAD {current_sha[:12]} != plan source_sha {plan['source_sha'][:12]}",
            file=sys.stderr,
        )
        return 1

    # Find platform in plan
    platform = args.platform
    platform_entry = next((p for p in plan["platforms"] if p["name"] == platform), None)
    if not platform_entry:
        print(f"ERROR: Platform {platform} not in plan", file=sys.stderr)
        return 1

    # Verify config hash
    config_path = Path(args.configs_dir) / f"{platform}.yml"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        return 1

    actual_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if actual_hash != platform_entry["config_hash"]:
        print(
            f"ERROR: Config hash mismatch for {platform} "
            f"(expected {platform_entry['config_hash'][:12]}..., got {actual_hash[:12]}...)",
            file=sys.stderr,
        )
        return 1

    # Build matrices
    preflight = []
    parallel = []
    serial = []

    for group in plan["test_groups"]:
        if group["platform"] != platform:
            continue

        # Flatten exec into entry for compatibility with run_ci_test_group.py
        exec_data = group.get("exec") or {}
        entry = {
            "device": group["device"],
            "name": group["group"],
            "kind": group["kind"],
            "level": group["level"],
            "required": group["required"],
            "class": group["resources"]["class"],
            "devices": group["resources"]["devices"],
            "exclusive": group["resources"]["exclusive"],
            "timeout": group["resources"]["timeout_minutes"],
            "coverage": group.get("coverage", False),
            # Matrix entries historically omitted runner for shell launchers;
            # preserve that configuration format by defaulting to script.
            "runner": exec_data.get("runner", "script"),
            "path": exec_data.get("path"),
            "args": exec_data.get("args", []),
            "env": exec_data.get("env", {}),
            # Wave assignment: carried for wave-split routing; not consumed by
            # run_ci_test_group.py but harmless as an extra matrix key.
            "wave_assignment": group.get("wave_assignment", "priority-wave-1"),
            # P5: workload overrides from the resolved workload profile.  The
            # workflow expands these into TE_WORKLOAD_* env vars so test scripts
            # can honour them without parsing the full plan JSON.
            "workload_overrides": plan.get("fastci", {}).get("workload_overrides", {}),
        }

        lanes = {"preflight": preflight, "parallel": parallel, "serial": serial}
        if group["lane"] not in lanes:
            print(
                f"ERROR: Unknown lane {group['lane']} for {group['group']}",
                file=sys.stderr,
            )
            return 1
        lanes[group["lane"]].append(entry)

    # Split parallel lane into priority waves.  wave0 = groups that must finish
    # before wave1 starts (required L0/L1 smoke tests); wave1 = everything else.
    # When priority_enabled is False the workflow uses parallel_matrix unchanged.
    priority_enabled = bool(plan.get("fastci", {}).get("priority_enabled", False))
    wave0 = [e for e in parallel if e.get("wave_assignment") == "priority-wave-0"]
    wave1 = [e for e in parallel if e.get("wave_assignment") != "priority-wave-0"]

    # Write outputs
    output = {
        "preflight_matrix": json.dumps({"include": preflight}),
        "parallel_matrix": json.dumps({"include": parallel}),
        "serial_matrix": json.dumps({"include": serial}),
        "wave0_matrix": json.dumps({"include": wave0}),
        "wave1_matrix": json.dumps({"include": wave1}),
        "has_parallel": json.dumps(len(parallel) > 0),
        "has_serial": json.dumps(len(serial) > 0),
        "has_wave0": json.dumps(len(wave0) > 0),
        "has_wave1": json.dumps(len(wave1) > 0),
        "priority_enabled": json.dumps(priority_enabled),
        "preflight_selected": json.dumps(len(preflight) > 0),
        "coverage_active": json.dumps(
            any(g.get("coverage") for g in plan["test_groups"] if g["platform"] == platform)
        ),
        "coverage_required": json.dumps(bool(platform_entry.get("coverage_required", False))),
        "stage": plan["stage"],
        "test_level": plan["test_level"],
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
    }

    if args.github_output:
        gh_out = os.environ.get("GITHUB_OUTPUT")
        if gh_out:
            with open(gh_out, "a") as f:
                for key, value in output.items():
                    f.write(f"{key}={value}\n")

    return 0


def evidence_status(evidence: str, junit_dir: str | None) -> tuple[str, str | None]:
    """Judge a group whose command exited 0 by its declared evidence type."""
    if evidence == "junit":
        return parse_junit_status(junit_dir)
    if evidence == "exit_code":
        return "passed", None
    return "report_error", "evidence"


def cmd_group_report(args: argparse.Namespace) -> int:
    """Generate group report from step outcomes and test results."""
    # Load inputs
    try:
        steps = json.loads(os.environ.get("FLAGCICD_STEPS", "{}"))
        plan = contract.load_plan(os.environ.get("FLAGCICD_PLAN", ""))
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    platform = args.platform
    device = args.device
    group = args.group

    # The plan owns the evidence type, so a workflow cannot weaken the check.
    planned = next(
        (
            g
            for g in plan["test_groups"]
            if (g["platform"], g["device"], g["group"]) == (platform, device, group)
        ),
        None,
    )
    evidence = planned.get("evidence") if planned else None

    # Determine status from steps.  Checkout and execute are mandatory for
    # every executable group; optional artifact/identity/setup steps are
    # present only when configured by the platform.
    status = "passed"
    failure_stage = None
    executed = False

    missing_steps = [name for name in ("checkout", "execute") if name not in steps]
    if planned is None:
        status = "report_error"
        failure_stage = "plan"
    elif missing_steps:
        status = "infra_error"
        failure_stage = "report"

    # Check step outcomes in order
    step_names = ["checkout", "artifact", "identity", "setup", "execute"]
    for step_name in step_names:
        if failure_stage in ("plan", "report"):
            break
        step = steps.get(step_name, {})
        outcome = step.get("outcome", "")

        if outcome == "cancelled":
            status = "cancelled"
            failure_stage = "execute" if step_name == "execute" else step_name
            break
        elif outcome == "failure":
            if step_name == "execute":
                # Execute failure: check exit code
                if args.exit_code_file and Path(args.exit_code_file).exists():
                    try:
                        exit_code = int(Path(args.exit_code_file).read_text().strip())
                        executed = True
                        if exit_code == 0:
                            status, failure_stage = evidence_status(evidence, args.junit_dir)
                        else:
                            status = "test_failed"
                            failure_stage = "test_execution"
                    except (OSError, ValueError):
                        status = "infra_error"
                        failure_stage = "execute"
                else:
                    # Missing exit code file
                    status = "infra_error"
                    failure_stage = "execute"
            elif step_name == "setup":
                status = (
                    "build_failed" if "build" in step.get("name", "").lower() else "infra_error"
                )
                failure_stage = step_name
                break
            else:
                status = "infra_error"
                failure_stage = step_name
                break
        elif outcome == "success" and step_name == "execute":
            executed = True
            status, failure_stage = evidence_status(evidence, args.junit_dir)
        elif step_name == "execute":
            status = "skipped"
            failure_stage = "execute"

    # Build report
    report = {
        "schema_version": contract.GROUP_REPORT_SCHEMA,
        "repository": plan["repository"],
        "stage": plan["stage"],
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
        "platform": platform,
        "device": device,
        "group": group,
        "kind": args.kind,
        "level": args.level,
        "status": status,
        "failure_stage": failure_stage,
        "required": args.required,
        "executed": executed,
        "duration_seconds": int(float(args.duration)) if args.duration else 0,
        "created_at": utc_now(),
    }

    # Write report
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def parse_junit_status(junit_dir: str | None) -> tuple[str, str | None]:
    """Parse junit XML files, return (status, failure_stage)."""
    if not junit_dir or not Path(junit_dir).exists():
        return "report_error", "junit_missing"

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    has_corrupt = False

    for xml_file in Path(junit_dir).rglob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Handle both <testsuites> and <testsuite> roots
            suites = [root] if root.tag == "testsuite" else root.findall("testsuite")

            for suite in suites:
                total_tests += int(suite.get("tests", 0))
                total_failures += int(suite.get("failures", 0))
                total_errors += int(suite.get("errors", 0))
                total_skipped += int(suite.get("skipped", 0))
        except (ET.ParseError, OSError, ValueError):
            has_corrupt = True

    if has_corrupt:
        return "report_error", "junit_parsing"

    if total_tests == 0:
        return "skipped", "no_tests_collected"

    if total_tests == total_skipped:
        return "skipped", "all_skipped"

    if total_failures > 0 or total_errors > 0:
        return "test_failed", "test_execution"

    return "passed", None


def cmd_platform(args: argparse.Namespace) -> int:
    """Aggregate group reports into platform report."""
    # Load plan
    plan_json = os.environ.get("FLAGCICD_PLAN", "")
    if not plan_json:
        print(
            "ERROR: FLAGCICD_PLAN environment variable is empty or not set",
            file=sys.stderr,
        )
        return 1

    try:
        plan = contract.load_plan(plan_json)
    except ValueError as exc:
        print(f"ERROR: Failed to load plan: {exc}", file=sys.stderr)
        return 1

    platform = args.platform
    reports_dir = Path(args.reports_dir)

    # Ensure reports directory exists (may be empty if no groups ran)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Collect group reports
    group_reports = []
    for report_file in reports_dir.rglob("group-report.json"):
        try:
            report = json.loads(report_file.read_text())
            if report.get("platform") == platform:
                group_reports.append(report)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: Failed to load {report_file}: {exc}", file=sys.stderr)

    print(
        f"INFO: Found {len(group_reports)} group reports for platform {platform}",
        file=sys.stderr,
    )

    # Expected reports: groups from plan for this platform
    expected = [(g["device"], g["group"]) for g in plan["test_groups"] if g["platform"] == platform]
    print(
        f"INFO: Expected {len(expected)} groups for platform {platform}",
        file=sys.stderr,
    )

    # Validate reports
    errors = []
    if args.coverage_required and args.coverage_result != "success":
        errors.append(f"Required coverage job did not succeed: {args.coverage_result or 'unknown'}")
    actual_keys = set()
    malformed = []
    for report in group_reports:
        device = report.get("device")
        group = report.get("group")
        if not isinstance(device, str) or not isinstance(group, str) or not device or not group:
            malformed.append(report)
            continue
        actual_keys.add((device, group))
    if malformed:
        errors.append(f"Malformed group reports: {len(malformed)}")
    expected_keys = set(expected)

    # If lint_only or no groups expected, this is valid
    if len(expected) == 0:
        print(
            f"INFO: No groups expected for platform {platform} (lint_only or all blocked)",
            file=sys.stderr,
        )
        if len(group_reports) > 0:
            errors.append(f"Unexpected reports when no groups were planned: {actual_keys}")
    else:
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        duplicates = [
            key
            for key in actual_keys
            if sum(1 for r in group_reports if (r.get("device"), r.get("group")) == key) > 1
        ]

        if missing:
            print(
                f"ERROR: Missing group reports for platform {platform}:",
                file=sys.stderr,
            )
            for device, group in sorted(missing):
                print(f"  - {device}/{group}", file=sys.stderr)
            print(f"", file=sys.stderr)
            print(f"Possible causes:", file=sys.stderr)
            print(
                f"  1. parallel_groups or serial_groups job was skipped (check job conditions)",
                file=sys.stderr,
            )
            print(
                f"  2. test_groups_common.yml failed before uploading reports",
                file=sys.stderr,
            )
            print(
                f"  3. group-report.json was not generated (check 'Generate group report' step)",
                file=sys.stderr,
            )
            print(
                f"  4. artifact upload failed (check 'Upload group report' step)",
                file=sys.stderr,
            )
            errors.append(f"Missing reports: {missing}")
        if unexpected:
            errors.append(f"Unexpected reports: {unexpected}")
        if duplicates:
            errors.append(f"Duplicate reports: {duplicates}")

    # Validate each report
    for report in group_reports:
        # Check schema
        if report.get("schema_version") != contract.GROUP_REPORT_SCHEMA:
            errors.append(
                f"Group {report.get('group')}: wrong schema {report.get('schema_version')}"
            )

        # Check plan consistency
        if report.get("plan_hash") != plan["plan_hash"]:
            errors.append(f"Group {report.get('group')}: plan_hash mismatch")
        if report.get("source_sha") != plan["source_sha"]:
            errors.append(f"Group {report.get('group')}: source_sha mismatch")
        if report.get("platform") != platform:
            errors.append(f"Group {report.get('group')}: platform mismatch")

    # Check required groups passed and executed
    status = "passed"
    required_failed = []
    required_not_executed = []

    required_keys = {
        (group["device"], group["group"])
        for group in plan["test_groups"]
        if group["platform"] == platform and group.get("required")
    }
    for report in group_reports:
        if (report.get("device"), report.get("group")) in required_keys:
            if report.get("status") != "passed":
                required_failed.append(
                    (report.get("device"), report.get("group"), report.get("status"))
                )
                status = "failed"
            # Check execution (status != skipped means it was attempted)
            if report.get("executed") is not True:
                required_not_executed.append((report.get("device"), report.get("group")))

    if required_failed:
        errors.append(f"Required groups failed: {required_failed}")
    if required_not_executed:
        errors.append(f"Required groups not executed: {required_not_executed}")

    # Fail if any errors
    if errors:
        status = "failed"

    # Build platform report
    platform_report = {
        "schema_version": contract.PLATFORM_REPORT_SCHEMA,
        "repository": plan["repository"],
        "stage": plan["stage"],
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "platform": platform,
        "status": status,
        "groups": group_reports,
        "errors": errors,
        "created_at": utc_now(),
    }

    # Write report
    args.output.write_text(json.dumps(platform_report, indent=2, sort_keys=True) + "\n")

    # Exit with failure if not passed
    return 0 if status == "passed" and not errors else 1


def cmd_gate(args: argparse.Namespace) -> int:
    """Global gate: validate all platform reports."""
    # Load plan and needs
    try:
        plan = contract.load_plan(os.environ.get("FLAGCICD_PLAN", ""))
        needs = json.loads(os.environ.get("FLAGCICD_NEEDS", "{}"))
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Check plan job succeeded
    errors = []
    if needs.get(args.plan_job, {}).get("result") != "success":
        errors.append(f"Plan job {args.plan_job} did not succeed")

    # Check pipeline job
    expected_pipeline = "success" if len(plan["platforms"]) > 0 else "skipped"
    if needs.get(args.pipeline_job, {}).get("result") != expected_pipeline:
        errors.append(f"Pipeline job must be {expected_pipeline}")

    # Load platform reports
    platform_reports = {}
    duplicate_platform_reports = set()
    if args.reports_dir and Path(args.reports_dir).exists():
        for report_file in Path(args.reports_dir).rglob("*.json"):
            try:
                report = json.loads(report_file.read_text())
                if report.get("schema_version") == contract.PLATFORM_REPORT_SCHEMA:
                    pname = report.get("platform")
                    if pname:
                        if pname in platform_reports:
                            duplicate_platform_reports.add(pname)
                        platform_reports[pname] = report
            except (OSError, json.JSONDecodeError):
                pass

    # Validate platforms
    for platform in [p["name"] for p in plan["platforms"]]:
        if platform not in platform_reports:
            errors.append(f"Missing platform report: {platform}")
            continue

        report = platform_reports[platform]

        # Validate schema
        if report.get("schema_version") != contract.PLATFORM_REPORT_SCHEMA:
            errors.append(f"Platform {platform}: wrong schema {report.get('schema_version')}")

        # Validate consistency
        if report.get("plan_hash") != plan["plan_hash"]:
            errors.append(f"Platform {platform}: plan_hash mismatch")
        if report.get("source_sha") != plan["source_sha"]:
            errors.append(f"Platform {platform}: source_sha mismatch")
        if report.get("repository") != plan["repository"]:
            errors.append(f"Platform {platform}: repository mismatch")
        if report.get("stage") != plan["stage"]:
            errors.append(f"Platform {platform}: stage mismatch")

        # Validate status
        if report.get("status") != "passed":
            errors.append(f"Platform {platform}: status={report.get('status')}")

        # Check for platform-level errors
        if report.get("errors"):
            errors.append(f"Platform {platform}: has errors: {report['errors']}")

    # Fail closed on unexpected platforms
    unexpected_platforms = set(platform_reports.keys()) - {p["name"] for p in plan["platforms"]}
    if unexpected_platforms:
        errors.append(f"Unexpected platform reports: {unexpected_platforms}")
    if duplicate_platform_reports:
        errors.append(
            f"Duplicate platform reports: {', '.join(sorted(duplicate_platform_reports))}"
        )

    # A required blocked group is an explicit contract gap.  Keep it in the
    # plan for visibility and fail the release/manual gate instead of treating
    # the absent matrix entry as a successful run.
    required_blocked = [item for item in plan.get("blocked", []) if item.get("required")]
    if required_blocked:
        errors.append(f"Required groups are blocked: {required_blocked}")

    # P6: FastCI efficiency metrics — derived from plan and collected reports so
    # the gate summary is self-contained and can feed dashboards without re-reading
    # individual group reports.  All metrics are advisory; none affect gate_status.
    all_group_reports: list[dict] = []
    for pr in platform_reports.values():
        all_group_reports.extend(pr.get("groups", []))

    # selected_test_ratio: fraction of catalog groups that were scheduled to run.
    # Catalog = selected + blocked + excluded + pruned.
    n_selected = len(plan.get("test_groups", []))
    n_blocked = len(plan.get("blocked", []))
    n_excluded = len(plan.get("excluded", []))
    n_pruned = len(plan.get("pruned", []))
    n_catalog = n_selected + n_blocked + n_excluded + n_pruned
    selected_test_ratio = round(n_selected / n_catalog, 4) if n_catalog else None

    # affected_test_recall: of the required groups that were planned, how many
    # actually passed?  Measures whether the selection missed required coverage.
    required_planned = [g for g in plan.get("test_groups", []) if g.get("required")]
    required_total = len(required_planned)
    required_passed = sum(
        1 for r in all_group_reports if r.get("required") and r.get("status") == "passed"
    )
    affected_test_recall = round(required_passed / required_total, 4) if required_total else None

    # total_duration_seconds: wall-clock sum across all executed groups — a
    # proxy for GPU-resource consumption comparable across runs.
    total_duration = sum(r.get("duration_seconds", 0) for r in all_group_reports)

    fastci_metrics = {
        "selected_test_ratio": selected_test_ratio,
        "required_groups_total": required_total,
        "required_groups_passed": required_passed,
        "affected_test_recall": affected_test_recall,
        "total_duration_seconds": total_duration,
        "workload_profile": plan.get("fastci", {}).get("workload_profile", "nightly_spec"),
        "priority_enabled": plan.get("fastci", {}).get("priority_enabled", False),
    }

    # Build gate summary
    gate_status = "passed" if not errors else "failed"
    gate_summary = {
        "schema_version": contract.GATE_SCHEMA,
        "repository": plan["repository"],
        "stage": plan["stage"],
        "test_level": plan["test_level"],
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "status": gate_status,
        "platforms": platform_reports,
        "fastci_metrics": fastci_metrics,
        "errors": errors,
        "created_at": utc_now(),
    }

    # Write summary
    args.output.write_text(json.dumps(gate_summary, indent=2, sort_keys=True) + "\n")

    # Write step summary
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a") as f:
            f.write(f"### CI Gate: {gate_status}\n\n")
            f.write(f"- Stage: {plan['stage']}\n")
            f.write(f"- Level: {plan['test_level']}\n")
            f.write(f"- Platforms: {len(plan['platforms'])}\n\n")
            # P6: efficiency metrics block — advisory, always present.
            m = fastci_metrics
            f.write("**FastCI Metrics:**\n\n")
            sr = m["selected_test_ratio"]
            f.write(f"- Selected test ratio: {sr if sr is not None else 'n/a'}\n")
            recall = m["affected_test_recall"]
            f.write(
                f"- Affected test recall: {recall if recall is not None else 'n/a'}"
                f" ({m['required_groups_passed']}/{m['required_groups_total']} required)\n"
            )
            f.write(f"- Total duration: {m['total_duration_seconds']}s\n")
            f.write(f"- Workload profile: {m['workload_profile']}\n")
            f.write(f"- Priority enabled: {m['priority_enabled']}\n\n")
            if errors:
                f.write("**Errors:**\n\n")
                for error in errors:
                    f.write(f"- {error}\n")

    return 0 if gate_status == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # slice
    slice_parser = subparsers.add_parser("slice", help="Slice plan for platform matrices")
    slice_parser.add_argument("--platform", required=True)
    slice_parser.add_argument("--configs-dir", default=".github/configs")
    slice_parser.add_argument("--github-output", action="store_true")

    # group-report
    group_parser = subparsers.add_parser("group-report", help="Generate group report")
    group_parser.add_argument("--platform", required=True)
    group_parser.add_argument("--device", required=True)
    group_parser.add_argument("--group", required=True)
    group_parser.add_argument("--kind", required=True)
    group_parser.add_argument("--level", required=True)
    group_parser.add_argument("--required", type=lambda x: x.lower() == "true", required=True)
    group_parser.add_argument("--exit-code-file", type=Path)
    group_parser.add_argument("--junit-dir")
    group_parser.add_argument("--duration", default="0")
    group_parser.add_argument("--output", type=Path, required=True)

    # platform
    platform_parser = subparsers.add_parser("platform", help="Aggregate platform report")
    platform_parser.add_argument("--platform", required=True)
    platform_parser.add_argument("--reports-dir", type=Path, required=True)
    platform_parser.add_argument("--coverage-result", default="")
    platform_parser.add_argument(
        "--coverage-required",
        action="store_true",
        help="Fail when the required coverage job did not succeed",
    )
    platform_parser.add_argument("--output", type=Path, required=True)

    # gate
    gate_parser = subparsers.add_parser("gate", help="Global gate check")
    gate_parser.add_argument("--plan-job", default="plan")
    gate_parser.add_argument("--pipeline-job", default="platform-pipeline")
    gate_parser.add_argument("--reports-dir", type=Path)
    gate_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "slice":
        return cmd_slice(args)
    elif args.command == "group-report":
        return cmd_group_report(args)
    elif args.command == "platform":
        return cmd_platform(args)
    elif args.command == "gate":
        return cmd_gate(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
