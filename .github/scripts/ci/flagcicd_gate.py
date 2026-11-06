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
    integration = []

    for group in plan["execution_groups"]:
        if group["platform"] != platform:
            continue

        # Flatten exec into entry for compatibility with run_ci_test_group.py
        exec_data = group.get("exec") or {}
        entry = {
            **exec_data,
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
            "test_groups": group["test_groups"],
            # Matrix entries historically omitted runner for shell launchers;
            # preserve that configuration format by defaulting to script.
            "runner": exec_data.get("runner", "script"),
            "path": exec_data.get("path"),
            "args": exec_data.get("args", []),
            "env": exec_data.get("env", {}),
        }

        lanes = {"preflight": preflight, "parallel": parallel, "serial": serial}
        if group["lane"] not in lanes:
            print(
                f"ERROR: Unknown lane {group['lane']} for {group['group']}",
                file=sys.stderr,
            )
            return 1
        if group["kind"] == "integration":
            integration.append(entry)
        else:
            lanes[group["lane"]].append(entry)

    # Partial test selection still produces useful coverage, but cannot be
    # compared with or published as a complete platform coverage baseline.
    expected_coverage_groups = set(platform_entry.get("coverage_groups", []))
    selected_coverage_groups = {
        group["group"]
        for group in plan["execution_groups"]
        if group["platform"] == platform and group.get("coverage")
    }
    coverage_comparable = (
        platform_entry.get("coverage_full_scope", True)
        and bool(expected_coverage_groups)
        and (expected_coverage_groups == selected_coverage_groups)
    )

    # Write outputs
    output = {
        "preflight_matrix": json.dumps({"include": preflight}),
        "parallel_matrix": json.dumps({"include": parallel}),
        "serial_matrix": json.dumps({"include": serial}),
        "integration_matrix": json.dumps({"include": integration}),
        "has_integration": json.dumps(bool(integration)),
        "has_parallel": json.dumps(len(parallel) > 0),
        "has_serial": json.dumps(len(serial) > 0),
        "preflight_selected": json.dumps(len(preflight) > 0),
        "coverage_active": json.dumps(
            any(g.get("coverage") for g in plan["execution_groups"] if g["platform"] == platform)
        ),
        "coverage_required": json.dumps(
            bool(platform_entry.get("coverage_required", False))
            and any(
                g.get("coverage") for g in plan["execution_groups"] if g["platform"] == platform
            )
        ),
        "coverage_comparable": json.dumps(coverage_comparable),
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
            for g in plan["execution_groups"]
            if (g["platform"], g["device"], g["group"]) == (platform, device, group)
        ),
        None,
    )
    evidence = planned.get("evidence") if planned else None
    junit_dir = args.junit_dir
    exec_data = (planned.get("exec") or {}) if planned else {}
    if exec_data.get("runner") == "pytest":
        # Match the runner's log_dir default and {workspace} expansion.
        workspace = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd()))
        log_dir = str(exec_data.get("log_dir", "logs")).replace("{workspace}", str(workspace))
        junit_dir = str(workspace / log_dir)

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
                            status, failure_stage = evidence_status(evidence, junit_dir)
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
            status, failure_stage = evidence_status(evidence, junit_dir)
        elif step_name == "execute":
            status = "skipped"
            failure_stage = "execute"

    # Build a child report for every logical test group assigned to this job.
    # The command is shared, while required/evidence/level stay owned by the plan.
    targets = exec_data.get("group_test_targets", {})
    junit_status = (
        parse_junit_status(junit_dir)[0] if executed and evidence == "junit" and targets else None
    )
    can_attribute = (
        bool(targets)
        and executed
        and status in ("passed", "test_failed")
        and junit_status in ("passed", "test_failed")
        and (status == "passed" or junit_status == "test_failed")
    )
    group_results = test_group_results(junit_dir, targets) if can_attribute else {}
    group_evidence = {name: result["status"] == "passed" for name, result in group_results.items()}
    identity = {
        "schema_version": contract.GROUP_REPORT_SCHEMA,
        "repository": plan["repository"],
        "stage": plan["stage"],
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
        "platform": platform,
        "device": device,
        "duration_seconds": int(float(args.duration)) if args.duration else 0,
        "created_at": utc_now(),
    }
    child_reports = []
    for logical in plan["test_groups"]:
        if (logical["platform"], logical["device"], logical["execution_group"]) != (
            platform,
            device,
            group,
        ):
            continue
        child_status, child_stage, child_executed = status, failure_stage, executed
        counts = None
        if logical["group"] in group_results:
            result = group_results[logical["group"]]
            child_status, child_stage = result["status"], result["failure_stage"]
            child_executed = result["executed"]
            counts = result["tests"]
        child_reports.append(
            {
                **identity,
                "group": logical["group"],
                "execution_group": group,
                "kind": logical["kind"],
                "level": logical["level"],
                "required": logical["required"],
                "evidence": logical["evidence"],
                "status": child_status,
                "failure_stage": child_stage,
                "executed": child_executed,
                "tests": counts,
            }
        )
    if status == "passed" and any(child["status"] != "passed" for child in child_reports):
        status, failure_stage = "report_error", "test_group_evidence"
    report = {
        **identity,
        "group": group,
        "kind": planned["kind"] if planned else args.kind,
        "level": planned["level"] if planned else args.level,
        "required": planned["required"] if planned else args.required,
        "status": status,
        "failure_stage": failure_stage,
        "executed": executed,
        "test_group_evidence": group_evidence,
        "test_group_reports": child_reports,
    }

    # Write report
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def test_group_results(junit_dir: str | None, targets: dict) -> dict[str, dict]:
    """Attribute actual testcase outcomes to selected groups using canonical node IDs."""
    totals = {
        name: {"total": 0, "passed": 0, "failures": 0, "errors": 0, "skipped": 0}
        for name in targets
    }
    try:
        for xml_file in Path(junit_dir).rglob("*.xml") if junit_dir else []:
            root = ET.parse(xml_file).getroot()
            for node in root.iter():
                node.tag = node.tag.rsplit("}", 1)[-1]
            for case in root.iter("testcase"):
                nodeids = [
                    prop.get("value", "")
                    for prop in case.findall("properties/property")
                    if prop.get("name") == "flagcicd_test_nodeid"
                ]
                if len(nodeids) != 1 or "::" not in nodeids[0]:
                    continue  # A file path or inferred JUnit classname cannot prove a node ran.
                nodeid = nodeids[0]
                path, _, node = nodeid.partition("::")
                contract.split_test_target(path)
                if not node:
                    continue
                outcome = next(
                    (
                        key
                        for key, tag in (
                            ("errors", "error"),
                            ("failures", "failure"),
                            ("skipped", "skipped"),
                        )
                        if case.find(tag) is not None
                    ),
                    "passed",
                )
                for name, patterns in targets.items():
                    if any(contract.test_target_matches(nodeid, pattern) for pattern in patterns):
                        totals[name]["total"] += 1
                        totals[name][outcome] += 1
    except (OSError, ET.ParseError, ValueError):
        totals = {name: {key: 0 for key in counts} for name, counts in totals.items()}
    results = {}
    for name, counts in totals.items():
        if counts["failures"] or counts["errors"]:
            status, stage = "test_failed", "test_execution"
        elif counts["passed"]:
            status, stage = "passed", None
        else:
            status, stage = "report_error", "test_group_evidence"
        results[name] = {
            "status": status,
            "failure_stage": stage,
            "executed": bool(counts["passed"] or counts["failures"] or counts["errors"]),
            "tests": counts,
        }
    return results


def parse_junit_status(junit_dir: str | None) -> tuple[str, str | None]:
    """Parse junit XML files, return (status, failure_stage)."""
    if not junit_dir or not Path(junit_dir).exists():
        return "report_error", "junit_missing"

    def counts(node):
        """Count leaf cases; validate optional summaries without double counting."""
        if node.tag not in ("testsuite", "testsuites"):
            raise ValueError("Unsupported JUnit root")
        cases = list(node.iter("testcase"))
        result = [len(cases), 0, 0, 0]
        for case in cases:
            outcomes = [case.find(tag) is not None for tag in ("failure", "error", "skipped")]
            if outcomes[2] and any(outcomes[:2]):
                raise ValueError("Skipped case also failed")
            for index, present in enumerate(outcomes, 1):
                result[index] += present
        for suite in node.iter():
            if suite.tag not in ("testsuite", "testsuites"):
                continue
            subtree = list(suite.iter("testcase"))
            actual = [len(subtree)] + [
                sum(case.find(tag) is not None for case in subtree)
                for tag in ("failure", "error", "skipped")
            ]
            for key, value in zip(("tests", "failures", "errors", "skipped"), actual):
                if key in suite.attrib and int(suite.attrib[key]) != value:
                    raise ValueError(f"Inconsistent JUnit {key}")
        return result

    totals = [0, 0, 0, 0]
    for xml_file in Path(junit_dir).rglob("*.xml"):
        try:
            root = ET.parse(xml_file).getroot()
            # Accept namespaced producers as well as pytest's plain XML.
            for node in root.iter():
                node.tag = node.tag.rsplit("}", 1)[-1]
            totals = [a + b for a, b in zip(totals, counts(root))]
        except (ET.ParseError, OSError, ValueError):
            return "report_error", "junit_parsing"

    total_tests, total_failures, total_errors, total_skipped = totals
    if total_failures or total_errors:
        return "test_failed", "test_execution"
    if total_tests == 0:
        return "skipped", "no_tests_collected"
    if total_tests == total_skipped:
        return "skipped", "all_skipped"
    return "passed", None


def validate_test_group_reports(
    plan: dict,
    platform: str,
    reports: list,
) -> list[str]:
    """Check logical group completeness, identity and evidence against the plan."""
    expected = {
        (group["device"], group["group"]): group
        for group in plan["test_groups"]
        if group["platform"] == platform
    }
    if not isinstance(reports, list):
        return [f"{platform}: malformed logical test-group reports"]
    errors, actual = [], set()
    for report in reports:
        if not isinstance(report, dict):
            errors.append(f"{platform}: malformed logical test-group report")
            continue
        key = (report.get("device"), report.get("group"))
        if not all(isinstance(value, str) for value in key):
            errors.append(f"{platform}: malformed logical group key")
            continue
        if key in actual:
            errors.append(f"{platform}: duplicate logical test-group report {key}")
        actual.add(key)
        logical = expected.get(key)
        if logical is None:
            errors.append(f"{platform}: unexpected logical test-group report {key}")
            continue
        for field in ("repository", "stage", "source_sha", "plan_hash"):
            if report.get(field) != plan[field]:
                errors.append(f"{platform}/{key}: {field} mismatch")
        if (
            report.get("schema_version") != contract.GROUP_REPORT_SCHEMA
            or report.get("platform") != platform
        ):
            errors.append(f"{platform}/{key}: logical report schema/platform mismatch")
        if report.get("execution_group") != logical["execution_group"]:
            errors.append(f"{platform}/{key}: execution_group mismatch")
        if logical.get("required") and (
            report.get("status") != "passed" or report.get("executed") is not True
        ):
            errors.append(f"{platform}/{key}: required logical group did not pass and execute")
    missing = {key for key, group in expected.items() if group.get("required")} - actual
    if missing:
        errors.append(f"{platform}: missing required logical test-group reports: {sorted(missing)}")
    return errors


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
    if getattr(args, "unit_only", False):
        # Keep original plan identity while validating only the pre-integration scope.
        plan = dict(plan)
        for key in ("execution_groups", "test_groups"):
            plan[key] = [group for group in plan[key] if group["kind"] in ("unit", "preflight")]
    reports_dir = Path(args.reports_dir)

    # Ensure reports directory exists (may be empty if no groups ran)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Collect group reports
    group_reports = []
    for report_file in reports_dir.rglob("group-report.json"):
        try:
            report = json.loads(report_file.read_text())
            if report.get("platform") == platform:
                if getattr(args, "unit_only", False) and report.get("kind") in (
                    "integration",
                    "compat",
                    "benchmark",
                    "release",
                ):
                    continue
                group_reports.append(report)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: Failed to load {report_file}: {exc}", file=sys.stderr)

    print(
        f"INFO: Found {len(group_reports)} group reports for platform {platform}",
        file=sys.stderr,
    )

    # Expected reports: groups from plan for this platform
    expected = [
        (g["device"], g["group"]) for g in plan["execution_groups"] if g["platform"] == platform
    ]
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
        required_keys = {
            (g["device"], g["group"])
            for g in plan["execution_groups"]
            if g["platform"] == platform and g.get("required")
        }
        missing = required_keys - actual_keys
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
        for group in plan["execution_groups"]
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

    logical_reports = []
    for report in group_reports:
        children = report.get("test_group_reports", [])
        if not isinstance(children, list):
            errors.append(f"{platform}/{report.get('group')}: malformed logical test-group reports")
            continue
        # Validate container ownership here; validate identity, evidence and
        # completeness once below, including duplicates across containers.
        for child in children:
            if isinstance(child, dict) and (
                child.get("execution_group") != report.get("group")
                or child.get("device") != report.get("device")
            ):
                errors.append(
                    f"{platform}/{child.get('group')}: logical report belongs to a different"
                    " execution group/device"
                )
        logical_reports.extend(children)
    errors.extend(validate_test_group_reports(plan, platform, logical_reports))

    # Fail if any errors
    if errors:
        status = "failed"

    # Build platform report
    platform_entry = next((entry for entry in plan["platforms"] if entry["name"] == platform), {})
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
        "test_groups": logical_reports,
        "selected_test_groups": platform_entry.get("selected_test_groups", []),
        "unsupported_test_groups": platform_entry.get("unsupported_test_groups", []),
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

        errors.extend(validate_test_group_reports(plan, platform, report.get("test_groups", [])))

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
            f.write("| Platform | Result |\n| --- | --- |\n")
            for platform in plan["platforms"]:
                report = platform_reports.get(platform["name"])
                result = report.get("status", "unknown") if report else "missing report"
                f.write(f"| {platform['name']} | {result} |\n")
            f.write("\n")
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
    platform_parser.add_argument("--unit-only", action="store_true")
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
