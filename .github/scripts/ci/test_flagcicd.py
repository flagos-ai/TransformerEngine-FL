"""Unit tests for FlagCICD v1 core components.

Tests gate validation, identity verification, and contract schemas without
requiring accelerator hardware or TransformerEngine installation.

Run: python -m unittest .github/scripts/ci/test_flagcicd.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

# Import modules under test
import flagcicd_gate as gate
import flagcicd_identity as identity
import flagcicd_contract as contract
import flagcicd_planner as planner


ROOT = Path(__file__).resolve().parents[3]
SHA = "a" * 40


def make_plan(platforms=(), test_groups=(), blocked=(), stage="pr", execution_groups=None):
    """Return a hash-valid plan containing logical groups and runtime jobs.

    Most isolated gate fixtures supply runtime jobs and use one logical group per
    job. Planner integration fixtures provide both lists explicitly.
    """
    if execution_groups is None:
        execution_groups = copy.deepcopy(list(test_groups))
        logical_groups = []
        for execution in execution_groups:
            names = execution.setdefault("test_groups", [execution["group"]])
            for name in names:
                logical = copy.deepcopy(execution)
                logical["group"] = name
                logical["execution_group"] = execution["group"]
                logical.pop("test_groups", None)
                logical_groups.append(logical)
    else:
        logical_groups = copy.deepcopy(list(test_groups))
        execution_groups = copy.deepcopy(list(execution_groups))
    plan = {
        "schema_version": contract.PLAN_SCHEMA,
        "repository": "TransformerEngine-FL",
        "stage": stage,
        "selection": "affected",
        "lint_only": False,
        "source_sha": SHA,
        "base_sha": None,
        "test_level": "L2",
        "test_level_rule_version": "v2",
        "changed_files": {"count": 0, "documentation_only": False},
        "common_path_triggers": [],
        "platforms": list(platforms),
        "test_groups": logical_groups,
        "execution_groups": execution_groups,
        "excluded": [],
        "blocked": list(blocked),
        "reasons": [],
        "coverage_policy": {"method": "absolute"},
    }
    plan["plan_hash"] = contract.compute_plan_hash(plan)
    return plan


def make_group(name, lane, platform="cuda", device="a100", evidence="junit", kind="unit"):
    return {
        "platform": platform,
        "device": device,
        "group": name,
        "test_groups": [name],
        "kind": kind,
        "level": "L0" if lane == "preflight" else "L1",
        "required": True,
        "lane": lane,
        "evidence": evidence,
        "resources": {
            "class": "single_device",
            "devices": 0 if lane == "preflight" else (8 if lane == "serial" else 1),
            "exclusive": lane == "serial",
            "timeout_minutes": 90 if lane == "preflight" else 30,
        },
        "coverage": lane == "parallel",
        "exec": (None if lane == "preflight" else {"path": f"run_{name}.sh", "runner": "script"}),
    }


def platform_report(plan, platform, **overrides):
    report = {
        "schema_version": contract.PLATFORM_REPORT_SCHEMA,
        "repository": plan["repository"],
        "stage": plan["stage"],
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "platform": platform,
        "status": "passed",
        "errors": [],
        "test_groups": [
            logical_report(plan, group)
            for group in plan["test_groups"]
            if group["platform"] == platform
        ],
    }
    report.update(overrides)
    return report


def logical_report(plan, group, **overrides):
    report = {
        "schema_version": contract.GROUP_REPORT_SCHEMA,
        "repository": plan["repository"],
        "stage": plan["stage"],
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "platform": group["platform"],
        "device": group["device"],
        "group": group["group"],
        "execution_group": group["execution_group"],
        "status": "passed",
        "executed": True,
        "required": group["required"],
    }
    report.update(overrides)
    return report


def execution_report(plan, execution, **overrides):
    report = {
        "schema_version": contract.GROUP_REPORT_SCHEMA,
        "repository": plan["repository"],
        "stage": plan["stage"],
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "platform": execution["platform"],
        "device": execution["device"],
        "group": execution["group"],
        "kind": execution["kind"],
        "status": "passed",
        "executed": True,
        "test_group_reports": [
            logical_report(plan, group)
            for group in plan["test_groups"]
            if (group["platform"], group["device"], group["execution_group"])
            == (execution["platform"], execution["device"], execution["group"])
        ],
    }
    report.update(overrides)
    return report


class ContractSchemaTests(unittest.TestCase):
    """Test contract schema constants and validation."""

    def test_all_catalog_kinds_are_valid(self):
        for kind in ("unit", "preflight", "integration", "compat", "benchmark", "release"):
            with self.subTest(kind=kind):
                group = make_group("sample", "serial", kind=kind)
                plan = make_plan(platforms=[{"name": "cuda"}], test_groups=[group])
                self.assertEqual(contract.validate_plan(plan), [])

    def test_schema_version_constants(self):
        """Verify schema version constants are defined."""
        self.assertEqual(contract.PLAN_SCHEMA, "flagcicd.plan/v1")
        self.assertEqual(contract.GROUP_REPORT_SCHEMA, "flagcicd.group-report/v1")
        self.assertEqual(contract.PLATFORM_REPORT_SCHEMA, "flagcicd.platform-report/v1")
        self.assertEqual(contract.GATE_SCHEMA, "flagcicd.gate/v1")
        self.assertEqual(contract.IDENTITY_SCHEMA, "flagcicd.artifact-identity/v1")

    def test_status_constants(self):
        """Verify status constants are defined."""
        self.assertIn("passed", contract.STATUSES)
        self.assertIn("test_failed", contract.STATUSES)
        self.assertIn("build_failed", contract.STATUSES)
        self.assertIn("infra_error", contract.STATUSES)

    def test_level_constants(self):
        """Verify test level constants."""
        self.assertEqual(contract.LEVELS, ("L0", "L1", "L2", "L3", "L4"))

    def test_canonical_json_deterministic(self):
        """Verify canonical JSON is deterministic."""
        obj = {"z": 1, "a": 2, "m": [3, 2, 1]}
        json1 = contract.canonical_json(obj)
        json2 = contract.canonical_json(obj)
        self.assertEqual(json1, json2)
        self.assertIn('"a":2', json1)
        self.assertTrue(json1.index('"a"') < json1.index('"z"'))

    def test_compute_plan_hash(self):
        """Verify plan hash computation."""
        plan = {
            "schema_version": "flagcicd.plan/v1",
            "source_sha": "a" * 40,
            "stage": "pr",
            "platforms": ["cuda"],
            "test_groups": [],
        }
        hash1 = contract.compute_plan_hash(plan)
        self.assertEqual(len(hash1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in hash1))

        # Same plan should produce same hash
        hash2 = contract.compute_plan_hash(plan)
        self.assertEqual(hash1, hash2)

        # Different plan should produce different hash
        plan["stage"] = "nightly"
        hash3 = contract.compute_plan_hash(plan)
        self.assertNotEqual(hash1, hash3)

    def test_plan_contract_requires_consistent_logical_execution_membership(self):
        runtime = make_group("pytorch_unit", "serial")
        runtime["test_groups"] = ["core", "optimizer"]
        valid = make_plan(test_groups=[runtime])
        self.assertEqual(contract.validate_plan(valid), [])
        for change in (
            "missing_jobs",
            "missing_logical",
            "duplicate_logical",
            "duplicate_membership",
            "wrong_execution",
        ):
            with self.subTest(change=change):
                plan = copy.deepcopy(valid)
                if change == "missing_jobs":
                    plan.pop("execution_groups")
                elif change == "missing_logical":
                    plan["test_groups"].pop()
                elif change == "duplicate_logical":
                    plan["test_groups"].append(copy.deepcopy(plan["test_groups"][0]))
                elif change == "duplicate_membership":
                    plan["execution_groups"][0]["test_groups"].append("core")
                else:
                    plan["test_groups"][0]["execution_group"] = "pytorch_debug"
                plan["plan_hash"] = contract.compute_plan_hash(plan)
                self.assertTrue(contract.validate_plan(plan))

    def test_coverage_report_contract_requires_no_risk_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, "report.json")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / ".github/scripts/flagcicd_report.py"),
                    "--platform",
                    "cuda",
                    "--test-level",
                    "L2",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertFalse(any("risk" in key for key in report))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / ".github/scripts/flagcicd_report.py"),
                    "--validate",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


class IdentityTests(unittest.TestCase):
    """Test artifact identity verification."""

    def test_write_identity_creates_valid_json(self):
        """Test writing identity file with required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "test.whl").write_bytes(b"wheel")

            # Use real argparse Namespace with all required fields
            args = argparse.Namespace(
                artifact_dir=str(artifact_dir),
                platform="cuda",
                source_sha="a" * 40,
                config_hash="b" * 64,
                coverage_active=True,
                test_marker="unittest",
                image_ref="flagscale/te-cuda:latest",
                image_digest="sha256:abc123",
                test_level="L1",
                timestamp="2024-01-01T00:00:00Z",
            )

            identity.cmd_write(args)

            identity_file = artifact_dir / "artifact-identity.json"
            self.assertTrue(identity_file.exists())
            data = json.loads(identity_file.read_text())

            self.assertEqual(data["schema_version"], contract.IDENTITY_SCHEMA)
            self.assertEqual(data["platform"], "cuda")
            self.assertEqual(data["source_sha"], "a" * 40)
            self.assertEqual(data["config_hash"], "b" * 64)
            self.assertTrue(data["coverage_instrumentation"])
            self.assertEqual(data["image_ref"], "flagscale/te-cuda:latest")
            self.assertEqual(identity.cmd_verify(args), 0)

    def test_verify_identity_accepts_matching_identity(self):
        """Test verification passes with matching identity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            identity_file = artifact_dir / "artifact-identity.json"
            (artifact_dir / "test.whl").write_bytes(b"wheel")
            identity_file.write_text(
                json.dumps(
                    {
                        "schema_version": contract.IDENTITY_SCHEMA,
                        "platform": "cuda",
                        "source_sha": "a" * 40,
                        "config_hash": "b" * 64,
                        "coverage_active": False,
                        "test_marker": None,
                        "wheels": [
                            {
                                "name": "test.whl",
                                "size": 5,
                                "sha256": hashlib.sha256(b"wheel").hexdigest(),
                            }
                        ],
                        "toolchain": {},
                    }
                )
            )

            args = Mock(
                artifact_dir=str(artifact_dir),
                platform="cuda",
                source_sha="a" * 40,
                config_hash="b" * 64,
            )

            # Should not raise
            result = identity.cmd_verify(args)
            self.assertEqual(result, 0)

    def test_verify_identity_rejects_platform_mismatch(self):
        """Test verification fails on platform mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            identity_file = artifact_dir / "artifact-identity.json"
            (artifact_dir / "test.whl").write_bytes(b"wheel")
            identity_file.write_text(
                json.dumps(
                    {
                        "schema_version": contract.IDENTITY_SCHEMA,
                        "platform": "cuda",
                        "source_sha": "a" * 40,
                        "config_hash": "b" * 64,
                        "coverage_active": False,
                        "test_marker": None,
                        "wheels": [
                            {
                                "name": "test.whl",
                                "size": 5,
                                "sha256": hashlib.sha256(b"wheel").hexdigest(),
                            }
                        ],
                        "toolchain": {},
                    }
                )
            )

            args = Mock(
                artifact_dir=str(artifact_dir),
                platform="ascend",  # Mismatch
                source_sha="a" * 40,
                config_hash="b" * 64,
            )

            result = identity.cmd_verify(args)
            self.assertNotEqual(result, 0)

    def test_verify_identity_rejects_sha_mismatch(self):
        """Test verification fails on source_sha mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            identity_file = artifact_dir / "artifact-identity.json"
            (artifact_dir / "test.whl").write_bytes(b"wheel")
            identity_file.write_text(
                json.dumps(
                    {
                        "schema_version": contract.IDENTITY_SCHEMA,
                        "platform": "cuda",
                        "source_sha": "a" * 40,
                        "config_hash": "b" * 64,
                        "coverage_active": False,
                        "test_marker": None,
                        "wheels": [
                            {
                                "name": "test.whl",
                                "size": 5,
                                "sha256": hashlib.sha256(b"wheel").hexdigest(),
                            }
                        ],
                        "toolchain": {},
                    }
                )
            )

            args = Mock(
                artifact_dir=str(artifact_dir),
                platform="cuda",
                source_sha="c" * 40,  # Mismatch
                config_hash="b" * 64,
            )

            result = identity.cmd_verify(args)
            self.assertNotEqual(result, 0)

    def test_payload_integrity(self):
        for change in (
            "unchanged",
            "same_size",
            "truncated",
            "missing",
            "extra",
            "duplicate",
            "escape",
            "empty",
            "schema",
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                wheel = folder / "test.whl"
                wheel.write_bytes(b"original")
                manifest = {
                    "schema_version": contract.IDENTITY_SCHEMA,
                    "platform": "cuda",
                    "source_sha": "a" * 40,
                    "config_hash": "b" * 64,
                    "wheels": [
                        {
                            "name": wheel.name,
                            "size": 8,
                            "sha256": hashlib.sha256(b"original").hexdigest(),
                        }
                    ],
                }
                if change == "same_size":
                    wheel.write_bytes(b"replaced")
                if change == "truncated":
                    wheel.write_bytes(b"x")
                if change == "missing":
                    wheel.unlink()
                if change == "extra":
                    (folder / "old.whl").write_bytes(b"old")
                if change == "duplicate":
                    manifest["wheels"] *= 2
                if change == "escape":
                    manifest["wheels"][0]["name"] = "../test.whl"
                if change == "empty":
                    manifest["wheels"] = []
                if change == "schema":
                    manifest["schema_version"] = "unknown"
                (folder / "artifact-identity.json").write_text(json.dumps(manifest))
                args = argparse.Namespace(
                    artifact_dir=folder,
                    platform="cuda",
                    source_sha="a" * 40,
                    config_hash="b" * 64,
                )
                self.assertEqual(identity.cmd_verify(args), 0 if change == "unchanged" else 1)


class GateSliceTests(unittest.TestCase):
    """cmd_slice partitions one platform's plan into lane matrices."""

    def run_slice(self, plan, config_text="device_types: [a100]\n"):
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = Path(tmpdir, "configs")
            configs.mkdir()
            (configs / "cuda.yml").write_text(config_text)
            output = Path(tmpdir, "github-output")
            output.write_text("")
            args = argparse.Namespace(platform="cuda", configs_dir=str(configs), github_output=True)
            head = Mock(stdout=plan["source_sha"] + "\n")
            env = {"FLAGCICD_PLAN": json.dumps(plan), "GITHUB_OUTPUT": str(output)}
            with patch.object(gate.subprocess, "run", return_value=head), patch.dict(
                gate.os.environ, env
            ):
                code = gate.cmd_slice(args)
            values = dict(
                line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
            )
        return code, values

    def plan_for(
        self,
        groups,
        config_text="device_types: [a100]\n",
        coverage_groups=None,
        execution_groups=None,
    ):
        entry = {
            "name": "cuda",
            "config_hash": hashlib.sha256(config_text.encode()).hexdigest(),
            "coverage_required": True,
        }
        if coverage_groups is not None:
            entry["coverage_groups"] = list(coverage_groups)
        other = make_group("pytorch_unit", "parallel", platform="ascend", device="910b")
        if execution_groups is not None:
            return make_plan(
                platforms=[entry], test_groups=groups, execution_groups=execution_groups
            )
        return make_plan(platforms=[entry], test_groups=[*groups, other])

    def test_lanes_are_partitioned_per_platform(self):
        plan = self.plan_for(
            [
                make_group("preflight", "preflight"),
                make_group("pytorch_unit", "parallel"),
                make_group("pytorch_serial_unit", "serial", evidence="exit_code"),
            ],
            coverage_groups=["pytorch_unit"],
        )
        code, out = self.run_slice(plan)
        self.assertEqual(code, 0)
        names = {
            lane: [entry["name"] for entry in json.loads(out[f"{lane}_matrix"])["include"]]
            for lane in ("preflight", "parallel", "serial")
        }
        self.assertEqual(
            names,
            {
                "preflight": ["preflight"],
                "parallel": ["pytorch_unit"],
                "serial": ["pytorch_serial_unit"],
            },
        )
        self.assertEqual(
            (out["has_parallel"], out["has_serial"], out["preflight_selected"]),
            ("true", "true", "true"),
        )
        self.assertEqual((out["coverage_active"], out["coverage_required"]), ("true", "true"))
        self.assertEqual(out["coverage_comparable"], "true")
        preflight = json.loads(out["preflight_matrix"])["include"][0]
        self.assertEqual((preflight["device"], preflight["timeout"]), ("a100", 90))
        serial = json.loads(out["serial_matrix"])["include"][0]
        self.assertEqual((serial["path"], serial["devices"]), ("run_pytorch_serial_unit.sh", 8))

    def test_integration_is_separated_from_both_unit_lanes(self):
        for lane in ("parallel", "serial"):
            with self.subTest(lane=lane):
                group = make_group("mcore", lane, kind="integration", evidence="exit_code")
                code, out = self.run_slice(self.plan_for([group]))
                self.assertEqual(code, 0)
                self.assertEqual(out["has_integration"], "true")
                self.assertEqual(out["has_parallel"], "false")
                self.assertEqual(out["has_serial"], "false")
                self.assertEqual(
                    json.loads(out["integration_matrix"])["include"][0]["name"], "mcore"
                )

    def test_empty_lanes_report_false(self):
        code, out = self.run_slice(self.plan_for([make_group("preflight", "preflight")]))
        self.assertEqual(code, 0)
        self.assertEqual((out["has_parallel"], out["has_serial"]), ("false", "false"))
        self.assertEqual(json.loads(out["parallel_matrix"]), {"include": []})
        self.assertEqual((out["coverage_active"], out["coverage_required"]), ("false", "false"))

    def test_group_selection_survives_matrix_slicing(self):
        group = make_group("pytorch_unit", "serial")
        selection = {
            "test_targets": ["tests/pytorch/test_fused_adam.py"],
            "group_test_targets": {"optimizer": ["tests/pytorch/test_fused_adam.py"]},
        }
        group["test_groups"] = ["optimizer"]
        group["exec"].update(selection)
        code, out = self.run_slice(self.plan_for([group], coverage_groups=["pytorch_unit"]))
        self.assertEqual(code, 0)
        entry = json.loads(out["serial_matrix"])["include"][0]
        self.assertEqual({key: entry[key] for key in selection}, selection)
        self.assertEqual(entry["test_groups"], ["optimizer"])

    def test_shared_execution_is_sliced_once_for_multiple_logical_groups(self):
        execution = make_group("pytorch_unit", "serial")
        execution["test_groups"] = ["core", "optimizer"]
        plan = self.plan_for([execution])
        self.assertEqual(
            [group["group"] for group in plan["test_groups"][:2]],
            ["core", "optimizer"],
        )
        code, out = self.run_slice(plan)
        self.assertEqual(code, 0)
        entries = json.loads(out["serial_matrix"])["include"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "pytorch_unit")
        self.assertEqual(entries[0]["test_groups"], ["core", "optimizer"])

    def test_group_subset_is_not_comparable_even_when_all_suites_are_selected(self):
        suites = [
            "pytorch_debug",
            "pytorch_unit",
            "pytorch_distributed",
            "pytorch_onnx",
        ]
        plan = self.plan_for(
            [make_group(name, "serial") for name in suites], coverage_groups=suites
        )
        plan["platforms"][0]["coverage_full_scope"] = False
        plan["plan_hash"] = contract.compute_plan_hash(plan)
        code, out = self.run_slice(plan)
        self.assertEqual(code, 0)
        self.assertEqual(out["coverage_comparable"], "false")

    def test_coverage_comparison_requires_complete_declared_scope(self):
        suites = [
            "pytorch_debug",
            "pytorch_unit",
            "pytorch_distributed",
            "pytorch_onnx",
        ]
        for selected, expected_scope, expected in (
            (suites, suites, "true"),
            (["pytorch_onnx"], suites, "false"),
            (suites, None, "false"),
            (suites, [], "false"),
            ([], suites, "false"),
            (suites + ["additional_coverage"], suites, "false"),
        ):
            with self.subTest(selected=selected, expected_scope=expected_scope):
                groups = [make_group(name, "parallel") for name in selected]
                code, out = self.run_slice(self.plan_for(groups, coverage_groups=expected_scope))
                self.assertEqual(code, 0)
                self.assertEqual(out["coverage_comparable"], expected)

    def test_planned_coverage_scope_survives_nightly_and_partial_pr_slicing(self):
        config_text = (ROOT / ".github/configs/cuda.yml").read_text()
        for stage, changed, expected in (
            ("nightly", [], "true"),
            ("pr", ["tests/pytorch/test_onnx_export.py"], "false"),
        ):
            with self.subTest(stage=stage):
                args = argparse.Namespace(
                    source_root=ROOT,
                    project=".flagcicd/project.yaml",
                    configs_dir=".github/configs",
                    stage=stage,
                    source_sha="",
                    base_ref="",
                    changed_files=json.dumps(changed),
                    platforms="",
                    requested_level="auto",
                    labels_from_event=False,
                )
                plan = planner.build_plan(args)
                code, out = self.run_slice(plan, config_text=config_text)
                self.assertEqual(code, 0)
                self.assertEqual(out["coverage_active"], "true")
                self.assertEqual(out["coverage_comparable"], expected)

    def test_partial_coverage_is_excluded_from_baseline_comparisons(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/all_tests_common.yml").read_text())
        outputs = workflow["jobs"]["checkout_and_config"]["outputs"]
        self.assertEqual(
            outputs["coverage_comparable"],
            "${{ steps.slice.outputs.coverage_comparable }}",
        )
        steps = {step["name"]: step for step in workflow["jobs"]["coverage"]["steps"]}
        for name in ("Check PR coverage regression", "Check daily 3-sigma regression"):
            self.assertIn("outputs.coverage_comparable == 'true'", steps[name]["if"])
        artifact_name = steps["Upload coverage report"]["with"]["name"]
        self.assertIn("outputs.coverage_comparable != 'true' && '-partial'", artifact_name)

    def test_unknown_lane_fails(self):
        code, _ = self.run_slice(self.plan_for([make_group("probe", "fast")]))
        self.assertEqual(code, 1)

    def test_config_hash_mismatch_fails(self):
        plan = self.plan_for([make_group("preflight", "preflight")])
        code, _ = self.run_slice(plan, config_text="device_types: [h100]\n")
        self.assertEqual(code, 1)

    def test_pytest_config_survives_planner_slice_and_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            probe = folder / "pytest_probe.py"
            probe.write_text(
                """import os, sys
from pathlib import Path
assert "--group-option" in sys.argv and "--step-option" in sys.argv
assert os.environ["GROUP_ENV"] == "group" and os.environ["STEP_ENV"] == "step"
xml = next(arg.split("=",1)[1] for arg in sys.argv if arg.startswith("--junitxml="))
Path(xml).write_text('<testsuite tests="1"><testcase name="probe"><properties><property name="flagcicd_test_nodeid" value="probe_target::probe"/></properties></testcase></testsuite>')
"""
            )
            project = planner.load_project(ROOT / ".flagcicd/project.yaml")
            project["test_groups"] = [
                group for group in project["test_groups"] if group["name"] in ("preflight", "core")
            ]
            next(group for group in project["test_groups"] if group["name"] == "core")[
                "test_targets"
            ] = ["probe_target"]
            platforms = planner.load_platforms(ROOT / ".github/configs")
            # The probe uses a synthetic catalog containing only preflight/core.
            platforms["cuda"]["config"]["unsupported_test_groups"] = {}
            pytest_entry = {
                "name": "pytorch_unit",
                "runner": "pytest",
                "max_retries": 0,
                "pytest_args": ["--group-option"],
                "log_dir": str(folder / "logs"),
                "env": {"GROUP_ENV": "group"},
                "steps": [
                    {
                        "name": "probe",
                        "targets": ["probe_target"],
                        "args": ["--step-option"],
                        "env": {"STEP_ENV": "step"},
                        "junit": "probe.xml",
                    }
                ],
                # Execution metadata must not override the catalog's required flag.
                "required": False,
            }
            entries = platforms["cuda"]["config"]["unit_test_matrix"]
            platforms["cuda"]["config"]["unit_test_matrix"] = [
                pytest_entry if entry["name"] == "pytorch_unit" else entry for entry in entries
            ]
            groups = planner.build_test_groups(project, platforms, ["cuda"], "L1", [])
            executions = planner.build_execution_groups(project, platforms, groups, full_scope=True)
            plan = self.plan_for(groups, execution_groups=executions)
            code, outputs = self.run_slice(plan)
            self.assertEqual(code, 0)
            entry = next(
                entry
                for entry in json.loads(outputs["serial_matrix"])["include"]
                if entry["name"] == "pytorch_unit"
            )
            self.assertTrue(entry["required"])
            env = dict(
                os.environ,
                TE_TEST_GROUP_JSON=json.dumps(entry),
                TE_TEST_PYTEST_COMMAND=f"{sys.executable} {probe}",
            )
            result = subprocess.run(
                [sys.executable, str(ROOT / "tests/test_utils/run_ci_test_group.py")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(gate.parse_junit_status(str(folder / "logs")), ("passed", None))

            args = argparse.Namespace(
                platform="cuda",
                device="a100",
                group="pytorch_unit",
                kind="unit",
                level="L1",
                required=True,
                exit_code_file=None,
                junit_dir=str(folder / "wrong"),
                duration="0",
                output=folder / "group-report.json",
            )
            steps = {
                "checkout": {"outcome": "success"},
                "execute": {"outcome": "success"},
            }
            with patch.dict(
                os.environ,
                FLAGCICD_PLAN=json.dumps(plan),
                FLAGCICD_STEPS=json.dumps(steps),
            ):
                self.assertEqual(gate.cmd_group_report(args), 0)
            self.assertEqual(json.loads(args.output.read_text())["status"], "passed")

    @unittest.skipUnless(importlib.util.find_spec("pytest"), "pytest is required")
    def test_plan_slice_script_and_real_junit_agree(self):
        """Run the entire test group selection handoff using tiny CPU-only tests."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            tests = folder / "tests/pytorch"
            tests.mkdir(parents=True)
            (tests / "test_deferred_init.py").write_text("def test_core(): assert True\n")
            optimizer = tests / "test_fused_optimizer.py"
            optimizer.write_text("def test_optimizer(): assert True\n")
            (tests / "test_unrelated.py").write_text("raise RuntimeError('must not import')\n")
            script = folder / "suite.sh"
            script.write_text(
                '"$PYTHON_BIN" -m pytest -q tests/pytorch --junitxml="$XML_LOG_DIR/suite.xml"\n'
            )
            project = planner.load_project(ROOT / ".flagcicd/project.yaml")
            platforms = planner.load_platforms(ROOT / ".github/configs")
            platforms["cuda"]["config"]["unit_test_matrix"] = [
                {
                    "name": "pytorch_unit",
                    "runner": "script",
                    "path": "suite.sh",
                    "env": {"PYTHON_BIN": sys.executable},
                    "max_retries": 0,
                }
            ]
            changed = ["tests/pytorch/test_fused_optimizer.py"]
            selection = planner.select_test_groups(project, changed, "L1", "pr")
            groups = planner.build_test_groups(
                project,
                platforms,
                ["cuda"],
                "L1",
                changed,
                selection=selection,
            )
            executions = planner.build_execution_groups(
                project, platforms, groups, full_scope=False
            )
            plan = self.plan_for(groups, execution_groups=executions)
            code, outputs = self.run_slice(plan)
            self.assertEqual(code, 0)
            entry = json.loads(outputs["serial_matrix"])["include"][0]
            log_dir = folder / "logs"
            log_dir.mkdir()
            env = dict(
                os.environ,
                TE_TEST_GROUP_JSON=json.dumps(entry),
                GITHUB_WORKSPACE=str(folder),
                XML_LOG_DIR=str(log_dir),
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                PYTEST_PLUGINS="",
            )
            for failure in (False, True):
                if failure:
                    optimizer.write_text("def test_optimizer(): assert False\n")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tests/test_utils/run_ci_test_group.py"),
                    ],
                    env=env,
                    cwd=folder,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, int(failure), result.stdout + result.stderr)
                args = argparse.Namespace(
                    platform="cuda",
                    device="a100",
                    group="pytorch_unit",
                    kind="unit",
                    level="L1",
                    required=True,
                    exit_code_file=None,
                    junit_dir=str(log_dir),
                    duration="0",
                    output=folder / "group-report.json",
                )
                steps = {
                    "checkout": {"outcome": "success"},
                    "execute": {"outcome": "success"},
                }
                with patch.dict(
                    os.environ,
                    FLAGCICD_PLAN=json.dumps(plan),
                    FLAGCICD_STEPS=json.dumps(steps),
                ):
                    self.assertEqual(gate.cmd_group_report(args), 0)
                report = json.loads(args.output.read_text())
                self.assertEqual(report["status"], "test_failed" if failure else "passed")
                if not failure:
                    self.assertEqual(
                        report["test_group_evidence"], {"core": True, "optimizer": True}
                    )
                self.assertEqual(
                    {child["group"] for child in report["test_group_reports"]},
                    {"core", "optimizer"},
                )
                for execution in plan["execution_groups"]:
                    if execution["group"] != "pytorch_unit":
                        other = folder / execution["group"] / "group-report.json"
                        other.parent.mkdir(exist_ok=True)
                        other.write_text(json.dumps(execution_report(plan, execution)))
                platform_args = argparse.Namespace(
                    platform="cuda",
                    reports_dir=folder,
                    coverage_required=False,
                    coverage_result="success",
                    output=folder / "platform-report.json",
                )
                with patch.dict(os.environ, FLAGCICD_PLAN=json.dumps(plan)):
                    self.assertEqual(gate.cmd_platform(platform_args), int(failure))
                checked_platform = json.loads(platform_args.output.read_text())
                self.assertEqual(
                    {child["group"] for child in checked_platform["test_groups"]},
                    {group["group"] for group in plan["test_groups"]},
                )


class GroupReportEvidenceTests(unittest.TestCase):
    """A group's declared evidence decides how a zero exit code is judged."""

    def test_missing_junit_is_not_reported_as_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, failure_stage = gate.parse_junit_status(str(Path(tmpdir) / "missing"))
        self.assertEqual((status, failure_stage), ("report_error", "junit_missing"))

    def test_empty_junit_directory_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, failure_stage = gate.parse_junit_status(tmpdir)
        self.assertEqual((status, failure_stage), ("skipped", "no_tests_collected"))

    def report(
        self,
        group,
        evidence,
        outcome="success",
        exit_code="0",
        plan_group=None,
        xml=None,
        extra_steps=None,
    ):
        plan = make_plan(test_groups=[plan_group or make_group(group, "serial", evidence=evidence)])
        steps = {
            "checkout": {"outcome": "success"},
            "execute": {"outcome": outcome},
            **(extra_steps or {}),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_file = Path(tmpdir, "exit_code")
            exit_file.write_text(exit_code)
            junit = Path(tmpdir, "junit")
            junit.mkdir()
            if xml is not None:
                (junit / "report.xml").write_text(xml)
            output = Path(tmpdir, "group-report.json")
            args = argparse.Namespace(
                platform="cuda",
                device="a100",
                group=group,
                kind="integration",
                level="L2",
                required=True,
                exit_code_file=exit_file,
                junit_dir=str(junit),
                duration="5",
                output=output,
            )
            env = {
                "FLAGCICD_PLAN": json.dumps(plan),
                "FLAGCICD_STEPS": json.dumps(steps),
            }
            with patch.dict(gate.os.environ, env):
                self.assertEqual(gate.cmd_group_report(args), 0)
            return json.loads(output.read_text())

    def test_exit_code_group_passes_without_junit(self):
        report = self.report("pytorch_mcore_integration", "exit_code")
        self.assertEqual((report["status"], report["executed"]), ("passed", True))

    def test_junit_group_without_tests_is_skipped(self):
        report = self.report("pytorch_unit", "junit")
        self.assertEqual(
            (report["status"], report["failure_stage"]),
            ("skipped", "no_tests_collected"),
        )

    def test_nonzero_exit_fails_regardless_of_evidence(self):
        report = self.report("pytorch_mcore_integration", "exit_code", "failure", "1")
        self.assertEqual(
            (report["status"], report["failure_stage"]),
            ("test_failed", "test_execution"),
        )

    def test_group_missing_from_plan_is_report_error(self):
        other = make_group("pytorch_unit", "parallel")
        report = self.report("unplanned", "exit_code", plan_group=other)
        self.assertEqual((report["status"], report["failure_stage"]), ("report_error", "plan"))

    def test_unknown_evidence_is_report_error(self):
        report = self.report("pytorch_unit", "stdout")
        self.assertEqual((report["status"], report["failure_stage"]), ("report_error", "evidence"))

    def selected_groups(self):
        group = make_group("pytorch_unit", "serial")
        group["test_groups"] = ["core", "optimizer"]
        group["exec"].update(
            {
                "test_targets": [
                    "tests/pytorch/test_recipe.py",
                    "tests/pytorch/test_fused_adam.py",
                ],
                "group_test_targets": {
                    "core": ["tests/pytorch/test_recipe.py"],
                    "optimizer": ["tests/pytorch/test_fused_adam.py"],
                },
            }
        )
        return group

    @staticmethod
    def selected_case(path, outcome="", node="probe"):
        return (
            '<testcase name="probe"><properties><property name="flagcicd_test_path"'
            f' value="{path}"/><property name="flagcicd_test_nodeid"'
            f' value="{path}::{node}"/></properties>{outcome}</testcase>'
        )

    def test_selected_groups_require_real_passing_test_evidence(self):
        group = self.selected_groups()
        cases = self.selected_case("tests/pytorch/test_recipe.py")
        cases += self.selected_case("tests/pytorch/test_fused_adam.py")
        report = self.report(
            "pytorch_unit",
            "junit",
            plan_group=group,
            xml=f"<testsuite>{cases}</testsuite>",
        )
        self.assertEqual((report["status"], report["executed"]), ("passed", True))
        self.assertEqual(report["test_group_evidence"], {"core": True, "optimizer": True})
        children = {child["group"]: child for child in report["test_group_reports"]}
        self.assertEqual(set(children), {"core", "optimizer"})
        for child in children.values():
            self.assertEqual(child["status"], "passed")
            self.assertTrue(child["executed"])
            self.assertEqual(child["execution_group"], "pytorch_unit")

    def test_unrelated_or_skipped_test_cannot_claim_group_evidence(self):
        group = self.selected_groups()
        for extra in (
            self.selected_case("tests/pytorch/test_sanity.py"),
            self.selected_case("tests/pytorch/test_fused_adam.py", "<skipped/>"),
            '<testcase name="test_fused_adam" classname="tests.pytorch.test_fused_adam"/>',
        ):
            with self.subTest(extra=extra):
                cases = self.selected_case("tests/pytorch/test_recipe.py") + extra
                report = self.report(
                    "pytorch_unit",
                    "junit",
                    plan_group=group,
                    xml=f"<testsuite>{cases}</testsuite>",
                )
                self.assertEqual(report["status"], "report_error")
                self.assertEqual(report["failure_stage"], "test_group_evidence")
                self.assertEqual(
                    {child["group"]: child["status"] for child in report["test_group_reports"]},
                    {"core": "passed", "optimizer": "report_error"},
                )

    def test_mixed_file_linear_case_cannot_certify_normalization_or_attention(self):
        catalog = {
            g["name"]: g
            for g in yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())["test_groups"]
        }
        targets = {
            name: catalog[name]["test_targets"] for name in ("normalization", "gemm", "attention")
        }
        probes = (
            ("tests/pytorch/test_sanity.py", "test_sanity_linear[small]", "gemm"),
            (
                "tests/pytorch/test_sanity.py",
                "test_sanity_normalization_amp[RMSNorm]",
                "normalization",
            ),
            ("tests/pytorch/test_numerics.py", "test_dpa_accuracy[small]", "attention"),
            ("tests/pytorch/test_fusible_ops.py", "TestBasicOps::test_identity", None),
            (
                "tests/plugin/backend/npu/test_backend_npu.py",
                "TestNPUGroupedGEMMAccuracy::test_grouped_gemm_basic",
                "gemm",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for path, node, expected in probes:
                with self.subTest(path=path, node=node):
                    Path(tmp, "report.xml").write_text(
                        "<testsuite>" + self.selected_case(path, node=node) + "</testsuite>"
                    )
                    self.assertEqual(
                        {
                            name: result["status"] == "passed"
                            for name, result in gate.test_group_results(tmp, targets).items()
                        },
                        {name: name == expected for name in targets},
                    )
            Path(tmp, "report.xml").write_text(
                '<testsuite><testcase name="test_sanity_linear"><properties><property'
                ' name="flagcicd_test_path"'
                ' value="tests/pytorch/test_sanity.py"/></properties></testcase></testsuite>'
            )
            self.assertTrue(
                all(
                    result["status"] == "report_error"
                    for result in gate.test_group_results(tmp, targets).values()
                )
            )

    def test_shared_execution_failure_is_attributed_to_its_own_group(self):
        group = self.selected_groups()
        cases = self.selected_case("tests/pytorch/test_recipe.py")
        cases += self.selected_case("tests/pytorch/test_fused_adam.py", "<failure>broken</failure>")
        report = self.report(
            "pytorch_unit",
            "junit",
            outcome="failure",
            exit_code="1",
            plan_group=group,
            xml=f"<testsuite>{cases}</testsuite>",
        )
        children = {child["group"]: child for child in report["test_group_reports"]}
        self.assertEqual(report["status"], "test_failed")
        self.assertEqual(children["core"]["status"], "passed")
        self.assertEqual(children["optimizer"]["status"], "test_failed")
        self.assertEqual(children["core"]["tests"]["passed"], 1)
        self.assertEqual(children["optimizer"]["tests"]["failures"], 1)
        # One successful parameter cannot hide another failed parameter of that group.
        cases += self.selected_case("tests/pytorch/test_fused_adam.py", node="probe[second]")
        report = self.report(
            "pytorch_unit",
            "junit",
            outcome="failure",
            exit_code="1",
            plan_group=group,
            xml=f"<testsuite>{cases}</testsuite>",
        )
        self.assertEqual(report["test_group_evidence"], {"core": True, "optimizer": False})

    def test_common_setup_failure_and_cancellation_are_not_exonerated_by_junit(self):
        group = self.selected_groups()
        cases = self.selected_case("tests/pytorch/test_recipe.py") + self.selected_case(
            "tests/pytorch/test_fused_adam.py"
        )
        for kwargs, expected in (
            ({"extra_steps": {"setup": {"outcome": "failure"}}}, "infra_error"),
            ({"outcome": "cancelled"}, "cancelled"),
            ({"outcome": "failure", "exit_code": "1"}, "test_failed"),
        ):
            with self.subTest(expected=expected):
                report = self.report(
                    "pytorch_unit",
                    "junit",
                    plan_group=group,
                    xml=f"<testsuite>{cases}</testsuite>",
                    **kwargs,
                )
                self.assertEqual(report["status"], expected)
                self.assertTrue(
                    all(child["status"] == expected for child in report["test_group_reports"])
                )

    def test_unattributed_failed_case_keeps_shared_job_failed(self):
        group = self.selected_groups()
        cases = self.selected_case("tests/pytorch/test_recipe.py") + self.selected_case(
            "tests/pytorch/test_fused_adam.py"
        )
        cases += self.selected_case("tests/pytorch/test_unknown.py", "<failure/>")
        report = self.report(
            "pytorch_unit",
            "junit",
            outcome="failure",
            exit_code="1",
            plan_group=group,
            xml=f"<testsuite>{cases}</testsuite>",
        )
        self.assertEqual(report["status"], "test_failed")
        self.assertTrue(all(child["status"] == "passed" for child in report["test_group_reports"]))

    def parse(self, xml):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "report.xml").write_text(xml)
            return gate.parse_junit_status(tmp)

    def test_malformed_reports_cannot_pass(self):
        reports = [
            '<testsuite tests="1" failures="0"><testcase><failure/></testcase></testsuite>',
            '<testsuite tests="1"/>',
            '<testsuite tests="-1"/>',
            '<testsuite tests="oops"/>',
            '<testsuite tests="2"><testcase/></testsuite>',
            "<testsuite><testcase><skipped/><error/></testcase></testsuite>",
            "<unknown><testcase/></unknown>",
            "<testsuite>",
        ]
        for xml in reports:
            with self.subTest(xml=xml):
                self.assertEqual(self.parse(xml), ("report_error", "junit_parsing"))

    def test_actual_case_outcomes_without_summary_attributes(self):
        for tag in ("failure", "error"):
            with self.subTest(tag=tag):
                self.assertEqual(
                    self.parse(f"<testsuite><testcase><{tag}/></testcase></testsuite>"),
                    ("test_failed", "test_execution"),
                )
        self.assertEqual(
            self.parse("<testsuite><testcase><skipped/></testcase></testsuite>"),
            ("skipped", "all_skipped"),
        )

    def test_nested_namespaced_suites_are_counted_once(self):
        xml = """<testsuites xmlns="urn:junit" tests="2" skipped="1">
          <testsuite tests="2" skipped="1"><testsuite tests="1"><testcase/></testsuite>
          <testcase><skipped/></testcase></testsuite></testsuites>"""
        self.assertEqual(self.parse(xml), ("passed", None))


class GatePlatformReportTests(unittest.TestCase):
    """Test gate platform-report validation logic."""

    def run_platform(self, plan, reports, unit_only=False):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for index, report in enumerate(reports):
                target = folder / str(index) / "group-report.json"
                target.parent.mkdir()
                target.write_text(json.dumps(report))
            args = argparse.Namespace(
                unit_only=unit_only,
                platform="cuda",
                reports_dir=folder,
                coverage_required=False,
                coverage_result="success",
                output=folder / "platform-report.json",
            )
            with patch.dict(os.environ, FLAGCICD_PLAN=json.dumps(plan)):
                code = gate.cmd_platform(args)
            return code, json.loads(args.output.read_text())

    def test_unit_gate_does_not_wait_for_integration_but_rejects_failed_units(self):
        unit = make_group("pytorch_unit", "serial")
        integration = make_group("mcore", "parallel", kind="integration", evidence="exit_code")
        plan = make_plan(platforms=[{"name": "cuda"}], test_groups=[unit, integration])
        valid = execution_report(plan, plan["execution_groups"][0])
        self.assertEqual(self.run_platform(plan, [valid], unit_only=True)[0], 0)
        self.assertEqual(self.run_platform(plan, [], unit_only=True)[0], 1)
        failed = copy.deepcopy(valid)
        failed["status"] = "failed"
        self.assertEqual(self.run_platform(plan, [failed], unit_only=True)[0], 1)
        integration_report = execution_report(plan, plan["execution_groups"][1])
        self.assertEqual(self.run_platform(plan, [valid, integration_report], unit_only=True)[0], 0)
        # Final platform validation must still require integration evidence.
        self.assertEqual(self.run_platform(plan, [valid])[0], 1)

    def test_logical_reports_must_cover_the_plan_and_match_execution(self):
        runtime = make_group("pytorch_unit", "serial")
        runtime["test_groups"] = ["core", "optimizer"]
        plan = make_plan(platforms=[{"name": "cuda"}], test_groups=[runtime])
        valid = execution_report(plan, plan["execution_groups"][0])
        code, report = self.run_platform(plan, [valid])
        self.assertEqual((code, report["status"]), (0, "passed"))
        self.assertEqual({child["group"] for child in report["test_groups"]}, {"core", "optimizer"})
        for change in (
            "missing",
            "duplicate",
            "source",
            "hash",
            "device",
            "execution",
            "unexpected",
            "unexecuted",
        ):
            with self.subTest(change=change):
                damaged = copy.deepcopy(valid)
                children = damaged["test_group_reports"]
                if change == "missing":
                    children.pop()
                elif change == "duplicate":
                    children.append(copy.deepcopy(children[0]))
                elif change == "unexpected":
                    children[-1]["group"] = "attention"
                elif change == "unexecuted":
                    children[-1]["executed"] = False
                else:
                    field, value = {
                        "source": ("source_sha", "b" * 40),
                        "hash": ("plan_hash", "0" * 64),
                        "device": ("device", "h100"),
                        "execution": ("execution_group", "pytorch_debug"),
                    }[change]
                    children[-1][field] = value
                code, report = self.run_platform(plan, [damaged])
                self.assertEqual(code, 1)
                self.assertTrue(report["errors"])

    def test_required_coverage_cannot_be_skipped(self):
        plan = {
            "schema_version": contract.PLAN_SCHEMA,
            "repository": "TransformerEngine-FL",
            "stage": "pr",
            "selection": "affected",
            "lint_only": False,
            "source_sha": "a" * 40,
            "base_sha": None,
            "test_level": "L1",
            "test_level_rule_version": "v1",
            "changed_files": {"count": 0, "documentation_only": False},
            "common_path_triggers": [],
            "platforms": [
                {
                    "name": "ascend",
                    "coverage_required": True,
                    "device_types": ["ascend"],
                }
            ],
            "test_groups": [],
            "execution_groups": [],
            "excluded": [],
            "blocked": [],
            "reasons": [],
            "coverage_policy": {"method": "absolute"},
        }
        plan["plan_hash"] = contract.compute_plan_hash(plan)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "platform-report.json"
            args = argparse.Namespace(
                platform="ascend",
                reports_dir=Path(tmpdir),
                coverage_result="skipped",
                coverage_required=True,
                output=output,
            )
            with patch.dict(
                gate.os.environ,
                {"FLAGCICD_PLAN": contract.canonical_json(plan)},
                clear=False,
            ):
                self.assertEqual(gate.cmd_platform(args), 1)
            report = json.loads(output.read_text())
            self.assertTrue(any("coverage" in error.lower() for error in report["errors"]))

    def test_failed_logical_group_is_reported_once(self):
        plan = make_plan(platforms=[{"name": "cuda"}], test_groups=[make_group("probe", "serial")])
        damaged = execution_report(plan, plan["execution_groups"][0])
        damaged["test_group_reports"][0]["status"] = "test_failed"
        code, report = self.run_platform(plan, [damaged])
        self.assertEqual(code, 1)
        errors = [error for error in report["errors"] if "required logical group" in error]
        self.assertEqual(len(errors), 1)

    def test_logical_reports_cannot_move_between_execution_containers(self):
        plan = make_plan(
            platforms=[{"name": "cuda"}],
            test_groups=[make_group("first", "serial"), make_group("second", "serial")],
        )
        reports = [execution_report(plan, group) for group in plan["execution_groups"]]
        reports[0]["test_group_reports"], reports[1]["test_group_reports"] = (
            reports[1]["test_group_reports"],
            reports[0]["test_group_reports"],
        )
        code, report = self.run_platform(plan, reports)
        self.assertEqual(code, 1)
        self.assertTrue(
            any("different execution group/device" in error for error in report["errors"])
        )

    def test_malformed_and_cross_container_duplicate_children_are_rejected(self):
        plan = make_plan(
            platforms=[{"name": "cuda"}],
            test_groups=[make_group("first", "serial"), make_group("second", "serial")],
        )
        for damage in ("malformed", "duplicate"):
            with self.subTest(damage=damage):
                reports = [execution_report(plan, group) for group in plan["execution_groups"]]
                if damage == "malformed":
                    reports[0]["test_group_reports"] = {}
                else:
                    reports[1]["test_group_reports"].append(
                        copy.deepcopy(reports[0]["test_group_reports"][0])
                    )
                code, report = self.run_platform(plan, reports)
                self.assertEqual(code, 1)
                self.assertTrue(any(damage in error for error in report["errors"]))

    def test_optional_failures_and_missing_reports_allow_final_gate(self):
        for required, include_report in (
            (False, True),
            (False, False),
            (True, True),
            (True, False),
        ):
            with self.subTest(
                required=required, include_report=include_report
            ), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                group = make_group("probe", "parallel")
                group["required"] = required
                plan = make_plan(platforms=[{"name": "cuda"}], test_groups=[group])
                report = {
                    "schema_version": contract.GROUP_REPORT_SCHEMA,
                    "platform": "cuda",
                    "device": "a100",
                    "group": "probe",
                    "status": "test_failed",
                    "executed": True,
                    "plan_hash": plan["plan_hash"],
                    "source_sha": plan["source_sha"],
                    "test_group_reports": [
                        logical_report(plan, plan["test_groups"][0], status="test_failed")
                    ],
                }
                if include_report:
                    (folder / "group-report.json").write_text(json.dumps(report))
                args = argparse.Namespace(
                    platform="cuda",
                    reports_dir=folder,
                    coverage_required=False,
                    coverage_result="failure",
                    output=folder / "platform-report.json",
                )
                with patch.dict(os.environ, FLAGCICD_PLAN=json.dumps(plan)):
                    code = gate.cmd_platform(args)
                self.assertEqual(code, int(required))
                platform_report = json.loads(args.output.read_text())
                code, _ = GateTests().run_gate(plan, [platform_report])
                self.assertEqual(code, int(required))

    def test_coverage_collection_failure_only_blocks_when_required(self):
        for required in (False, True):
            with self.subTest(required=required), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                plan = make_plan(platforms=[{"name": "cuda"}])
                args = argparse.Namespace(
                    platform="cuda",
                    reports_dir=folder,
                    coverage_required=required,
                    coverage_result="failure",
                    output=folder / "platform-report.json",
                )
                with patch.dict(os.environ, FLAGCICD_PLAN=json.dumps(plan)):
                    self.assertEqual(gate.cmd_platform(args), int(required))

    def test_workflow_failure_policy_is_on_concrete_jobs(self):
        tests = yaml.safe_load((ROOT / ".github/workflows/test_groups_common.yml").read_text())
        coverage = yaml.safe_load((ROOT / ".github/workflows/all_tests_common.yml").read_text())
        self.assertEqual(
            tests["jobs"]["test_group"]["continue-on-error"],
            "${{ matrix.required != true }}",
        )
        self.assertEqual(
            coverage["jobs"]["coverage"]["continue-on-error"],
            "${{ needs.checkout_and_config.outputs.coverage_required != 'true' }}",
        )


class GateTests(unittest.TestCase):
    """cmd_gate fails closed on every report/plan inconsistency."""

    def run_gate(self, plan, reports, pipeline="success"):
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir, "reports")
            for index, report in enumerate(reports):
                path = reports_dir / f"flagcicd-platform-{index}" / "platform-report.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(report))
            output = Path(tmpdir, "gate.json")
            args = argparse.Namespace(
                plan_job="plan",
                pipeline_job="platform-pipeline",
                reports_dir=reports_dir,
                output=output,
            )
            needs = {
                "plan": {"result": "success"},
                "platform-pipeline": {"result": pipeline},
            }
            env = {
                "FLAGCICD_PLAN": json.dumps(plan),
                "FLAGCICD_NEEDS": json.dumps(needs),
            }
            with patch.dict(gate.os.environ, env):
                code = gate.cmd_gate(args)
            return code, json.loads(output.read_text())

    def setUp(self):
        self.plan = make_plan(platforms=[{"name": "cuda"}, {"name": "ascend"}])

    def test_all_reports_pass(self):
        reports = [
            platform_report(self.plan, "cuda"),
            platform_report(self.plan, "ascend"),
        ]
        code, summary = self.run_gate(self.plan, reports)
        self.assertEqual((code, summary["status"], summary["errors"]), (0, "passed", []))

    def test_missing_platform_report_fails(self):
        code, summary = self.run_gate(self.plan, [platform_report(self.plan, "cuda")])
        self.assertEqual(code, 1)
        self.assertIn("Missing platform report: ascend", summary["errors"])

    def test_duplicate_platform_report_fails(self):
        reports = [
            platform_report(self.plan, "cuda"),
            platform_report(self.plan, "cuda"),
            platform_report(self.plan, "ascend"),
        ]
        code, summary = self.run_gate(self.plan, reports)
        self.assertEqual(code, 1)
        self.assertIn("Duplicate platform reports: cuda", summary["errors"])

    def test_plan_hash_mismatch_fails(self):
        reports = [
            platform_report(self.plan, "cuda", plan_hash="0" * 64),
            platform_report(self.plan, "ascend"),
        ]
        code, summary = self.run_gate(self.plan, reports)
        self.assertEqual(code, 1)
        self.assertIn("Platform cuda: plan_hash mismatch", summary["errors"])

    def test_required_blocked_group_fails(self):
        blocked = [{"platform": "cuda", "group": "model_validation", "required": True}]
        plan = make_plan(platforms=[{"name": "cuda"}], blocked=blocked, stage="release")
        code, summary = self.run_gate(plan, [platform_report(plan, "cuda")])
        self.assertEqual(code, 1)
        self.assertTrue(any("Required groups are blocked" in e for e in summary["errors"]))

    def test_optional_blocked_group_does_not_fail(self):
        blocked = [{"platform": "cuda", "group": "pytorch_benchmark", "required": False}]
        plan = make_plan(platforms=[{"name": "cuda"}], blocked=blocked)
        code, _ = self.run_gate(plan, [platform_report(plan, "cuda")])
        self.assertEqual(code, 0)

    def test_final_gate_rechecks_logical_test_group_reports(self):
        runtime = make_group("pytorch_unit", "serial")
        runtime["test_groups"] = ["core", "optimizer"]
        plan = make_plan(platforms=[{"name": "cuda"}], test_groups=[runtime])
        valid = platform_report(plan, "cuda")
        self.assertEqual(self.run_gate(plan, [valid])[0], 0)
        for change in (
            "missing",
            "duplicate",
            "identity",
            "execution",
            "unexpected",
            "skipped",
        ):
            with self.subTest(change=change):
                report = copy.deepcopy(valid)
                children = report["test_groups"]
                if change == "missing":
                    children.pop()
                elif change == "duplicate":
                    children.append(copy.deepcopy(children[0]))
                elif change == "identity":
                    children[-1]["source_sha"] = "b" * 40
                elif change == "execution":
                    children[-1]["execution_group"] = "pytorch_debug"
                elif change == "unexpected":
                    children[-1]["group"] = "attention"
                else:
                    children[-1]["status"] = "skipped"
                code, summary = self.run_gate(plan, [report])
                self.assertEqual(code, 1)
                self.assertTrue(summary["errors"])

    def test_failed_pipeline_job_fails(self):
        reports = [
            platform_report(self.plan, "cuda"),
            platform_report(self.plan, "ascend"),
        ]
        code, summary = self.run_gate(self.plan, reports, pipeline="failure")
        self.assertEqual(code, 1)
        self.assertIn("Pipeline job must be success", summary["errors"])


class RegressionTests(unittest.TestCase):
    """flagcicd_regression accepts the raw coverage json the workflow writes."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "flagcicd_regression", ROOT / ".github/scripts/flagcicd_regression.py"
        )
        cls.regression = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.regression)

    def run_cli(self, current, *extra):
        with tempfile.TemporaryDirectory() as tmpdir:
            current_path = Path(tmpdir, "current.json")
            current_path.write_text(json.dumps({"totals": {"percent_covered": current}}))
            args = []
            for flag, value in zip(extra[::2], extra[1::2]):
                path = Path(tmpdir, flag.strip("-") + ".json")
                path.write_text(json.dumps(value))
                args += [flag, str(path)]
            output = Path(tmpdir, "verdict.json")
            result = subprocess.run(
                [
                    "python3",
                    str(ROOT / ".github/scripts/flagcicd_regression.py"),
                    "--current",
                    str(current_path),
                    *args,
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            return result.returncode, json.loads(output.read_text())["result"]

    def test_raw_coverage_json_against_raw_baseline(self):
        code, result = self.run_cli(70.0, "--baseline", {"totals": {"percent_covered": 75.0}})
        self.assertEqual((code, result["status"]), (1, "regression"))
        code, result = self.run_cli(74.0, "--baseline", {"totals": {"percent_covered": 75.0}})
        self.assertEqual((code, result["status"]), (0, "acceptable"))

    def test_zero_variance_history_flags_any_drop(self):
        code, result = self.run_cli(79.9, "--historical", [80.0] * 8)
        self.assertEqual((code, result["status"]), (1, "outlier_below"))
        self.assertIn("zero variance", result["message"])
        code, result = self.run_cli(80.0, "--historical", [80.0] * 8)
        self.assertEqual((code, result["status"]), (0, "within_bounds"))

    def test_unknown_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "report.json")
            path.write_text(json.dumps({"schema_version": "other/v1"}))
            with self.assertRaises(ValueError):
                self.regression.load_report(path)


if __name__ == "__main__":
    unittest.main()
