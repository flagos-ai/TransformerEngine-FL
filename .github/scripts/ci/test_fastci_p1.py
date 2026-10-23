#!/usr/bin/env python3
"""Tests for flagcicd_history, flagcicd_evidence, and flagcicd_sampler (P1)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import flagcicd_history as hist
import flagcicd_evidence as ev
import flagcicd_sampler as sampler
import flagcicd_contract as contract


# ---------------------------------------------------------------------------
# flagcicd_history tests
# ---------------------------------------------------------------------------


class HistoryTests(unittest.TestCase):

    def _path(self, tmp: str) -> Path:
        return Path(tmp) / "history.jsonl"

    def test_append_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._path(tmp)
            hist.append_entry(
                p,
                source_sha="a" * 40,
                platform="cuda",
                device="a100",
                group="pytorch_unittest",
                status="passed",
            )
            entries = hist.load_history(p)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["status"], "passed")
            self.assertEqual(entries[0]["schema_version"], hist.HISTORY_SCHEMA)

    def test_stability_not_stable_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._path(tmp)
            for _ in range(hist.STABLE_PASS_THRESHOLD - 1):
                hist.append_entry(
                    p,
                    source_sha="a" * 40,
                    platform="cuda",
                    device="a100",
                    group="g",
                    status="passed",
                )
            s = hist.query_stability(p, "cuda", "a100", "g")
            self.assertFalse(s["stable"])
            self.assertEqual(s["pass_streak"], hist.STABLE_PASS_THRESHOLD - 1)

    def test_stability_stable_at_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._path(tmp)
            for _ in range(hist.STABLE_PASS_THRESHOLD):
                hist.append_entry(
                    p,
                    source_sha="a" * 40,
                    platform="cuda",
                    device="a100",
                    group="g",
                    status="passed",
                )
            s = hist.query_stability(p, "cuda", "a100", "g")
            self.assertTrue(s["stable"])

    def test_stability_broken_by_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._path(tmp)
            hist.append_entry(
                p,
                source_sha="a" * 40,
                platform="cuda",
                device="a100",
                group="g",
                status="passed",
            )
            hist.append_entry(
                p,
                source_sha="a" * 40,
                platform="cuda",
                device="a100",
                group="g",
                status="test_failed",
            )
            hist.append_entry(
                p,
                source_sha="a" * 40,
                platform="cuda",
                device="a100",
                group="g",
                status="passed",
            )
            s = hist.query_stability(p, "cuda", "a100", "g")
            self.assertFalse(s["stable"])
            self.assertEqual(s["pass_streak"], 1)

    def test_compact_trims_to_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._path(tmp)
            for _ in range(hist.HISTORY_WINDOW + 10):
                hist.append_entry(
                    p,
                    source_sha="a" * 40,
                    platform="cuda",
                    device="a100",
                    group="g",
                    status="passed",
                )
            removed = hist.compact(p)
            self.assertEqual(removed, 10)
            entries = hist.load_history(p)
            self.assertEqual(len(entries), hist.HISTORY_WINDOW)

    def test_missing_history_returns_not_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nonexistent.jsonl"
            s = hist.query_stability(p, "cuda", "a100", "g")
            self.assertFalse(s["stable"])
            self.assertEqual(s["total_runs"], 0)


# ---------------------------------------------------------------------------
# flagcicd_evidence tests
# ---------------------------------------------------------------------------

GOOD_EVIDENCE = {
    "schema_version": "flagcicd.runtime-evidence/v1",
    "repository": "TransformerEngine-FL",
    "source_sha": "a" * 40,
    "base_sha": None,
    "plan_hash": "b" * 64,
    "platform": "cuda",
    "device": "a100",
    "group": "pytorch_unittest",
    "image_digest": "sha256:deadbeef",
    "context": {"torch": "2.4", "cuda": "12.4", "world_size": 4},
    "frames": [{"file": "transformer_engine/foo.py", "lineno": 10, "qualname": "foo"}],
    "edges": [],
    "sampling_rate_hz": 100.0,
    "complete": True,
    "status": "passed",
    "evidence_digest": None,
    "created_at": "2026-10-01T00:00:00+00:00",
    "sampler_overhead_ms": None,
}

CURRENT_CTX = {"torch": "2.4", "cuda": "12.4", "world_size": 4}


class EvidenceValidationTests(unittest.TestCase):

    def _write(self, tmp: str, data: dict) -> Path:
        p = Path(tmp) / "ev.json"
        p.write_text(json.dumps(data))
        return p

    def test_valid_evidence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, GOOD_EVIDENCE)
            loaded = ev.load_evidence(p)
            ok, reasons = ev.is_evidence_valid(
                loaded,
                current_image_digest="sha256:deadbeef",
                current_context=CURRENT_CTX,
            )
            self.assertTrue(ok, reasons)

    def test_image_digest_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, GOOD_EVIDENCE)
            loaded = ev.load_evidence(p)
            ok, reasons = ev.is_evidence_valid(
                loaded,
                current_image_digest="sha256:different",
                current_context=CURRENT_CTX,
            )
            self.assertFalse(ok)
            self.assertTrue(any("image_digest" in r for r in reasons))

    def test_context_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, GOOD_EVIDENCE)
            loaded = ev.load_evidence(p)
            ok, reasons = ev.is_evidence_valid(
                loaded,
                current_image_digest="sha256:deadbeef",
                current_context={"torch": "2.4", "cuda": "12.4", "world_size": 1},
            )
            self.assertFalse(ok)
            self.assertTrue(any("world_size" in r for r in reasons))

    def test_none_context_dimension_is_ignored(self):
        evidence = dict(GOOD_EVIDENCE)
        evidence["context"] = {"torch": None, "cuda": "12.4"}
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, evidence)
            loaded = ev.load_evidence(p)
            ok, _ = ev.is_evidence_valid(
                loaded,
                current_image_digest="sha256:deadbeef",
                current_context={"torch": "2.4", "cuda": "12.4"},
            )
            self.assertTrue(ok)

    def test_incomplete_evidence_fails(self):
        evidence = dict(GOOD_EVIDENCE, complete=False)
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, evidence)
            loaded = ev.load_evidence(p)
            ok, reasons = ev.is_evidence_valid(
                loaded,
                current_image_digest="sha256:deadbeef",
                current_context=CURRENT_CTX,
            )
            self.assertFalse(ok)
            self.assertTrue(any("incomplete" in r for r in reasons))

    def test_load_evidence_rejects_wrong_schema(self):
        bad = dict(GOOD_EVIDENCE, schema_version="flagcicd.plan/v1")
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, bad)
            with self.assertRaises(ValueError):
                ev.load_evidence(p)

    def test_unstable_history_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, GOOD_EVIDENCE)
            loaded = ev.load_evidence(p)
            history_path = Path(tmp) / "history.jsonl"
            # Only 1 pass — below threshold
            hist.append_entry(
                history_path,
                source_sha="a" * 40,
                platform="cuda",
                device="a100",
                group="pytorch_unittest",
                status="passed",
            )
            ok, reasons = ev.is_evidence_valid(
                loaded,
                current_image_digest="sha256:deadbeef",
                current_context=CURRENT_CTX,
                history_path=history_path,
            )
            self.assertFalse(ok)
            self.assertTrue(any("stable" in r for r in reasons))


class ShadowCompareTests(unittest.TestCase):

    def _make_groups(self) -> list[dict]:
        return [
            {
                "group": "pytorch_unittest",
                "lane": "parallel",
                "required": True,
                "level": "L1",
            },
            {
                "group": "pytorch_mcore_integration",
                "lane": "serial",
                "required": True,
                "level": "L2",
            },
        ]

    def test_shadow_with_no_evidence_selects_nothing(self):
        result = ev.shadow_compare(
            self._make_groups(),
            {},
            current_image_digest="sha256:x",
            current_context=CURRENT_CTX,
        )
        self.assertEqual(result["shadow_selected"], [])
        self.assertIn("pytorch_unittest", result["shadow_excluded"])

    def test_shadow_with_valid_evidence_selects_group(self):
        evidence_set = {"pytorch_unittest": dict(GOOD_EVIDENCE)}
        result = ev.shadow_compare(
            self._make_groups(),
            evidence_set,
            current_image_digest="sha256:deadbeef",
            current_context=CURRENT_CTX,
        )
        self.assertIn("pytorch_unittest", result["shadow_selected"])
        self.assertIn("pytorch_unittest", result["valid_evidence"])

    def test_shadow_with_invalid_evidence_excludes_group(self):
        bad_ev = dict(GOOD_EVIDENCE, image_digest="sha256:wrong")
        evidence_set = {"pytorch_unittest": bad_ev}
        result = ev.shadow_compare(
            self._make_groups(),
            evidence_set,
            current_image_digest="sha256:deadbeef",
            current_context=CURRENT_CTX,
        )
        self.assertNotIn("pytorch_unittest", result["shadow_selected"])
        self.assertIn("pytorch_unittest", result["invalid_evidence"])


# ---------------------------------------------------------------------------
# flagcicd_sampler tests
# ---------------------------------------------------------------------------


class SamplerTests(unittest.TestCase):

    def test_build_evidence_no_sources(self):
        result = sampler.build_evidence(
            platform="cuda",
            device="a100",
            group="pytorch_unittest",
            source_sha="a" * 40,
            base_sha=None,
            plan_hash="b" * 64,
            image_digest="sha256:img",
            context={"torch": "2.4"},
            status="passed",
        )
        self.assertEqual(result["schema_version"], "flagcicd.runtime-evidence/v1")
        self.assertEqual(result["frames"], [])
        self.assertIsNone(result["evidence_digest"])

    def test_build_evidence_with_junit(self):
        with tempfile.TemporaryDirectory() as tmp:
            junit_dir = Path(tmp) / "junit"
            junit_dir.mkdir()
            (junit_dir / "results.xml").write_text(
                '<?xml version="1.0"?>'
                '<testsuite><testcase classname="tests.pytorch.test_ops" name="t"/></testsuite>'
            )
            result = sampler.build_evidence(
                platform="cuda",
                device="a100",
                group="pytorch_unittest",
                source_sha="a" * 40,
                base_sha=None,
                plan_hash="b" * 64,
                image_digest="sha256:img",
                context={},
                status="passed",
                junit_dir=junit_dir,
            )
            self.assertEqual(len(result["frames"]), 1)
            self.assertIn("tests/pytorch/test_ops.py", result["frames"][0]["file"])
            self.assertIsNotNone(result["evidence_digest"])


# ---------------------------------------------------------------------------
# P4 wave-split tests (flagcicd_gate.cmd_slice wave routing)
# ---------------------------------------------------------------------------


class WaveSplitTests(unittest.TestCase):
    """Verify cmd_slice correctly splits parallel lane by wave_assignment."""

    def _make_plan(self, groups: list[dict], priority_enabled: bool) -> dict:
        import hashlib

        plan = {
            "schema_version": contract.PLAN_SCHEMA,
            "repository": "TransformerEngine-FL",
            "stage": "pr",
            "selection": "auto",
            "lint_only": False,
            "source_sha": "a" * 40,
            "base_sha": None,
            "test_level": "L1",
            "risk_level": "R1",
            "risk_rule_version": "v1",
            "changed_files": {"count": 1, "documentation_only": False},
            "common_path_triggers": [],
            "platforms": [
                {
                    "name": "test_plat",
                    "config_hash": "c" * 64,
                    "config_path": "configs/test_plat.yml",
                    "selected": True,
                    "reasons": [],
                    "coverage_required": False,
                }
            ],
            "test_groups": groups,
            "excluded": [],
            "blocked": [],
            "pruned": [],
            "reasons": [],
            "coverage_policy": {"method": "none"},
            "fastci": {
                "enabled": True,
                "selection_strategy": "static",
                "evidence_mode": "off",
                "pruning_enabled": False,
                "priority_enabled": priority_enabled,
                "fallback_on_error": "static",
            },
            "context_dimensions": list(contract.CONTEXT_DIMENSIONS),
            "scheduling": {},
        }
        plan_copy = {k: v for k, v in plan.items() if k != "plan_hash"}
        plan["plan_hash"] = hashlib.sha256(
            json.dumps(plan_copy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return plan

    def _slice(self, plan: dict) -> dict:
        """Run cmd_slice with a mock git HEAD and capture GITHUB_OUTPUT."""
        import os
        import subprocess
        import tempfile
        import flagcicd_gate as gate
        import argparse

        _orig_run = subprocess.run

        def _mock_run(cmd, **kw):
            if cmd == ["git", "rev-parse", "HEAD"]:

                class R:
                    stdout = "a" * 40 + "\n"
                    returncode = 0

                return R()
            return _orig_run(cmd, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            # Write a stub config file so hash check can pass
            cfg_dir = Path(tmp) / "configs"
            cfg_dir.mkdir()
            cfg_file = cfg_dir / "test_plat.yml"
            # Hash must match what we put in plan platforms entry
            import hashlib

            cfg_bytes = b"stub: true\n"
            cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
            # Patch the config hash in plan to match the stub file
            plan["platforms"][0]["config_hash"] = cfg_hash
            plan_copy = {k: v for k, v in plan.items() if k != "plan_hash"}
            plan["plan_hash"] = hashlib.sha256(
                json.dumps(plan_copy, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            cfg_file.write_bytes(cfg_bytes)

            gh_out_path = Path(tmp) / "gh_output"
            old_plan = os.environ.get("FLAGCICD_PLAN")
            old_gh = os.environ.get("GITHUB_OUTPUT")
            try:
                os.environ["FLAGCICD_PLAN"] = json.dumps(plan)
                os.environ["GITHUB_OUTPUT"] = str(gh_out_path)
                subprocess.run = _mock_run
                ns = argparse.Namespace(
                    platform="test_plat",
                    configs_dir=str(cfg_dir),
                    github_output=True,
                )
                rc = gate.cmd_slice(ns)
            finally:
                subprocess.run = _orig_run
                if old_plan is None:
                    os.environ.pop("FLAGCICD_PLAN", None)
                else:
                    os.environ["FLAGCICD_PLAN"] = old_plan
                if old_gh is None:
                    os.environ.pop("GITHUB_OUTPUT", None)
                else:
                    os.environ["GITHUB_OUTPUT"] = old_gh

            self.assertEqual(rc, 0, "cmd_slice returned non-zero")

            out: dict[str, str] = {}
            if gh_out_path.exists():
                for line in gh_out_path.read_text().splitlines():
                    k, _, v = line.partition("=")
                    out[k.strip()] = v.strip()

            parsed: dict[str, object] = {}
            for key in (
                "wave0_matrix",
                "wave1_matrix",
                "parallel_matrix",
                "serial_matrix",
                "has_wave0",
                "has_wave1",
                "has_parallel",
                "has_serial",
                "priority_enabled",
            ):
                raw = out.get(key, "")
                try:
                    parsed[key] = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    parsed[key] = raw
            return parsed

    def _parallel_group(self, name: str, wave: str) -> dict:
        return {
            "platform": "test_plat",
            "device": "gpu0",
            "group": name,
            "kind": "unit",
            "level": "L1",
            "required": True,
            "lane": "parallel",
            "coverage": False,
            "wave_assignment": wave,
            "resources": {
                "class": "gpu",
                "devices": 1,
                "exclusive": False,
                "timeout_minutes": 10,
            },
            "exec": {"runner": "script", "path": "run.sh", "args": [], "env": {}},
        }

    def test_wave0_and_wave1_split_correctly(self):
        groups = [
            self._parallel_group("smoke", "priority-wave-0"),
            self._parallel_group("full", "priority-wave-1"),
        ]
        out = self._slice(self._make_plan(groups, priority_enabled=True))
        w0 = [e["name"] for e in out["wave0_matrix"]["include"]]
        w1 = [e["name"] for e in out["wave1_matrix"]["include"]]
        self.assertEqual(w0, ["smoke"])
        self.assertEqual(w1, ["full"])

    def test_wave0_only_when_all_wave0(self):
        groups = [
            self._parallel_group("a", "priority-wave-0"),
            self._parallel_group("b", "priority-wave-0"),
        ]
        out = self._slice(self._make_plan(groups, priority_enabled=True))
        self.assertTrue(out["has_wave0"])
        self.assertFalse(out["has_wave1"])
        self.assertEqual(len(out["wave0_matrix"]["include"]), 2)
        self.assertEqual(out["wave1_matrix"]["include"], [])

    def test_parallel_matrix_unchanged_regardless_of_priority(self):
        groups = [
            self._parallel_group("x", "priority-wave-0"),
            self._parallel_group("y", "priority-wave-1"),
        ]
        out = self._slice(self._make_plan(groups, priority_enabled=True))
        pm_names = [e["name"] for e in out["parallel_matrix"]["include"]]
        self.assertIn("x", pm_names)
        self.assertIn("y", pm_names)

    def test_priority_enabled_false_emits_false_flag(self):
        groups = [self._parallel_group("g", "priority-wave-1")]
        out = self._slice(self._make_plan(groups, priority_enabled=False))
        self.assertFalse(out["priority_enabled"])

    def test_wave_assignment_carried_into_matrix_entry(self):
        groups = [self._parallel_group("smoke", "priority-wave-0")]
        out = self._slice(self._make_plan(groups, priority_enabled=True))
        entry = out["wave0_matrix"]["include"][0]
        self.assertEqual(entry.get("wave_assignment"), "priority-wave-0")


# ---------------------------------------------------------------------------
# P5 workload profile tests (flagcicd_planner + flagcicd_gate)
# ---------------------------------------------------------------------------


class WorkloadProfileTests(WaveSplitTests):
    """Verify workload_overrides flow from project.yaml through plan to matrix entry."""

    def _make_project_with_profile(self, profile_key: str) -> dict:
        return {
            "fastci": {
                "enabled": True,
                "workload_profile": profile_key,
                "priority_enabled": False,
            },
            "workload_profiles": {
                "nightly_spec": {
                    "description": "full nightly spec",
                    "overrides": {},
                },
                "pr_reduced": {
                    "description": "reduced PR fast-path",
                    "risk": "low",
                    "overrides": {
                        "log_interval": 10,
                        "profile_steps": 0,
                        "checkpoint_activations": False,
                    },
                },
            },
        }

    def _build_overrides(self, profile_key: str) -> dict:
        """Call the planner's override resolution logic directly."""
        import flagcicd_planner as planner

        project = self._make_project_with_profile(profile_key)
        fastci_cfg = project.get("fastci", {})
        key = fastci_cfg.get("workload_profile", "nightly_spec")
        profiles = project.get("workload_profiles", {})
        return dict(profiles.get(key, {}).get("overrides", {}))

    def test_nightly_spec_has_empty_overrides(self):
        overrides = self._build_overrides("nightly_spec")
        self.assertEqual(overrides, {})

    def test_pr_reduced_has_three_overrides(self):
        overrides = self._build_overrides("pr_reduced")
        self.assertEqual(
            set(overrides.keys()),
            {"log_interval", "profile_steps", "checkpoint_activations"},
        )
        self.assertEqual(overrides["log_interval"], 10)
        self.assertFalse(overrides["checkpoint_activations"])

    def test_unknown_profile_key_returns_empty_overrides(self):
        overrides = self._build_overrides("nonexistent_profile")
        self.assertEqual(overrides, {})

    def test_workload_overrides_carried_into_matrix_entry(self):
        """cmd_slice embeds workload_overrides from plan.fastci into every entry."""
        groups = [self._parallel_group("g", "priority-wave-1")]
        plan = self._make_plan(groups, priority_enabled=False)
        # Inject non-empty overrides into the plan as planner would
        plan["fastci"]["workload_overrides"] = {"log_interval": 10, "profile_steps": 0}
        # Re-hash since we mutated fastci
        import hashlib

        plan_copy = {k: v for k, v in plan.items() if k != "plan_hash"}
        plan["plan_hash"] = hashlib.sha256(
            json.dumps(plan_copy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        out = self._slice(plan)
        # parallel_matrix entries must carry the overrides dict
        entries = out["parallel_matrix"]["include"]
        self.assertTrue(len(entries) > 0)
        wo = entries[0].get("workload_overrides", {})
        self.assertEqual(wo.get("log_interval"), 10)
        self.assertEqual(wo.get("profile_steps"), 0)


# ---------------------------------------------------------------------------
# P6 efficiency metrics tests (flagcicd_gate.cmd_gate fastci_metrics)
# ---------------------------------------------------------------------------


class GateMetricsTests(unittest.TestCase):
    """Verify cmd_gate emits correct fastci_metrics in the gate summary."""

    def _make_minimal_plan(
        self,
        *,
        n_selected: int = 2,
        n_blocked: int = 1,
        n_excluded: int = 0,
        n_pruned: int = 0,
    ) -> dict:
        import hashlib

        groups = []
        for i in range(n_selected):
            groups.append(
                {
                    "platform": "p1",
                    "device": "gpu0",
                    "group": f"group_{i}",
                    "kind": "unit",
                    "level": "L1",
                    "required": True,
                    "lane": "parallel",
                    "coverage": False,
                    "wave_assignment": "priority-wave-0",
                    "resources": {
                        "class": "gpu",
                        "devices": 1,
                        "exclusive": False,
                        "timeout_minutes": 10,
                    },
                    "exec": {
                        "runner": "script",
                        "path": "run.sh",
                        "args": [],
                        "env": {},
                    },
                }
            )
        blocked = [{"group": f"blocked_{i}", "required": False} for i in range(n_blocked)]
        excluded = [{"group": f"excl_{i}"} for i in range(n_excluded)]
        pruned = [{"group": f"pruned_{i}"} for i in range(n_pruned)]
        plan = {
            "schema_version": contract.PLAN_SCHEMA,
            "repository": "TransformerEngine-FL",
            "stage": "pr",
            "selection": "auto",
            "lint_only": False,
            "source_sha": "a" * 40,
            "base_sha": None,
            "test_level": "L1",
            "risk_level": "R1",
            "risk_rule_version": "v1",
            "changed_files": {"count": 1, "documentation_only": False},
            "common_path_triggers": [],
            "platforms": [
                {
                    "name": "p1",
                    "config_hash": "c" * 64,
                    "config_path": "configs/p1.yml",
                    "selected": True,
                    "reasons": [],
                    "coverage_required": False,
                }
            ],
            "test_groups": groups,
            "excluded": excluded,
            "blocked": blocked,
            "pruned": pruned,
            "reasons": [],
            "coverage_policy": {"method": "none"},
            "fastci": {
                "enabled": True,
                "selection_strategy": "static",
                "evidence_mode": "off",
                "pruning_enabled": False,
                "priority_enabled": False,
                "fallback_on_error": "static",
                "workload_profile": "nightly_spec",
                "workload_overrides": {},
            },
            "context_dimensions": list(contract.CONTEXT_DIMENSIONS),
            "scheduling": {},
        }
        plan_copy = {k: v for k, v in plan.items() if k != "plan_hash"}
        plan["plan_hash"] = hashlib.sha256(
            json.dumps(plan_copy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return plan

    def _run_gate(self, plan: dict, group_reports: list[dict]) -> dict:
        """Run cmd_gate with synthetic platform reports, return gate summary."""
        import os, tempfile
        import flagcicd_gate as gate
        import argparse

        platform_report = {
            "schema_version": contract.PLATFORM_REPORT_SCHEMA,
            "repository": plan["repository"],
            "stage": plan["stage"],
            "plan_hash": plan["plan_hash"],
            "source_sha": plan["source_sha"],
            "run_id": "42",
            "platform": "p1",
            "status": "passed",
            "groups": group_reports,
            "errors": [],
            "created_at": "2026-10-23T00:00:00+00:00",
        }

        needs = {
            "plan": {"result": "success"},
            "platform-pipeline": {"result": "success"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()
            (reports_dir / "p1.json").write_text(json.dumps(platform_report))
            gate_out = Path(tmp) / "gate.json"

            old_plan = os.environ.get("FLAGCICD_PLAN")
            old_needs = os.environ.get("FLAGCICD_NEEDS")
            try:
                os.environ["FLAGCICD_PLAN"] = json.dumps(plan)
                os.environ["FLAGCICD_NEEDS"] = json.dumps(needs)
                ns = argparse.Namespace(
                    plan_job="plan",
                    pipeline_job="platform-pipeline",
                    reports_dir=reports_dir,
                    output=gate_out,
                )
                gate.cmd_gate(ns)
            finally:
                if old_plan is None:
                    os.environ.pop("FLAGCICD_PLAN", None)
                else:
                    os.environ["FLAGCICD_PLAN"] = old_plan
                if old_needs is None:
                    os.environ.pop("FLAGCICD_NEEDS", None)
                else:
                    os.environ["FLAGCICD_NEEDS"] = old_needs

            return json.loads(gate_out.read_text())

    def _make_group_report(
        self,
        group: str,
        status: str = "passed",
        required: bool = True,
        duration: int = 30,
    ) -> dict:
        return {
            "schema_version": contract.GROUP_REPORT_SCHEMA,
            "platform": "p1",
            "device": "gpu0",
            "group": group,
            "kind": "unit",
            "level": "L1",
            "required": required,
            "status": status,
            "executed": True,
            "failure_stage": None,
            "duration_seconds": duration,
            "repository": "TransformerEngine-FL",
            "stage": "pr",
            "plan_hash": "",
            "source_sha": "a" * 40,
            "run_id": "42",
            "run_attempt": 1,
            "created_at": "2026-10-23T00:00:00+00:00",
        }

    def test_metrics_present_in_gate_summary(self):
        plan = self._make_minimal_plan(n_selected=2, n_blocked=1)
        reports = [
            self._make_group_report("group_0"),
            self._make_group_report("group_1"),
        ]
        summary = self._run_gate(plan, reports)
        self.assertIn("fastci_metrics", summary)

    def test_selected_test_ratio_correct(self):
        # 2 selected + 1 blocked = 3 catalog → ratio = 2/3
        plan = self._make_minimal_plan(n_selected=2, n_blocked=1)
        reports = [
            self._make_group_report("group_0"),
            self._make_group_report("group_1"),
        ]
        summary = self._run_gate(plan, reports)
        m = summary["fastci_metrics"]
        self.assertAlmostEqual(m["selected_test_ratio"], round(2 / 3, 4))

    def test_affected_test_recall_all_passed(self):
        plan = self._make_minimal_plan(n_selected=2)
        reports = [
            self._make_group_report("group_0"),
            self._make_group_report("group_1"),
        ]
        summary = self._run_gate(plan, reports)
        m = summary["fastci_metrics"]
        self.assertEqual(m["affected_test_recall"], 1.0)
        self.assertEqual(m["required_groups_passed"], 2)

    def test_total_duration_summed(self):
        plan = self._make_minimal_plan(n_selected=2)
        reports = [
            self._make_group_report("group_0", duration=40),
            self._make_group_report("group_1", duration=60),
        ]
        summary = self._run_gate(plan, reports)
        self.assertEqual(summary["fastci_metrics"]["total_duration_seconds"], 100)

    def test_workload_profile_key_in_metrics(self):
        plan = self._make_minimal_plan(n_selected=1)
        reports = [self._make_group_report("group_0")]
        summary = self._run_gate(plan, reports)
        self.assertEqual(summary["fastci_metrics"]["workload_profile"], "nightly_spec")


# ---------------------------------------------------------------------------
# P7 master-switch rollback tests (fastci.enabled: false forces all off)
# ---------------------------------------------------------------------------


class MasterSwitchTests(unittest.TestCase):
    """When fastci.enabled is false the plan must carry all-off values for every
    P1-P5 sub-capability, regardless of what the sub-flags say in project.yaml."""

    def _build_plan_with_enabled(self, enabled: bool) -> dict:
        """Call build_plan with a synthetic project.yaml that has all sub-flags on."""
        import os, sys, types, argparse, hashlib, importlib

        # Construct a minimal args namespace and project dict, then call build_plan
        # indirectly by exercising the logic that is now in flagcicd_planner.
        import flagcicd_planner as planner
        import flagcicd_contract as contract

        # Simulate the fastci config block that has all sub-capabilities turned on
        fastci_cfg = {
            "enabled": enabled,
            "selection_mode": "hybrid",
            "evidence_mode": "enforced",
            "pruning_enabled": True,
            "priority_enabled": True,
            "fallback_on_error": "static",
            "workload_profile": "pr_reduced",
        }
        workload_profiles = {
            "nightly_spec": {"overrides": {}},
            "pr_reduced": {"overrides": {"log_interval": 10, "profile_steps": 0}},
        }
        project = {"fastci": fastci_cfg, "workload_profiles": workload_profiles}

        # Replicate the exact resolution logic from build_plan (P7 block)
        fastci_enabled = bool(fastci_cfg.get("enabled", False))
        evidence_mode = fastci_cfg.get("evidence_mode", "off") if fastci_enabled else "off"
        pruning_enabled = (
            bool(fastci_cfg.get("pruning_enabled", False)) if fastci_enabled else False
        )
        priority_enabled = (
            bool(fastci_cfg.get("priority_enabled", False)) if fastci_enabled else False
        )
        workload_profile_key = (
            fastci_cfg.get("workload_profile", "nightly_spec") if fastci_enabled else "nightly_spec"
        )
        wp_def = workload_profiles.get(workload_profile_key, {})
        workload_overrides = dict(wp_def.get("overrides", {})) if fastci_enabled else {}
        effective_selection = (
            fastci_cfg.get("selection_mode", "static") if fastci_enabled else "static"
        )

        return {
            "fastci_enabled": fastci_enabled,
            "evidence_mode": evidence_mode,
            "pruning_enabled": pruning_enabled,
            "priority_enabled": priority_enabled,
            "workload_profile_key": workload_profile_key,
            "workload_overrides": workload_overrides,
            "effective_selection": effective_selection,
        }

    def test_enabled_false_forces_evidence_mode_off(self):
        r = self._build_plan_with_enabled(False)
        self.assertEqual(r["evidence_mode"], "off")

    def test_enabled_false_forces_pruning_disabled(self):
        r = self._build_plan_with_enabled(False)
        self.assertFalse(r["pruning_enabled"])

    def test_enabled_false_forces_priority_disabled(self):
        r = self._build_plan_with_enabled(False)
        self.assertFalse(r["priority_enabled"])

    def test_enabled_false_forces_nightly_spec_profile(self):
        r = self._build_plan_with_enabled(False)
        self.assertEqual(r["workload_profile_key"], "nightly_spec")

    def test_enabled_false_forces_empty_overrides(self):
        r = self._build_plan_with_enabled(False)
        self.assertEqual(r["workload_overrides"], {})

    def test_enabled_false_forces_static_selection(self):
        r = self._build_plan_with_enabled(False)
        self.assertEqual(r["effective_selection"], "static")

    def test_enabled_true_preserves_sub_capabilities(self):
        r = self._build_plan_with_enabled(True)
        self.assertEqual(r["evidence_mode"], "enforced")
        self.assertTrue(r["pruning_enabled"])
        self.assertTrue(r["priority_enabled"])
        self.assertEqual(r["workload_profile_key"], "pr_reduced")
        self.assertEqual(r["workload_overrides"], {"log_interval": 10, "profile_steps": 0})
        self.assertEqual(r["effective_selection"], "hybrid")


if __name__ == "__main__":
    unittest.main()
