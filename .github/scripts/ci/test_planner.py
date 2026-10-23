#!/usr/bin/env python3
"""Tests for flagcicd_planner.py"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


class PlannerTests(unittest.TestCase):
    def run_planner(self, stage: str, **kwargs) -> dict:
        """Run planner and return parsed plan."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            cmd = [
                sys.executable,
                str(ROOT / ".github/scripts/ci/flagcicd_planner.py"),
                "--stage",
                stage,
                "--source-root",
                str(ROOT),
                "--output",
                output_path,
            ]

            for key, value in kwargs.items():
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Planner failed: {result.stderr}")

            return json.loads(Path(output_path).read_text())
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_nightly_full_matrix(self):
        """Nightly selects all platforms at L3."""
        plan = self.run_planner("nightly")

        self.assertEqual(plan["stage"], "nightly")
        self.assertEqual(plan["test_level"], "L3")
        self.assertEqual(plan["selection"], "all")
        self.assertTrue(len(plan["platforms"]) >= 8)  # All known platforms
        self.assertFalse(plan["lint_only"])

        # Should have test groups for all platforms
        platforms_in_groups = {g["platform"] for g in plan["test_groups"]}
        platforms_in_plan = {p["name"] for p in plan["platforms"]}
        self.assertEqual(platforms_in_groups, platforms_in_plan)

        # Should have preflight for each platform
        preflight_count = sum(1 for g in plan["test_groups"] if g["group"] == "preflight")
        total_devices = sum(len(p["device_types"]) for p in plan["platforms"])
        self.assertEqual(preflight_count, total_devices)

    def test_pr_vendor_only(self):
        """PR with vendor-only changes selects that platform."""
        plan = self.run_planner(
            "pr",
            base_ref="main",
            changed_files=json.dumps(["tests/plugin/backend/cuda/test_fused_attn.py"]),
        )

        self.assertEqual(plan["stage"], "pr")
        self.assertEqual(plan["selection"], "affected")
        self.assertFalse(plan["lint_only"])

        # Should select cuda platform
        platform_names = [p["name"] for p in plan["platforms"]]
        self.assertIn("cuda", platform_names)

    def test_pr_common_path(self):
        """PR with common path triggers all platforms."""
        plan = self.run_planner(
            "pr",
            base_ref="main",
            changed_files=json.dumps(["transformer_engine/pytorch/module/linear.py"]),
        )

        self.assertEqual(plan["stage"], "pr")
        self.assertFalse(plan["lint_only"])

        # Should select all platforms (common path)
        self.assertTrue(len(plan["platforms"]) >= 8)

    def test_risk_order_is_numeric(self):
        """Higher risk rules must escalate even when labels have two digits in future."""
        plan = self.run_planner(
            "branch",
            changed_files=json.dumps(["transformer_engine/pytorch/csrc/extension.cpp"]),
        )
        self.assertEqual(plan["test_level"], "L3")
        self.assertEqual(plan["risk_level"], "R3")

    def test_mcore_integration_change_escalates_to_l2(self):
        """The integration launcher must not be silently omitted at PR L1."""
        plan = self.run_planner(
            "pr",
            base_ref="main",
            changed_files=json.dumps(["qa/L1_pytorch_mcore_integration/test.sh"]),
        )

        self.assertEqual(plan["test_level"], "L2")
        self.assertTrue(
            any(group["group"] == "pytorch_mcore_integration" for group in plan["test_groups"])
        )

    def test_pr_docs_only(self):
        """PR with docs-only changes is lint-only."""
        plan = self.run_planner(
            "pr",
            base_ref="main",
            changed_files=json.dumps(["README.md", "docs/index.rst"]),
        )

        self.assertEqual(plan["stage"], "pr")
        self.assertTrue(plan["lint_only"])
        self.assertEqual(len(plan["platforms"]), 0)
        self.assertEqual(len(plan["test_groups"]), 0)

    def test_manual_requested(self):
        """Manual stage with requested platforms."""
        plan = self.run_planner(
            "manual",
            platforms=json.dumps(["cuda", "ascend"]),
        )

        self.assertEqual(plan["stage"], "manual")
        self.assertEqual(plan["selection"], "requested")

        platform_names = [p["name"] for p in plan["platforms"]]
        self.assertEqual(sorted(platform_names), ["ascend", "cuda"])

    def test_release_blocked_groups(self):
        """Release L4 has blocked groups."""
        plan = self.run_planner("release")

        self.assertEqual(plan["stage"], "release")
        self.assertEqual(plan["test_level"], "L4")

        group_names = {g["group"] for g in plan["test_groups"]}

        self.assertEqual(len(plan["platforms"]), 8)  # All platforms selected
        self.assertNotIn("model_validation", group_names)
        self.assertTrue(plan["blocked"])
        self.assertTrue(any(item["required"] for item in plan["blocked"]))

    def test_nightly_coverage_policy_is_statistical(self):
        plan = self.run_planner("nightly")
        self.assertEqual(plan["coverage_policy"]["method"], "statistical")
        self.assertEqual(plan["coverage_policy"]["window_runs"], 14)
        self.assertEqual(plan["coverage_policy"]["minimum_history"], 8)

    def test_plan_hash_deterministic(self):
        """Plan hash is deterministic for same inputs."""
        plan1 = self.run_planner("nightly")
        plan2 = self.run_planner("nightly")

        # Hashes should match (excluding created_at if present)
        self.assertEqual(plan1["plan_hash"], plan2["plan_hash"])
        self.assertEqual(plan1["source_sha"], plan2["source_sha"])

    def test_resource_resolution(self):
        """Resources are correctly resolved with nproc_per_node."""
        plan = self.run_planner("nightly")

        # Find a multi_device group
        multi_device_groups = [
            g for g in plan["test_groups"] if g["resources"]["class"] == "multi_device"
        ]
        self.assertTrue(len(multi_device_groups) > 0)

        # Check that devices is resolved (not "nproc_per_node" string)
        for group in multi_device_groups:
            devices = group["resources"]["devices"]
            self.assertIsInstance(devices, int)
            self.assertGreater(devices, 0)

    def test_lane_assignment(self):
        """Lanes are correctly assigned."""
        plan = self.run_planner("nightly")

        # Check lane types
        lanes = {g["lane"] for g in plan["test_groups"]}
        self.assertIn("preflight", lanes)
        self.assertIn("parallel", lanes)
        # serial may or may not exist depending on groups

        # Preflight is preflight lane
        preflight_groups = [g for g in plan["test_groups"] if g["group"] == "preflight"]
        for g in preflight_groups:
            self.assertEqual(g["lane"], "preflight")

    def test_vendor_launcher_change_selects_its_group(self):
        """Changing a platform launcher must run the group it launches, not preflight only."""
        project = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
        checked = 0
        for config in sorted((ROOT / ".github/configs").glob("*.yml")):
            if config.stem in {"template", "image_sources", "platforms"}:
                continue
            platform_cfg = yaml.safe_load(config.read_text())
            for group_def in project["test_groups"]:
                for entry in platform_cfg.get(group_def.get("matrix") or "", []) or []:
                    if entry.get("name") != group_def["name"] or entry.get("blocked"):
                        continue
                    plan = self.run_planner(
                        "pr", base_ref="main", changed_files=json.dumps([entry["path"]])
                    )
                    selected = {(g["platform"], g["group"]) for g in plan["test_groups"]}
                    self.assertIn((config.stem, group_def["name"]), selected, entry["path"])
                    self.assertEqual([p["name"] for p in plan["platforms"]], [config.stem])
                    checked += 1
        self.assertGreaterEqual(checked, 16)

    def test_mixed_change_takes_highest_file_risk(self):
        """A low-risk test file must not lower the level of an unmatched core file."""
        plan = self.run_planner(
            "pr",
            base_ref="main",
            changed_files=json.dumps(
                ["tests/pytorch/test_sanity.py", "transformer_engine/pytorch/module/linear.py"]
            ),
        )
        self.assertEqual((plan["test_level"], plan["risk_level"]), ("L2", "R2"))
        self.assertTrue(
            any(group["group"] == "pytorch_mcore_integration" for group in plan["test_groups"])
        )

    def test_routing_reasons_and_excluded_ledger(self):
        plan = self.run_planner(
            "pr",
            base_ref="main",
            requested_level="L2",
            changed_files=json.dumps(["tests/pytorch/test_sanity.py"]),
        )
        self.assertEqual(plan["common_path_triggers"], ["tests/pytorch/test_sanity.py"])
        self.assertTrue(all(p["reasons"] for p in plan["platforms"]))
        self.assertIn("common path: tests/pytorch/test_sanity.py", plan["platforms"][0]["reasons"])
        excluded = {(item["platform"], item["group"], item["reason"]) for item in plan["excluded"]}
        self.assertIn(
            ("cuda", "pytorch_mcore_integration", "no changed path matches paths_watch"), excluded
        )
        self.assertFalse(
            any(group["group"] == "pytorch_mcore_integration" for group in plan["test_groups"])
        )

    def test_every_group_declares_evidence(self):
        plan = self.run_planner("nightly")
        evidence = {g["group"]: g["evidence"] for g in plan["test_groups"]}
        self.assertEqual(evidence["preflight"], "exit_code")
        self.assertEqual(evidence["pytorch_unittest"], "junit")
        self.assertEqual(evidence["pytorch_mcore_integration"], "exit_code")

    def test_matrix_entry_timeout_overrides_catalog(self):
        plan = self.run_planner("nightly")
        timeouts = {
            (g["platform"], g["group"]): g["resources"]["timeout_minutes"]
            for g in plan["test_groups"]
        }
        self.assertEqual(timeouts[("musa", "pytorch_unittest")], 120)
        self.assertEqual(timeouts[("cuda", "pytorch_unittest")], 240)
        self.assertEqual(timeouts[("ascend", "pytorch_unittest")], 90)


if __name__ == "__main__":
    unittest.main()
