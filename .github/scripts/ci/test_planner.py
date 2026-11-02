#!/usr/bin/env python3
"""Tests for flagcicd_planner.py"""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
import flagcicd_planner as planner

ROOT = Path(__file__).resolve().parents[3]
UNIT_SUITES = {
    "pytorch_debug": "debug",
    "pytorch_unit": "unittest",
    "pytorch_distributed": "distributed",
    "pytorch_onnx": "onnx",
}


class PlannerTests(unittest.TestCase):
    def test_disabled_platform_is_not_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "on.yml").write_text(
                "enabled: true\nnproc_per_node: 1\npaths_watch: [vendor/on/]\n"
            )
            (root / "off.yml").write_text("enabled: false\nnproc_per_node: 1\n")
            self.assertEqual(set(planner.load_platforms(root)), {"on"})
            self.assertEqual(set(planner.gm._load_watch_rules(root)), {"on"})
            (root / "off.yml").write_text("enabled: 'false'\n")
            with self.assertRaises(ValueError):
                planner.load_platforms(root)
            with self.assertRaises(ValueError):
                planner.gm._load_watch_rules(root)

    def assert_all_platforms(self, plan):
        expected = set(planner.load_platforms(ROOT / ".github/configs"))
        self.assertTrue(expected)
        self.assertIn("cuda", expected)
        self.assertEqual({entry["name"] for entry in plan["platforms"]}, expected)

    def supported_suites(self, platform):
        return set(UNIT_SUITES) - ({"pytorch_debug"} if platform == "hygon" else set())

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

    def test_project_has_one_functional_test_group_catalog(self):
        """Routing, dependencies and execution mapping live in the same catalog."""
        project = planner.load_project(ROOT / ".flagcicd/project.yaml")
        self.assertNotIn("functional_groups", project)
        self.assertNotIn("functional_routing", project)
        self.assertIn("routing", project)
        catalog = {group["name"]: group for group in project["test_groups"]}
        self.assertTrue(
            {
                "core",
                "plugin_core",
                "reference_backend",
                "attention",
                "distributed",
                "onnx",
                "mcore",
            }
            <= set(catalog)
        )
        self.assertFalse(set(UNIT_SUITES) & set(catalog))
        self.assertEqual(catalog["core"]["depends_on"], [])
        self.assertEqual(catalog["plugin_core"]["depends_on"], ["core"])
        self.assertEqual(catalog["reference_backend"]["depends_on"], ["plugin_core"])
        self.assertEqual(catalog["distributed"]["level"], "L2")
        self.assertEqual(catalog["onnx"]["level"], "L2")
        self.assertEqual(
            catalog["core"]["execution"],
            {"matrix": "unit_test_matrix", "entry": "pytorch_unit"},
        )
        self.assertEqual(project["levels"]["L1"]["required_groups"], ["core"])

    def test_logical_groups_share_an_execution_job_without_losing_evidence(self):
        plan = self.run_planner(
            "pr",
            changed_files=json.dumps(
                ["transformer_engine/plugin/core/backends/reference/reference.py"]
            ),
        )
        self.assertEqual(
            set(plan["group_selection"]["groups"]),
            {"preflight", "core", "plugin_core", "reference_backend"},
        )
        cuda_groups = [group for group in plan["test_groups"] if group["platform"] == "cuda"]
        self.assertEqual(
            {group["group"] for group in cuda_groups},
            {"preflight", "core", "plugin_core", "reference_backend"},
        )
        for group in cuda_groups:
            self.assertNotIn("exec", group)
            if group["group"] != "preflight":
                self.assertEqual(group["execution_group"], "pytorch_unit")
                self.assertEqual(group["evidence"], "junit")
                self.assertTrue(group["test_targets"])
        cuda_jobs = [group for group in plan["execution_groups"] if group["platform"] == "cuda"]
        for device in next(p for p in plan["platforms"] if p["name"] == "cuda")["device_types"]:
            jobs = [group for group in cuda_jobs if group["device"] == device]
            self.assertEqual({group["group"] for group in jobs}, {"preflight", "pytorch_unit"})
            unit = next(group for group in jobs if group["group"] == "pytorch_unit")
            self.assertEqual(set(unit["test_groups"]), {"core", "plugin_core", "reference_backend"})
            self.assertEqual(set(unit["exec"]["group_test_targets"]), set(unit["test_groups"]))

    def test_shared_execution_rejects_incompatible_group_policies(self):
        """Coalescing jobs must not turn an optional group into a required one."""
        project = planner.load_project(ROOT / ".flagcicd/project.yaml")
        platforms = planner.load_platforms(ROOT / ".github/configs")
        changed = ["tests/pytorch/test_fused_optimizer.py"]
        selection = planner.select_test_groups(project, changed, "L2", "pr")
        logical = planner.build_test_groups(
            project, platforms, ["cuda"], "L2", changed, selection=selection
        )
        valid_jobs = planner.build_execution_groups(project, platforms, logical, False)
        unit_jobs = [job for job in valid_jobs if job["group"] == "pytorch_unit"]
        self.assertEqual(len(unit_jobs), 1)
        self.assertEqual(set(unit_jobs[0]["test_groups"]), {"core", "optimizer"})
        for field, value in (
            ("required", False),
            ("evidence", "exit_code"),
            ("coverage", False),
        ):
            with self.subTest(field=field):
                incompatible = copy.deepcopy(logical)
                optimizer = next(group for group in incompatible if group["group"] == "optimizer")
                optimizer[field] = value
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    planner.build_execution_groups(project, platforms, incompatible, False)
        incompatible = copy.deepcopy(logical)
        optimizer = next(group for group in incompatible if group["group"] == "optimizer")
        optimizer["resources"]["devices"] += 1
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            planner.build_execution_groups(project, platforms, incompatible, False)

    def test_shared_execution_rejects_different_platform_adapter(self):
        """The same runtime entry name in another matrix is a different adapter."""
        project = planner.load_project(ROOT / ".flagcicd/project.yaml")
        platforms = planner.load_platforms(ROOT / ".github/configs")
        optimizer = next(group for group in project["test_groups"] if group["name"] == "optimizer")
        optimizer["execution"]["matrix"] = "integration_test_matrix"
        unit = next(
            entry
            for entry in platforms["cuda"]["config"]["unit_test_matrix"]
            if entry["name"] == "pytorch_unit"
        )
        other_adapter = copy.deepcopy(unit)
        other_adapter["path"] = "tests/plugin/backend/cuda/different_command.sh"
        platforms["cuda"]["config"]["integration_test_matrix"].append(other_adapter)
        changed = ["tests/pytorch/test_fused_optimizer.py"]
        selection = planner.select_test_groups(project, changed, "L2", "pr")
        logical = planner.build_test_groups(
            project, platforms, ["cuda"], "L2", changed, selection=selection
        )
        with self.assertRaisesRegex(ValueError, "Incompatible.*mapping"):
            planner.build_execution_groups(project, platforms, logical, False)

    def test_invalid_test_targets_cannot_escape_or_hide_the_selected_scope(self):
        project = planner.load_project(ROOT / ".flagcicd/project.yaml")
        for target in (
            "",
            "/tmp/test_escape.py",
            "../test_escape.py",
            "tests/test_core.py::",
            None,
        ):
            with self.subTest(target=target):
                invalid = copy.deepcopy(project)
                core = next(group for group in invalid["test_groups"] if group["name"] == "core")
                core["test_targets"] = [target]
                with self.assertRaisesRegex(ValueError, "target"):
                    planner.validate_test_groups(invalid)

    def test_configured_node_targets_match_existing_test_definitions(self):
        project = planner.load_project(ROOT / ".flagcicd/project.yaml")
        count = 0
        for group in project["test_groups"]:
            for target in group.get("test_targets", []):
                path, node = planner.contract.split_test_target(target)
                if not node:
                    continue
                tree = ast.parse((ROOT / path).read_text())
                names = []
                for item in tree.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                        names.append(item.name)
                    elif isinstance(item, ast.ClassDef):
                        names.extend(
                            item.name + "::" + method.name
                            for method in item.body
                            if isinstance(method, ast.FunctionDef)
                            and method.name.startswith("test_")
                        )
                self.assertTrue(
                    any(
                        planner.contract.test_target_matches(path + "::" + name, target)
                        for name in names
                    ),
                    target,
                )
                count += 1
        self.assertGreaterEqual(count, 15)

    def test_nightly_full_matrix(self):
        """Nightly selects all platforms at fixed L2."""
        plan = self.run_planner("nightly")

        self.assertEqual(plan["stage"], "nightly")
        self.assertEqual(plan["test_level"], "L2")
        self.assertFalse(any("risk" in key for key in plan))
        self.assertEqual(plan["test_level_rule_version"], "v2")
        self.assertEqual(plan["selection"], "all")
        self.assertTrue(len(plan["platforms"]) >= 8)  # All known platforms
        self.assertFalse(plan["lint_only"])
        self.assertEqual(
            {g["group"] for g in plan["execution_groups"]},
            {"preflight", *UNIT_SUITES, "pytorch_mcore_integration"},
        )
        self.assertTrue(all(g["level"] in ("L0", "L1", "L2") for g in plan["execution_groups"]))
        self.assertEqual(plan["blocked"], [])
        # Splitting reports must preserve each platform's supported suite coverage.
        for platform in plan["platforms"]:
            self.assertEqual(
                set(platform["coverage_groups"]), self.supported_suites(platform["name"])
            )
            for device in platform["device_types"]:
                actual = {
                    g["group"]
                    for g in plan["execution_groups"]
                    if g["platform"] == platform["name"] and g["device"] == device
                }
                self.assertEqual(
                    actual,
                    {
                        "preflight",
                        *self.supported_suites(platform["name"]),
                        "pytorch_mcore_integration",
                    },
                )
        for group in plan["execution_groups"]:
            if group["group"] in UNIT_SUITES:
                self.assertTrue(group["required"])
                self.assertEqual(group["lane"], "serial")
                self.assertEqual(group["resources"]["class"], "multi_device")
                self.assertEqual(group["exec"]["args"], [UNIT_SUITES[group["group"]]])

        # Should have test groups for all platforms
        platforms_in_groups = {g["platform"] for g in plan["execution_groups"]}
        platforms_in_plan = {p["name"] for p in plan["platforms"]}
        self.assertEqual(platforms_in_groups, platforms_in_plan)

        # Should have preflight for each platform
        preflight_count = sum(1 for g in plan["execution_groups"] if g["group"] == "preflight")
        total_devices = sum(len(p["device_types"]) for p in plan["platforms"])
        self.assertEqual(preflight_count, total_devices)

    def test_nightly_level_is_fixed_even_with_l3_request(self):
        plan = self.run_planner("nightly", requested_level="L3")
        self.assertEqual(plan["test_level"], "L2")
        self.assertEqual(plan["coverage_policy"]["method"], "statistical")

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

    def test_native_extension_escalates_to_l3(self):
        """Native-extension paths require L3."""
        plan = self.run_planner(
            "branch",
            changed_files=json.dumps(["transformer_engine/pytorch/csrc/extension.cpp"]),
        )
        self.assertEqual(plan["test_level"], "L3")

    def test_mcore_integration_change_escalates_to_l2(self):
        """The integration launcher must not be silently omitted at PR L1."""
        plan = self.run_planner(
            "pr",
            base_ref="main",
            changed_files=json.dumps(["qa/L1_pytorch_mcore_integration/test.sh"]),
        )

        self.assertEqual(plan["test_level"], "L2")
        self.assertTrue(
            any(group["group"] == "pytorch_mcore_integration" for group in plan["execution_groups"])
        )

    def test_suite_test_paths_include_core_and_escalate_specialized_suites(self):
        """Functional routing adds the core baseline and never drops an L2 suite."""
        cases = {
            "tests/pytorch/test_onnx_export.py": ("pytorch_onnx", "L2"),
            "tests/pytorch/debug/test_config.py": ("pytorch_debug", "L1"),
            "tests/pytorch/distributed/test_numerics.py": ("pytorch_distributed", "L2"),
            "tests/pytorch/test_recipe.py": ("pytorch_unit", "L1"),
        }
        for path, (expected, level) in cases.items():
            with self.subTest(path=path):
                plan = self.run_planner("pr", changed_files=json.dumps([path]))
                self.assertEqual(plan["test_level"], level)
                self.assert_all_platforms(plan)
                for platform in plan["platforms"]:
                    # The full baseline scope must survive partial PR selection.
                    self.assertEqual(
                        set(platform["coverage_groups"]), self.supported_suites(platform["name"])
                    )
                    actual = {
                        g["group"]
                        for g in plan["execution_groups"]
                        if g["platform"] == platform["name"]
                    }
                    self.assertEqual(
                        actual,
                        {"preflight", "pytorch_unit"}
                        | ({expected} & self.supported_suites(platform["name"])),
                    )

    def test_mixed_suite_test_changes_select_the_union(self):
        plan = self.run_planner(
            "pr",
            changed_files=json.dumps(
                ["tests/pytorch/test_recipe.py", "tests/pytorch/test_onnx_export.py"]
            ),
        )
        self.assert_all_platforms(plan)
        for platform in plan["platforms"]:
            actual = {
                g["group"] for g in plan["execution_groups"] if g["platform"] == platform["name"]
            }
            self.assertEqual(actual, {"preflight", "pytorch_unit", "pytorch_onnx"})

    def test_unknown_shared_source_keeps_all_suites_and_integration(self):
        plan = self.run_planner(
            "pr",
            changed_files=json.dumps(["transformer_engine/pytorch/new_shared_component.py"]),
        )
        self.assertEqual(plan["test_level"], "L2")
        self.assert_all_platforms(plan)
        for platform in plan["platforms"]:
            actual = {
                g["group"] for g in plan["execution_groups"] if g["platform"] == platform["name"]
            }
            self.assertEqual(
                actual,
                {
                    "preflight",
                    *self.supported_suites(platform["name"]),
                    "pytorch_mcore_integration",
                },
            )

    def test_unit_launcher_change_selects_all_four_suites_on_its_platform(self):
        plan = self.run_planner(
            "pr",
            changed_files=json.dumps(["tests/plugin/backend/cuda/run_unit_tests.sh"]),
        )
        self.assertEqual(plan["test_level"], "L2")
        self.assertEqual([p["name"] for p in plan["platforms"]], ["cuda"])
        self.assertEqual(
            {g["group"] for g in plan["execution_groups"]},
            {"preflight", *UNIT_SUITES, "pytorch_mcore_integration"},
        )

    def test_functional_modules_narrow_files_and_keep_core_baseline(self):
        plan = self.run_planner(
            "pr",
            changed_files=json.dumps(["transformer_engine/pytorch/optimizers/fused_adam.py"]),
        )
        self.assertEqual(plan["test_level"], "L2")
        self.assertFalse(plan["group_selection"]["full_scope"])
        self.assertEqual(plan["group_selection"]["groups"], ["core", "optimizer", "preflight"])
        for platform in plan["platforms"]:
            self.assertFalse(platform["coverage_full_scope"])
            group = next(
                g
                for g in plan["execution_groups"]
                if g["platform"] == platform["name"] and g["group"] == "pytorch_unit"
            )
            self.assertEqual(
                {g["group"] for g in plan["execution_groups"] if g["platform"] == platform["name"]},
                {"preflight", "pytorch_unit"},
            )
            modules = group["exec"]["group_test_targets"]
            unsupported = {g["group"] for g in platform["unsupported_test_groups"]}
            self.assertEqual("optimizer" in modules, "optimizer" not in unsupported)
            self.assertIn("core", modules)
            self.assertNotIn("tests/", group["exec"]["test_targets"])
            self.assertNotIn("tests/pytorch/test_onnx_export.py", group["exec"]["test_targets"])

    def test_dependencies_and_module_unions(self):
        plan = self.run_planner(
            "pr",
            changed_files=json.dumps(
                [
                    "transformer_engine/plugin/core/backends/reference/reference.py",
                    "tests/pytorch/debug/test_config.py",
                ]
            ),
        )
        self.assertEqual(
            plan["group_selection"]["groups"],
            ["core", "debug", "plugin_core", "preflight", "reference_backend"],
        )
        cuda = next(p for p in plan["platforms"] if p["name"] == "cuda")
        self.assertEqual(cuda["unsupported_test_groups"], [])
        self.assertEqual(
            set(cuda["selected_test_groups"]),
            {"core", "debug", "plugin_core", "preflight", "reference_backend"},
        )
        for platform in plan["platforms"]:
            self.assertEqual(
                {g["group"] for g in plan["execution_groups"] if g["platform"] == platform["name"]},
                {"preflight", "pytorch_unit"}
                | ({"pytorch_debug"} & self.supported_suites(platform["name"])),
            )

    def test_shared_impact_and_unknown_paths_fan_out(self):
        for path in [
            ".github/configs/cuda.yml",
            "transformer_engine/pytorch/fp8.py",
            "transformer_engine/new_unknown.py",
        ]:
            with self.subTest(path=path):
                plan = self.run_planner("pr", changed_files=json.dumps([path]))
                self.assertTrue(plan["group_selection"]["full_scope"])
                self.assertTrue(plan["group_selection"]["reasons"])
                self.assertEqual(
                    {g["group"] for g in plan["execution_groups"]},
                    {"preflight", *UNIT_SUITES, "pytorch_mcore_integration"},
                )
                for group in plan["execution_groups"]:
                    if group["group"] == "pytorch_unit":
                        self.assertEqual(group["exec"]["test_targets"], ["tests/"])

    def test_platform_unsupported_modules_are_not_claimed_as_tested(self):
        plan = self.run_planner("nightly")
        for platform in plan["platforms"]:
            supported = set(platform["selected_test_groups"])
            unsupported = platform["unsupported_test_groups"]
            self.assertTrue(all(entry["reason"] for entry in unsupported))
            self.assertFalse(supported & {entry["group"] for entry in unsupported})
            unit = next(
                g
                for g in plan["execution_groups"]
                if g["platform"] == platform["name"] and g["group"] == "pytorch_unit"
            )
            self.assertTrue(set(unit["exec"]["group_test_targets"]) <= supported)
        hygon = next(p for p in plan["platforms"] if p["name"] == "hygon")
        self.assertIn(
            "optimizer",
            {entry["group"] for entry in hygon["unsupported_test_groups"]},
        )

    def test_coverage_gaps_are_explicit_in_full_and_partial_plans(self):
        gaps = {
            "cuda": {"quantization"},
            "kunlun": {"quantization"},
            "metax": {"quantization", "attention"},
            "musa": {"quantization"},
            "ppu": {"quantization"},
            "hygon": {"debug"},
        }
        for stage, options in (
            ("nightly", {}),
            ("pr", {"changed_files": json.dumps(["tests/pytorch/test_recipe.py"])}),
            ("pr", {"changed_files": json.dumps(["tests/pytorch/debug/test_config.py"])}),
        ):
            plan = self.run_planner(stage, **options)
            for platform in plan["platforms"]:
                name = platform["name"]
                with self.subTest(stage=stage, options=options, platform=name):
                    selected = set(platform["selected_test_groups"])
                    unavailable = {
                        group["group"]: group["reason"]
                        for group in platform["unsupported_test_groups"]
                    }
                    for gap in gaps.get(name, set()) & set(plan["group_selection"]["groups"]):
                        self.assertNotIn(gap, selected)
                        self.assertTrue(unavailable[gap])
                        self.assertTrue(
                            any(
                                group["platform"] == name and group["group"] == gap
                                for group in plan["excluded"]
                            )
                        )
                    if name == "hygon":
                        self.assertNotIn("pytorch_debug", platform["coverage_groups"])
                        self.assertFalse(
                            any(
                                group["platform"] == name and group["group"] == "pytorch_debug"
                                for group in plan["execution_groups"]
                            )
                        )

    def test_bad_module_dependencies_are_rejected(self):
        project = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
        for dependency in ("missing", "core"):
            with self.subTest(dependency=dependency):
                invalid = copy.deepcopy(project)
                next(group for group in invalid["test_groups"] if group["name"] == "core")[
                    "depends_on"
                ] = [dependency]
                with self.assertRaisesRegex(ValueError, "dependency"):
                    planner.validate_test_groups(invalid)
        selection = planner.select_test_groups(project, [], "L2", "nightly")
        with self.assertRaisesRegex(ValueError, "required.*baseline"):
            planner.platform_test_groups(
                project,
                {"unsupported_test_groups": {"core": "not wired"}},
                selection,
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
        self.assertEqual(len(plan["execution_groups"]), 0)
        self.assertEqual(plan["group_selection"]["groups"], [])

    def test_manual_l1_and_distributed_source_routing(self):
        plan = self.run_planner("manual", platforms="cuda", requested_level="L1")
        self.assertEqual(
            {group["group"] for group in plan["execution_groups"]},
            {"preflight", "pytorch_debug", "pytorch_unit"},
        )
        plan = self.run_planner(
            "pr",
            changed_files=json.dumps(["transformer_engine/pytorch/distributed.py"]),
        )
        self.assertEqual(plan["test_level"], "L2")
        self.assertEqual(plan["group_selection"]["groups"], ["core", "distributed", "preflight"])
        self.assertEqual(
            {group["group"] for group in plan["execution_groups"]},
            {"preflight", "pytorch_unit", "pytorch_distributed"},
        )

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

        group_names = {g["group"] for g in plan["execution_groups"]}

        self.assert_all_platforms(plan)  # All platforms selected
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
            g for g in plan["execution_groups"] if g["resources"]["class"] == "multi_device"
        ]
        self.assertTrue(len(multi_device_groups) > 0)

        # Check that devices is resolved (not "nproc_per_node" string)
        for group in multi_device_groups:
            devices = group["resources"]["devices"]
            self.assertIsInstance(devices, int)
            self.assertGreater(devices, 0)

    def test_platform_device_budgets_are_preserved_in_plan(self):
        """Platform-specific counts resolve exactly; single-device groups stay at one."""
        plan = self.run_planner("nightly")
        expected = {}
        for platform in plan["platforms"]:
            config = yaml.safe_load(
                (ROOT / ".github/configs" / f"{platform['name']}.yml").read_text()
            )
            expected[platform["name"]] = config["nproc_per_node"]
            self.assertEqual(platform["nproc_per_node"], config["nproc_per_node"])
        for group in plan["execution_groups"]:
            resources = group["resources"]
            if resources["class"] == "multi_device":
                count = expected[group["platform"]]
                self.assertEqual(resources["devices"], count)
                self.assertEqual(group["lane"], "serial" if count > 1 else "parallel")
            elif resources["class"] == "single_device":
                self.assertEqual(resources["devices"], 1)

    def test_invalid_or_missing_device_budget_is_rejected(self):
        """A missing count must not silently schedule a multi-device job as one card."""
        for value in (None, 0, -1, True, "4", 1.5):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                config = {} if value is None else {"nproc_per_node": value}
                (Path(tmp) / "example.yml").write_text(yaml.safe_dump(config))
                with self.assertRaisesRegex(
                    RuntimeError, "nproc_per_node must be a positive integer"
                ):
                    self.run_planner("nightly", configs_dir=tmp)

    def test_lane_assignment(self):
        """Lanes are correctly assigned."""
        plan = self.run_planner("nightly")

        # Check lane types
        lanes = {g["lane"] for g in plan["execution_groups"]}
        self.assertIn("preflight", lanes)
        self.assertIn("parallel", lanes)
        # serial may or may not exist depending on groups

        # Preflight is preflight lane
        preflight_groups = [g for g in plan["execution_groups"] if g["group"] == "preflight"]
        for g in preflight_groups:
            self.assertEqual(g["lane"], "preflight")

    def test_vendor_launcher_change_selects_its_execution_group(self):
        """Shared launchers fan out on their platform; logical groups share one job."""
        checked = 0
        for config in sorted((ROOT / ".github/configs").glob("*.yml")):
            if config.stem in {"template", "image_sources", "platforms"}:
                continue
            platform_cfg = yaml.safe_load(config.read_text())
            launchers = {}
            for matrix_name in ("unit_test_matrix", "integration_test_matrix"):
                for entry in platform_cfg.get(matrix_name, []) or []:
                    if entry.get("blocked"):
                        continue
                    launchers.setdefault(entry["path"], set()).add(entry["name"])
            for path, suites in launchers.items():
                if config.stem == "hygon":
                    suites = suites - {"pytorch_debug"}
                plan = self.run_planner("pr", changed_files=json.dumps([path]))
                selected = {g["group"] for g in plan["execution_groups"]}
                self.assertTrue(suites <= selected, path)
                self.assertEqual([p["name"] for p in plan["platforms"]], [config.stem])
                keys = [(g["platform"], g["device"], g["group"]) for g in plan["execution_groups"]]
                self.assertEqual(len(keys), len(set(keys)))
                checked += 1
        self.assertGreaterEqual(checked, 16)

    def test_mixed_change_takes_highest_file_level(self):
        """An L1 test file must not lower the level of an unmatched core file."""
        plan = self.run_planner(
            "pr",
            base_ref="main",
            changed_files=json.dumps(
                [
                    "tests/pytorch/test_sanity.py",
                    "transformer_engine/pytorch/new_shared_component.py",
                ]
            ),
        )
        self.assertEqual(plan["test_level"], "L2")
        self.assertTrue(
            any(group["group"] == "pytorch_mcore_integration" for group in plan["execution_groups"])
        )

    def test_routing_reasons_and_excluded_ledger(self):
        plan = self.run_planner(
            "pr",
            base_ref="main",
            requested_level="L2",
            changed_files=json.dumps(["tests/pytorch/test_fused_optimizer.py"]),
        )
        self.assertEqual(plan["common_path_triggers"], ["tests/pytorch/test_fused_optimizer.py"])
        self.assertTrue(all(p["reasons"] for p in plan["platforms"]))
        self.assertIn(
            "common path: tests/pytorch/test_fused_optimizer.py",
            plan["platforms"][0]["reasons"],
        )
        excluded = {(item["platform"], item["group"], item["reason"]) for item in plan["excluded"]}
        self.assertIn(
            (
                "cuda",
                "mcore",
                "test group not selected",
            ),
            excluded,
        )
        self.assertFalse(
            any(group["group"] == "pytorch_mcore_integration" for group in plan["execution_groups"])
        )

    def test_every_group_declares_evidence(self):
        plan = self.run_planner("nightly")
        evidence = {g["group"]: g["evidence"] for g in plan["execution_groups"]}
        self.assertEqual(evidence["preflight"], "exit_code")
        for name in UNIT_SUITES:
            self.assertEqual(evidence[name], "junit")
        self.assertEqual(evidence["pytorch_mcore_integration"], "exit_code")

    def test_matrix_entry_timeout_overrides_catalog(self):
        plan = self.run_planner("nightly")
        timeouts = {
            (g["platform"], g["group"]): g["resources"]["timeout_minutes"]
            for g in plan["execution_groups"]
        }
        for name in UNIT_SUITES:
            self.assertEqual(timeouts[("musa", name)], 120)
            self.assertEqual(timeouts[("cuda", name)], 240)
            self.assertEqual(timeouts[("ascend", name)], 90)

    def test_action_and_diff_work_from_another_directory(self):
        action = yaml.safe_load((ROOT / ".github/actions/flagcicd-plan/action.yml").read_text())
        script = next(s["run"] for s in action["runs"]["steps"] if s.get("id") == "plan")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            repo = folder / "repo"
            repo.mkdir()
            (repo / ".github").mkdir()
            (repo / ".github/configs").symlink_to(
                ROOT / ".github/configs", target_is_directory=True
            )
            (repo / ".flagcicd").symlink_to(ROOT / ".flagcicd", target_is_directory=True)
            env = dict(
                os.environ,
                GIT_AUTHOR_NAME="Test",
                GIT_AUTHOR_EMAIL="test@example.com",
                GIT_COMMITTER_NAME="Test",
                GIT_COMMITTER_EMAIL="test@example.com",
            )

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(repo), *args],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init")
            target = repo / "tests/plugin/backend/cuda/probe.py"
            target.parent.mkdir(parents=True)
            target.write_text("before")
            git("add", ".")
            git("commit", "-m", "base")
            git("update-ref", "refs/remotes/origin/main", "HEAD")
            target.write_text("after")
            git("commit", "-am", "change")
            env.update(
                STAGE="pr",
                SOURCE_ROOT="repo",
                BASE_REF="main",
                CHANGED_FILES="",
                PLATFORMS="",
                REQUESTED_LEVEL="auto",
                LABELS_FROM_EVENT="false",
                GITHUB_ACTION_PATH=str(ROOT / ".github/actions/flagcicd-plan"),
                GITHUB_OUTPUT=str(folder / "outputs"),
                GITHUB_STEP_SUMMARY=str(folder / "summary"),
            )
            result = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=folder,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            plan = json.loads((repo / "plan.json").read_text())
            self.assertEqual([p["name"] for p in plan["platforms"]], ["cuda"])
            outputs = dict(
                line.split("=", 1) for line in (folder / "outputs").read_text().splitlines()
            )
            self.assertEqual(Path(outputs["plan_path"]), repo.resolve() / "plan.json")
            upload = action["runs"]["steps"][-1]["with"]
            self.assertEqual(upload["path"], "${{ steps.plan.outputs.plan_path }}")
            self.assertEqual(upload["if-no-files-found"], "error")
            # Direct invocation must also read the diff under --source-root.
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / ".github/scripts/ci/flagcicd_planner.py"),
                    "--stage",
                    "pr",
                    "--base-ref",
                    "main",
                    "--source-root",
                    str(repo),
                ],
                cwd=folder,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([p["name"] for p in json.loads(result.stdout)["platforms"]], ["cuda"])


if __name__ == "__main__":
    unittest.main()
