"""Regression checks for platform routing and the actual workflow gate commands.

Run without importing TransformerEngine or requiring accelerator hardware:
    python -m unittest discover -s .github/scripts/ci -p 'test_*.py'
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

import ci_requests
import generate_matrix as matrix


ROOT = Path(__file__).resolve().parents[3]


class PlatformRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = matrix._load_watch_rules(ROOT / ".github/configs")
        cls.platforms = sorted(cls.rules)

    def select(self, *paths, event="pull_request"):
        return matrix.select_platforms(self.rules, list(paths), event)[0]

    def test_platform_discovery_excludes_metadata(self):
        self.assertTrue(self.platforms, "At least one platform configuration is required")
        self.assertTrue(matrix._NON_PLATFORM_STEMS.isdisjoint(self.platforms))

    def test_each_configured_watch_path_selects_its_platform(self):
        for platform, watch in self.rules.items():
            with self.subTest(platform=platform):
                self.assertTrue(watch, f"Declare paths_watch for {platform}")
                self.assertIn(f".github/configs/{platform}.yml", watch)
            for entry in watch:
                # Probe directory ownership without depending on a particular
                # test-script name or requiring every platform to share a layout.
                path = entry + "routing_probe.py" if entry.endswith("/") else entry
                with self.subTest(platform=platform, path=path):
                    self.assertEqual(self.select(path), [platform])

    def test_platform_addition_and_removal_are_config_driven(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            for platform in ("existing", "new-accelerator"):
                (configs / f"{platform}.yml").write_text(
                    f"paths_watch:\n  - .github/configs/{platform}.yml\n"
                    f"  - tests/plugin/backend/{platform}/\n"
                )
            # Metadata must be ignored before attempting to parse it as a
            # platform config, regardless of its contents.
            for stem in matrix._NON_PLATFORM_STEMS:
                (configs / f"{stem}.yml").write_text("[]\n")
            rules = matrix._load_watch_rules(configs)
            self.assertEqual(set(rules), {"existing", "new-accelerator"})
            self.assertEqual(
                matrix.select_platforms(
                    rules, ["tests/plugin/backend/new-accelerator/test.py"], "pull_request"
                )[0],
                ["new-accelerator"],
            )
            (configs / "existing.yml").unlink()
            rules = matrix._load_watch_rules(configs)
            self.assertEqual(set(rules), {"new-accelerator"})

    def test_metadata_only_directory_is_not_a_platform_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            (configs / "template.yml").write_text("paths_watch: []\n")
            with self.assertRaisesRegex(ValueError, "No platform configurations"):
                matrix._load_watch_rules(configs)

    def test_vendor_aliases(self):
        for vendor, platform in (("npu", "ascend"), ("kunlunxin", "kunlun")):
            self.assertEqual(
                self.select(f"transformer_engine/plugin/core/backends/vendor/{vendor}/ops.py"),
                [platform],
            )

    def test_multiple_platforms_and_documentation(self):
        self.assertEqual(
            self.select(
                "tests/plugin/backend/ascend/test.py",
                "tests/plugin/backend/cuda/test.py",
                "README.md",
            ),
            ["ascend", "cuda"],
        )

    def test_shared_or_unknown_path_in_mixed_pr_runs_all(self):
        for path in (
            "transformer_engine/pytorch/attention.py",
            "transformer_engine/plugin/core/registry.py",
            "transformer_engine/plugin/core/backends/reference/impl/gemm.py",
            "transformer_engine/plugin/core/backends/flagos/flagos.py",
            ".github/workflows/all_tests_common.yml",
            ".github/scripts/ci/generate_matrix.py",
            "tests/test_utils/run_ci_test_group.py",
            "examples/pytorch/new_example.py",
            "setup.py",
            "unknown/new.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    self.select("tests/plugin/backend/cuda/test.py", path), self.platforms
                )

    def test_documentation_only(self):
        self.assertEqual(self.select("README.rst", "docs/conf.py", "examples/README.md"), [])

    def test_missing_diff_and_non_pr_events_run_all(self):
        self.assertEqual(self.select(), self.platforms)
        for event in ("push", "schedule", "workflow_dispatch"):
            self.assertEqual(self.select("README.md", event=event), self.platforms)

    def test_path_boundaries(self):
        self.assertFalse(
            matrix._matches("other/.github/configs/cuda.yml", ".github/configs/cuda.yml")
        )
        self.assertFalse(matrix._matches("tests/cuda_extra/test.py", "tests/cuda/"))

    def test_missing_watch_rules_run_all(self):
        rules = dict(self.rules, cuda=[])
        self.assertEqual(
            matrix.select_platforms(rules, ["tests/plugin/backend/ascend/test.py"], "pull_request")[
                0
            ],
            self.platforms,
        )

    def test_invalid_configs_fail(self):
        for content in ("[]", "paths_watch: cuda", "paths_watch: [null]", "invalid: ["):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                Path(directory, "cuda.yml").write_text(content)
                with self.assertRaises((ValueError, yaml.YAMLError)):
                    matrix._load_watch_rules(Path(directory))

    def test_rename_keeps_source_and_destination(self):
        with patch.object(matrix.subprocess, "run") as run:
            run.return_value.stdout = (
                "tests/plugin/backend/cuda/test.py\0tests/plugin/backend/ascend/test.py\0"
            )
            paths = matrix._changed_files("main")
            self.assertIn("--no-renames", run.call_args.args[0])
            self.assertEqual(self.select(*paths), ["ascend", "cuda"])


class WorkflowGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
        cls.common = yaml.safe_load((ROOT / ".github/workflows/all_tests_common.yml").read_text())[
            "jobs"
        ]

    def test_top_level_gate_status_combinations(self):
        gate = self.ci["ci-complete"]
        # !cancelled() still runs after failures but lets a cancel stop the gate.
        self.assertEqual(gate["if"], "${{ !cancelled() }}")
        self.assertEqual(set(gate["needs"]), {"lint", "plan", "platform-pipeline"})
        # ci-complete now uses flagcicd-gate composite action in steps
        # Gate logic tested in flagcicd_gate.py gate subcommand
        self.assertEqual(len(gate["steps"]), 2)
        self.assertEqual(gate["steps"][0]["uses"], "actions/checkout@v4")
        self.assertEqual(gate["steps"][1]["uses"], "./.github/actions/flagcicd-gate")

    def test_ci_self_tests_run_in_lint(self):
        commands = [step.get("run", "") for step in self.ci["lint"]["steps"]]
        self.assertTrue(
            any(
                "python -m unittest discover -s .github/scripts/ci -p 'test_*.py'" in command
                for command in commands
            )
        )

    def test_no_always_conditions_in_flagcicd_workflows(self):
        # always() cannot be cancelled; flagcicd jobs and steps use !cancelled().
        for name in (
            "ci",
            "daily",
            "ci-requested",
            "release-acceptance",
            "all_tests_common",
            "test_groups_common",
        ):
            text = (ROOT / f".github/workflows/{name}.yml").read_text()
            self.assertNotIn("always()", text, name)

    def test_label_routing_and_event_payload(self):
        rules = matrix._load_watch_rules(ROOT / ".github/configs")
        for labels, expected in (
            (["bug"], sorted(rules)),
            (["platform:cuda", "platform:ascend", "platform:cuda"], ["ascend", "cuda"]),
            (["platform:cuda", "platform:all"], sorted(rules)),
        ):
            self.assertEqual(
                matrix.select_platforms(rules, ["shared.py"], "pull_request", labels)[0], expected
            )
        for label in ("platform:", "platform:unknown"):
            with self.assertRaises(ValueError):
                matrix.select_platforms(rules, ["shared.py"], "pull_request", [label])
        self.assertEqual(
            matrix.select_platforms(rules, ["README.md"], "pull_request", ["platform:cuda"])[0],
            ["cuda"],
        )
        for event in ("push", "schedule", "workflow_dispatch"):
            self.assertEqual(
                matrix.select_platforms(rules, [], event, ["platform:cuda"])[0], sorted(rules)
            )
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory, "event.json")
            event.write_text(json.dumps({"pull_request": {"labels": [{"name": "platform:musa"}]}}))
            result = subprocess.run(
                [
                    "python3",
                    str(ROOT / ".github/scripts/ci/generate_matrix.py"),
                    "--event-name",
                    "pull_request",
                    "--configs-dir",
                    str(ROOT / ".github/configs"),
                    "--changed-files",
                    "shared.py",
                ],
                env=dict(os.environ, GITHUB_EVENT_PATH=str(event)),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(result.stdout.removeprefix("platforms=")), ["musa"])

    def test_comment_authorization_and_selection(self):
        for command in (
            "/ci run cuda,ascend",
            "/test cuda ascend",
            "/test all",
            "/test unknown",
            "/test cuda;id",
            "/test cuda\n/test ascend",
        ):
            for permission in ("admin", "maintain", "write", "triage", "read", "none"):
                for state in ("open", "closed"):
                    event = {
                        "action": "created",
                        "issue": {"number": 42, "pull_request": {"url": "unused"}},
                        "comment": {"body": command, "user": {"login": "writer"}},
                    }
                    api = Mock(
                        side_effect=[
                            {"permission": permission},
                            {"state": state, "head": {"sha": "a" * 40}},
                        ]
                    )
                    if (
                        command in ("/ci run cuda,ascend", "/test cuda ascend", "/test all")
                        and permission in ("admin", "maintain", "write")
                        and state == "open"
                    ):
                        self.assertEqual(
                            ci_requests.resolve(event, api, {"ascend", "cuda"}),
                            (["ascend", "cuda"], "a" * 40),
                        )
                    else:
                        with self.assertRaises((ValueError, PermissionError)):
                            ci_requests.resolve(event, api, {"ascend", "cuda"})
        for event in ({"action": "edited"}, {"action": "created", "issue": {"number": 42}}):
            api = Mock()
            with self.assertRaises(ValueError):
                ci_requests.resolve(event, api, {"cuda"})
            api.assert_not_called()

    def test_retry_limits_and_interrupts(self):
        spec = importlib.util.spec_from_file_location(
            "ci_runner", ROOT / "tests/test_utils/run_ci_test_group.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for codes, retries in (
            ([0], 2),
            ([1, 1, 0], 2),
            ([1, 2, 3], 2),
            ([1], 0),
            ([-15], 2),
            ([130], 2),
            ([137], 2),
        ):
            with patch.object(
                runner, "_load_group", return_value={"name": "probe", "max_retries": retries}
            ), patch.object(runner, "_run_script", side_effect=codes) as execute, patch.object(
                runner.time, "sleep"
            ):
                self.assertEqual(runner.main(), codes[-1])
                self.assertEqual(execute.call_count, len(codes))
        for retries in (-1, 3, True):
            with patch.object(
                runner, "_load_group", return_value={"name": "probe", "max_retries": retries}
            ), self.assertRaises(SystemExit):
                runner.main()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "probe.sh").write_text("if [ ! -f attempt ]; then touch attempt; exit 1; fi\n")
            group = {"name": "real-script", "path": "probe.sh", "max_retries": 1}
            with patch.object(runner, "REPO_ROOT", root), patch.object(
                runner.time, "sleep"
            ), patch.dict(os.environ, TE_TEST_GROUP_JSON=json.dumps(group)):
                self.assertEqual(runner.main(), 0)

    def test_pinned_checkouts_and_requested_gate(self):
        for name in ("all_tests_common", "test_groups_common"):
            jobs = yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())["jobs"]
            for job in jobs.values():
                if job.get("uses", "").startswith("./.github/workflows/"):
                    self.assertEqual(job["with"]["checkout_ref"], "${{ inputs.checkout_ref }}")
                for step in job.get("steps", []):
                    if step.get("uses", "").startswith("actions/checkout@"):
                        self.assertEqual(step["with"]["ref"], "${{ inputs.checkout_ref }}")
        # Every entry plans at a fixed ref, runs that plan's SHA and gates on it.
        for name, stage in (
            ("ci", None),
            ("daily", "nightly"),
            ("ci-requested", "manual"),
            ("release-acceptance", "release"),
        ):
            workflow = yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
            jobs = workflow["jobs"]
            self.assertEqual(workflow["permissions"]["contents"], "read", name)
            pipeline = next(job for job in jobs.values() if "uses" in job)
            self.assertEqual(
                pipeline["with"]["checkout_ref"], "${{ needs.plan.outputs.source_sha }}", name
            )
            self.assertEqual(pipeline["with"]["plan"], "${{ needs.plan.outputs.plan }}", name)
            plan_step = next(
                step
                for step in jobs["plan"]["steps"]
                if step.get("uses") == "./.github/actions/flagcicd-plan"
            )
            if stage:
                self.assertEqual(plan_step["with"]["stage"], stage, name)
            gate = next(
                job
                for job in jobs.values()
                if any(
                    step.get("uses") == "./.github/actions/flagcicd-gate"
                    for step in job.get("steps", [])
                )
            )
            self.assertIn("!cancelled()", gate["if"], name)
        requested = yaml.safe_load((ROOT / ".github/workflows/ci-requested.yml").read_text())
        on = requested.get("on", requested.get(True))
        self.assertEqual(set(on), {"issue_comment", "workflow_dispatch"})
        self.assertEqual(
            requested["jobs"]["plan"]["steps"][0]["with"]["ref"],
            "${{ needs.resolve.outputs.checkout_ref }}",
        )
        plan_inputs = requested["jobs"]["plan"]["steps"][1]["with"]
        self.assertEqual(plan_inputs["platforms"], "${{ needs.resolve.outputs.platforms }}")
        self.assertEqual(plan_inputs["requested-level"], "${{ needs.resolve.outputs.test_level }}")
        self.assertIn("needs.resolve.result != 'skipped'", requested["jobs"]["complete"]["if"])

    def test_dispatch_request_resolution(self):
        available = {"ascend", "cuda"}
        self.assertEqual(
            ci_requests.resolve_dispatch(
                {"platforms": "cuda", "test_level": "L3", "checkout_ref": "v2.1"},
                available,
                "f" * 40,
            ),
            (["cuda"], "v2.1", "L3"),
        )
        self.assertEqual(
            ci_requests.resolve_dispatch({"platforms": "all"}, available, "f" * 40),
            (["ascend", "cuda"], "f" * 40, "auto"),
        )
        for inputs in (
            {"platforms": "unknown"},
            {"platforms": ""},
            {"platforms": "cuda;id"},
            {"platforms": "cuda", "test_level": "L9"},
            {"platforms": "cuda", "checkout_ref": "main; rm"},
        ):
            with self.assertRaises(ValueError, msg=inputs):
                ci_requests.resolve_dispatch(inputs, available, "f" * 40)


if __name__ == "__main__":
    unittest.main()
