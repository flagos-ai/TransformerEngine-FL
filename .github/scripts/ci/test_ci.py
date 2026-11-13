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
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import yaml

import ci_requests
import generate_matrix as matrix


ROOT = Path(__file__).resolve().parents[3]


class PlatformRoutingTests(unittest.TestCase):
    def test_workflow_plan_consumers_download_files_before_loading(self):
        for relative in (
            ".github/workflows/all_tests_common.yml",
            ".github/workflows/test_groups_common.yml",
            ".github/actions/flagcicd-gate/action.yml",
        ):
            config = yaml.safe_load((ROOT / relative).read_text())
            jobs = config.get("jobs", {"composite": config.get("runs", {})})
            for name, job in jobs.items():
                downloaded = False
                for step in job.get("steps", []):
                    if (
                        step.get("uses", "").startswith("actions/download-artifact@")
                        and step.get("with", {}).get("path") == ".flagcicd-plan"
                    ):
                        self.assertEqual(step["with"]["name"], "${{ inputs.plan_artifact }}")
                        downloaded = True
                    env = step.get("env", {})
                    self.assertNotIn("FLAGCICD_PLAN", env, (relative, name))
                    if "FLAGCICD_PLAN_FILE" in env:
                        self.assertTrue(downloaded, (relative, name))
        for name in ("ci", "nightly", "ci-requested", "release-acceptance"):
            workflow = yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
            # toJson(needs) also includes these outputs: never export the full
            # plan, or the final gate would exceed the limit through NEEDS.
            self.assertNotIn("plan", workflow["jobs"]["plan"]["outputs"])
            self.assertIn("plan_artifact", workflow["jobs"]["plan"]["outputs"])

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
                    rules,
                    ["tests/plugin/backend/new-accelerator/test.py"],
                    "pull_request",
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
                    self.select("tests/plugin/backend/cuda/test.py", path),
                    self.platforms,
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

    def test_all_entries_call_the_platform_workflow_directly(self):
        for name, job in (
            ("ci", "platform-pipeline"),
            ("ci-requested", "platform-pipeline"),
            ("release-acceptance", "platform-pipeline"),
            ("nightly", "platforms"),
        ):
            with self.subTest(workflow=name):
                workflow = yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
                self.assertEqual(
                    workflow["jobs"][job]["uses"],
                    "./.github/workflows/all_tests_common.yml",
                )

    def test_no_always_conditions_in_flagcicd_workflows(self):
        # always() cannot be cancelled; flagcicd jobs and steps use !cancelled().
        for name in (
            "ci",
            "nightly",
            "ci-requested",
            "release-acceptance",
            "all_tests_common",
            "test_groups_common",
        ):
            text = (ROOT / f".github/workflows/{name}.yml").read_text()
            self.assertNotIn("always()", text, name)

    def test_platform_workflow_uses_only_static_execution_lanes(self):
        """Every planned lane remains executable after removing optional acceleration."""
        jobs = self.common
        self.assertIn("parallel_groups", jobs)
        self.assertIn("serial_groups", jobs)
        self.assertNotIn("wave0_groups", jobs)
        self.assertNotIn("wave1_groups", jobs)
        for lane in ("parallel", "serial"):
            job = jobs[f"{lane}_groups"]
            self.assertEqual(job["uses"], "./.github/workflows/test_groups_common.yml")
            self.assertEqual(
                job["with"]["matrix"],
                "${{ needs.checkout_and_config.outputs." + lane + "_matrix }}",
            )
        self.assertIn("parallel_groups", jobs["serial_groups"]["needs"])
        self.assertEqual(jobs["serial_groups"]["with"]["max_parallel"], 1)
        for name in ("coverage", "platform_gate"):
            self.assertIn("parallel_groups", jobs[name]["needs"])
            self.assertIn("serial_groups", jobs[name]["needs"])

    def test_label_routing(self):
        rules = matrix._load_watch_rules(ROOT / ".github/configs")
        for labels, expected in (
            (["bug"], sorted(rules)),
            (["platform:cuda", "platform:ascend", "platform:cuda"], ["ascend", "cuda"]),
            (["platform:cuda", "platform:all"], sorted(rules)),
        ):
            self.assertEqual(
                matrix.select_platforms(rules, ["shared.py"], "pull_request", labels)[0],
                expected,
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
                matrix.select_platforms(rules, [], event, ["platform:cuda"])[0],
                sorted(rules),
            )

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
                        and state == "open"
                    ):
                        self.assertEqual(
                            ci_requests.resolve(event, api, {"ascend", "cuda"}),
                            (
                                ["ascend", "cuda"],
                                "a" * 40,
                                permission in ("admin", "maintain", "write"),
                            ),
                        )
                    else:
                        with self.assertRaises((ValueError, PermissionError)):
                            ci_requests.resolve(event, api, {"ascend", "cuda"})
        for event in (
            {"action": "edited"},
            {"action": "created", "issue": {"number": 42}},
        ):
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
                runner,
                "_load_group",
                return_value={"name": "probe", "max_retries": retries},
            ), patch.object(runner, "_run_script", side_effect=codes) as execute, patch.object(
                runner.time, "sleep"
            ):
                self.assertEqual(runner.main(), codes[-1])
                self.assertEqual(execute.call_count, len(codes))
        for retries in (-1, 3, True):
            with patch.object(
                runner,
                "_load_group",
                return_value={"name": "probe", "max_retries": retries},
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

    def test_hygon_debug_requires_importable_inspect_dependency(self):
        # Exercise the real shell launcher without importing accelerator modules
        # or installing packages. The fake interpreter handles every QA command.
        fake_source = """import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ["HYGON_CHECK_ROOT"])
mode = os.environ["HYGON_CHECK_MODE"]
if args and args[0] == "-":
    sys.stdin.read()
    if args[1] == "nvdlfw_inspect.api":
        available = mode == "present" or (mode == "install" and (root / "installed").exists())
        sys.exit(0 if available else 1)
    sys.exit(0)
with (root / "commands").open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[:3] == ["-m", "pip", "install"]:
    if mode == "pip_failure":
        sys.exit(1)
    (root / "installed").touch()
    sys.exit(0)
if args[:2] == ["-m", "pytest"]:
    sys.exit(0)
sys.exit(2)
"""
        for mode, disable_install, expected_status, expected_pytest, expected_pip in (
            ("present", "0", 0, 4, 0),
            ("install", "0", 0, 4, 1),
            ("pip_failure", "0", 1, 0, 1),
            ("import_failure", "0", 1, 0, 1),
            ("missing", "1", 1, 0, 0),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                interpreter = root / "fake-python"
                interpreter.write_text(f"#!{sys.executable}\n{fake_source}")
                interpreter.chmod(0o755)
                result = subprocess.run(
                    [
                        "bash",
                        str(ROOT / "tests/plugin/backend/hygon/run_unit_tests.sh"),
                        "debug",
                    ],
                    env=dict(
                        os.environ,
                        HYGON_CHECK_ROOT=str(root),
                        HYGON_CHECK_MODE=mode,
                        HYGON_SKIP_DEP_INSTALL=disable_install,
                        NVINSPECT_PACKAGE_SPEC="nvinspect-test-package",
                        PYTHON_BIN=str(interpreter),
                        TE_PATH=str(ROOT),
                        TE_LIB_PATH=str(root / "lib"),
                        XML_LOG_DIR=str(root / "logs"),
                        DTK_ENV_SH=str(root / "no-dtk-env.sh"),
                    ),
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, expected_status, result.stdout + result.stderr)
                commands = (
                    [json.loads(line) for line in (root / "commands").read_text().splitlines()]
                    if (root / "commands").exists()
                    else []
                )
                pytest = [command for command in commands if command[:2] == ["-m", "pytest"]]
                pip = [command for command in commands if command[:3] == ["-m", "pip", "install"]]
                self.assertEqual(len(pytest), expected_pytest)
                self.assertEqual(
                    pip,
                    [["-m", "pip", "install", "nvinspect-test-package"]] * expected_pip,
                )
                if expected_status:
                    self.assertIn("Hygon debug tests require importable", result.stderr)

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
            ("nightly", "nightly"),
            ("ci-requested", "manual"),
            ("release-acceptance", "release"),
        ):
            workflow = yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
            jobs = workflow["jobs"]
            self.assertEqual(workflow["permissions"]["contents"], "read", name)
            pipeline = next(job for job in jobs.values() if "uses" in job)
            self.assertEqual(
                pipeline["with"]["checkout_ref"],
                "${{ needs.plan.outputs.source_sha }}",
                name,
            )
            self.assertEqual(
                pipeline["with"]["plan_artifact"],
                "${{ needs.plan.outputs.plan_artifact }}",
                name,
            )
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
        self.assertEqual(
            requested["jobs"]["plan"]["if"], "needs.resolve.outputs.approved == 'true'"
        )
        self.assertIn(
            "needs.resolve.outputs.approved != 'false'",
            requested["jobs"]["complete"]["if"],
        )
        self.assertIn("'/ci approve '", requested["jobs"]["resolve"]["if"])
        self.assertEqual(
            requested["jobs"]["resolve"]["outputs"]["approved"],
            "${{ steps.request.outputs.approved }}",
        )

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

    def test_only_maintainers_can_approve_same_pr_requests(self):
        issue_url = "https://api.github.com/repos/flagos-ai/TransformerEngine-FL/issues/42"
        for permission in ("write", "maintain", "admin", "read", "triage", "none"):
            for original_body, original_issue in (
                ("/ci run cuda", issue_url),
                ("/test all", issue_url),
                ("/ci run cuda", issue_url + "0"),
                ("ordinary comment", issue_url),
                ("/ci approve 12", issue_url),
                ("/ci run unknown", issue_url),
            ):
                with self.subTest(permission=permission, body=original_body, issue=original_issue):
                    event = {
                        "action": "created",
                        "issue": {
                            "number": 42,
                            "pull_request": {"url": "unused"},
                            "url": issue_url,
                        },
                        "comment": {
                            "body": "/ci approve 123",
                            "user": {"login": "approver"},
                        },
                    }
                    api = Mock(
                        side_effect=[
                            {"permission": permission},
                            {"body": original_body, "issue_url": original_issue},
                            {"state": "open", "head": {"sha": "b" * 40}},
                        ]
                    )
                    valid = (
                        permission in {"write", "maintain", "admin"}
                        and original_issue == issue_url
                        and original_body in {"/ci run cuda", "/test all"}
                    )
                    if valid:
                        platforms = (
                            ["cuda"] if original_body.endswith("cuda") else ["ascend", "cuda"]
                        )
                        self.assertEqual(
                            ci_requests.resolve(event, api, {"ascend", "cuda"}),
                            (platforms, "b" * 40, True),
                        )
                        self.assertEqual(api.call_args_list[1].args, ("issues/comments/123",))
                    else:
                        with self.assertRaises((ValueError, PermissionError)):
                            ci_requests.resolve(event, api, {"ascend", "cuda"})
                    if permission not in {"write", "maintain", "admin"}:
                        self.assertEqual(api.call_count, 1)

    def test_public_request_writes_pending_outputs_and_approval_instructions(self):
        for permission in ("none", "write"):
            with self.subTest(permission=permission), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                event = folder / "event.json"
                event.write_text(
                    json.dumps(
                        {
                            "action": "created",
                            "issue": {"number": 42, "pull_request": {"url": "unused"}},
                            "comment": {
                                "id": 123,
                                "body": "/ci run cuda",
                                "user": {"login": "requester"},
                            },
                        }
                    )
                )
                outputs, summary = folder / "outputs", folder / "summary"
                env = {
                    "GITHUB_EVENT_PATH": str(event),
                    "GITHUB_EVENT_NAME": "issue_comment",
                    "GITHUB_REPOSITORY": "flagos-ai/TransformerEngine-FL",
                    "GITHUB_OUTPUT": str(outputs),
                    "GITHUB_STEP_SUMMARY": str(summary),
                }
                responses = [
                    json.dumps({"permission": permission}),
                    json.dumps({"state": "open", "head": {"sha": "a" * 40}}),
                ]
                with patch.dict(os.environ, env), patch.object(
                    ci_requests.subprocess, "check_output", side_effect=responses
                ):
                    ci_requests.main()
                result = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
                self.assertEqual(json.loads(result["platforms"]), ["cuda"])
                self.assertEqual(result["approved"], "true" if permission == "write" else "false")
                self.assertEqual(result["test_level"], "auto")
                if permission == "none":
                    self.assertIn("/ci approve 123", summary.read_text())
                else:
                    self.assertFalse(summary.exists())


@unittest.skipUnless(importlib.util.find_spec("pytest"), "pytest is required for collection tests")
class PytestSelectionTests(unittest.TestCase):
    """Exercise real pytest processes, without importing accelerator libraries."""

    def run_pytest(self, root, arguments, targets=None, *, raw=None):
        env = dict(os.environ)
        env.pop("TE_CI_TEST_TARGETS_JSON", None)
        env.update(
            GITHUB_WORKSPACE=str(root),
            TE_PATH=str(root),
            PYTEST_PLUGINS="ci_pytest_selection",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTHONPATH=os.pathsep.join(
                part
                for part in (
                    str(ROOT / "tests/test_utils"),
                    str(ROOT / ".github/scripts/ci"),
                    env.get("PYTHONPATH", ""),
                )
                if part
            ),
        )
        if targets is not None or raw is not None:
            env["TE_CI_TEST_TARGETS_JSON"] = raw if raw is not None else json.dumps(targets)
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *arguments],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_selection_prevents_excluded_test_import_and_records_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_selected.py").write_text("def test_pass(): assert True\n")
            (root / "test_excluded.py").write_text("raise RuntimeError('must not import')\n")
            result = self.run_pytest(
                root,
                ["test_selected.py", "test_excluded.py", "--junitxml=report.xml"],
                ["test_selected.py"],
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cases = ET.parse(root / "report.xml").findall(".//testcase")
            self.assertEqual(len(cases), 1)
            self.assertEqual(
                [(p.get("name"), p.get("value")) for p in cases[0].findall("properties/property")],
                [
                    ("flagcicd_test_path", "test_selected.py"),
                    ("flagcicd_test_nodeid", "test_selected.py::test_pass"),
                ],
            )
            result = self.run_pytest(
                root, ["test_excluded.py", "--junitxml=empty.xml"], ["test_selected.py"]
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("must not import", result.stdout + result.stderr)
            self.assertEqual(ET.parse(root / "empty.xml").findall(".//testcase"), [])

    def test_directory_and_glob_selection_keep_original_nodeids_and_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "suite/subdirectory"
            nested.mkdir(parents=True)
            (nested / "test_selected.py").write_text(
                "def test_pass(): assert True\ndef test_other(): assert False\n"
            )
            (nested / "test_excluded.py").write_text("raise RuntimeError('must not import')\n")
            result = self.run_pytest(
                root, ["suite", "-k", "test_pass"], ["suite/**/test_selected.py"]
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("1 passed, 1 deselected", result.stdout)
            result = self.run_pytest(
                root,
                ["suite/subdirectory/test_selected.py::test_pass"],
                ["suite/subdirectory/test_selected.py"],
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("1 passed", result.stdout)
            result = self.run_pytest(
                root,
                ["suite/subdirectory/test_selected.py::test_other"],
                ["suite/subdirectory/"],
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("1 failed", result.stdout)

    def test_node_selection_preserves_parameters_class_targets_and_platform_filters(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_mixed.py").write_text(
                "import pytest\n"
                "@pytest.mark.parametrize('value', [1, 2], ids=['small[case]', 'other'])\n"
                "def test_linear(value): assert value > 0\n"
                "class TestNormalization:\n"
                "    def test_norm(self): assert True\n"
                "def test_attention(): assert False\n"
            )
            result = self.run_pytest(
                root,
                ["test_mixed.py", "--junitxml=selected.xml"],
                ["test_mixed.py::test_linear"],
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cases = ET.parse(root / "selected.xml").findall(".//testcase")
            self.assertEqual(len(cases), 2)
            self.assertEqual(
                {
                    p.get("value")
                    for case in cases
                    for p in case.findall("properties/property")
                    if p.get("name") == "flagcicd_test_nodeid"
                },
                {
                    "test_mixed.py::test_linear[small[case]]",
                    "test_mixed.py::test_linear[other]",
                },
            )
            for target, arguments in (
                ("test_mixed.py::TestNorm*", ["test_mixed.py", "-k", "norm"]),
                ("test_mixed.py::test_linear[small[case]]", ["test_mixed.py"]),
                ("test_mixed.py::test_linear", ["test_mixed.py", "-k", "other"]),
            ):
                with self.subTest(target=target, arguments=arguments):
                    result = self.run_pytest(root, arguments, [target])
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("1 passed", result.stdout)

    def test_unselected_node_invocation_has_no_passing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_mixed.py").write_text(
                "def test_linear(): assert False\ndef test_norm(): assert True\n"
            )
            result = self.run_pytest(
                root,
                ["test_mixed.py::test_linear", "--junitxml=empty.xml"],
                ["test_mixed.py::test_norm"],
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(ET.parse(root / "empty.xml").findall(".//testcase"), [])
            result = self.run_pytest(
                root,
                ["test_mixed.py", "-k", "not (test_linear)", "--junitxml=filtered.xml"],
                ["test_mixed.py::test_linear"],
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(ET.parse(root / "filtered.xml").findall(".//testcase"), [])
            result = self.run_pytest(
                root, ["test_mixed.py::missing_node"], ["test_mixed.py::test_norm"]
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_absent_selection_keeps_normal_pytest_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_failure.py").write_text("def test_fail(): assert False\n")
            result = self.run_pytest(root, ["test_failure.py"])
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("1 failed", result.stdout)

    def test_invalid_selection_cannot_disable_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_selected.py").write_text("def test_pass(): assert True\n")
            for raw in (
                "",
                "broken-json",
                "[]",
                "{}",
                '[""]',
                '["../test.py"]',
                '["/tmp/test.py"]',
                '["test.py::"]',
            ):
                with self.subTest(raw=raw):
                    result = self.run_pytest(root, ["test_selected.py"], raw=raw)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("1 passed", result.stdout)

    def test_shared_plugin_preset_discovery_and_environment(self):
        runner = ROOT / "tests/test_utils/run_ci_test_group.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = [
                "tests/plugin/plugin/test_policy.py",
                "tests/plugin/backend/flagos/test_policy.py",
                "tests/plugin/backend/npu/test_backend.py",
                "tests/plugin/accuracy/test_compute_scale.py",
            ]
            for target in targets:
                path = root / target
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("def test_pass(): pass\n")
            env = dict(
                os.environ,
                TE_TEST_GROUP_PRESET="plugin-unit",
                GITHUB_WORKSPACE=str(root),
                TE_FL_PREFER="flagos",
            )
            for platform, expected in (
                ("cuda", [targets[0], targets[1], targets[3]]),
                ("ascend", targets[:3]),
            ):
                env["PLATFORM"] = platform
                result = subprocess.run(
                    [sys.executable, str(runner), "--list-targets"],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(set(result.stdout.splitlines()), set(expected))
            for target, expected in (
                (targets[0], "unset"),
                (targets[1], "flagos"),
                (targets[2], "vendor"),
            ):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(runner),
                        "--target-env",
                        target,
                        sys.executable,
                        "-c",
                        "import os; print(os.environ.get('TE_FL_PREFER', 'unset'))",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)
            # Equal basenames must get distinct reports and independent modules.
            (root / targets[0]).write_text(
                "import os, sys\n"
                "sys.modules['preset_isolation_probe'] = object()\n"
                "def test_pass(): assert 'TE_FL_PREFER' not in os.environ\n"
            )
            (root / targets[1]).write_text(
                "import os, sys\n"
                "def test_pass():\n"
                "    assert 'preset_isolation_probe' not in sys.modules\n"
                "    assert os.environ['TE_FL_PREFER'] == 'flagos'\n"
            )
            env.update(PLATFORM="cuda", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
            env.pop("PYTEST_PLUGINS", None)
            result = subprocess.run(
                [sys.executable, str(runner)], env=env, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reports = list((root / "logs").glob("*.xml"))
            self.assertEqual(len(reports), 3)
            self.assertEqual(sum(len(ET.parse(p).findall(".//testcase")) for p in reports), 3)
            env["GITHUB_WORKSPACE"] = str(root / "missing")
            result = subprocess.run(
                [sys.executable, str(runner), "--list-targets"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_isolated_optional_module_skip_does_not_hide_empty_or_failed_tests(self):
        runner = ROOT / "tests/test_utils/run_ci_test_group.py"
        scenarios = (
            ("import pytest\npytest.importorskip('ci_probe_dependency_that_does_not_exist')\n", 0),
            ("# genuinely empty file\n", 1),
            ("def test_failure(): assert False\n", 1),
            ("raise RuntimeError('collection failed')\n", 1),
            ("import pytest\npytest.skip('module unavailable', allow_module_level=True)\n", 0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_pass.py").write_text("def test_pass(): pass\n")
            (root / "logs").mkdir()
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("TE_CI_")
                and key not in ("TE_TEST_GROUP_PRESET", "PYTEST_PLUGINS", "PYTEST_ADDOPTS")
            }
            env.update(
                GITHUB_WORKSPACE=str(root),
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                TE_TEST_GROUP_JSON=json.dumps(
                    {
                        "name": "probe",
                        "runner": "pytest",
                        "max_retries": 0,
                        "steps": [
                            {
                                "name": "plugin",
                                "targets": ["test_pass.py", "test_probe.py"],
                                "isolate_files": True,
                                "use_platform_runner": False,
                            }
                        ],
                    }
                ),
            )
            for source, expected in scenarios:
                with self.subTest(source=source):
                    (root / "test_probe.py").write_text(source)
                    # A previous attempt's skip report must not mask an empty run.
                    (root / "logs/test_probe.xml").write_text(
                        "<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>"
                    )
                    result = subprocess.run(
                        [sys.executable, str(runner)], env=env, capture_output=True, text=True
                    )
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    if expected == 0:
                        self.assertIn("[SKIP] test_probe.py", result.stdout)
                    else:
                        self.assertNotIn("[SKIP] test_probe.py", result.stdout)

    def test_platform_quarantines_reach_nested_dispatch_without_hiding_other_cases(self):
        import shutil

        functions = (
            "test_pad_matches_native_exactly",
            "test_cross_entropy_forward_and_backward_match_native_exactly",
            "test_grouped_dbias_functions_match_native_exactly",
            "test_permutation_apis_match_native_exactly",
            "test_moe_matches_native_exactly",
            "test_mhc_apis_match_native_exactly",
            "test_dispatch_entry_does_not_shrink_the_compiled_graph",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = root / "tests/plugin"
            (plugin / "plugin").mkdir(parents=True)
            (plugin / "backend/flagos").mkdir(parents=True)
            shutil.copy(ROOT / "tests/plugin/conftest.py", plugin / "conftest.py")
            config_dir = root / ".github/configs"
            config_dir.mkdir(parents=True)
            platform_targets = {
                "hygon": [
                    "tests/plugin/plugin/triton_dispatch_numerics_cases.py::"
                    "test_cross_entropy_forward_and_backward_match_native_exactly"
                ],
                "metax": [
                    "tests/plugin/plugin/triton_dispatch_numerics_cases.py::"
                    "test_cross_entropy_forward_and_backward_match_native_exactly",
                    "tests/plugin/plugin/triton_dispatch_numerics_cases.py::"
                    "test_mhc_apis_match_native_exactly",
                ],
                "iluvatar": [
                    "tests/plugin/plugin/triton_dispatch_numerics_cases.py::"
                    "test_mhc_apis_match_native_exactly"
                ],
                "ppu": [
                    "tests/plugin/plugin/triton_dispatch_numerics_cases.py::"
                    "test_dispatch_entry_does_not_shrink_the_compiled_graph",
                    "tests/plugin/backend/flagos/test_optimizer_numerical.py::"
                    "test_multi_tensor_adam_matches_formula[False-0.0-0-0]",
                ],
            }
            for platform, targets in platform_targets.items():
                (config_dir / f"{platform}.yml").write_text(
                    "ignored_tests:\n" + "".join(f"  - {target!r}\n" for target in targets)
                )

            cases = plugin / "plugin/triton_dispatch_numerics_cases.py"
            cases.write_text("".join(f"def {name}(): pass\n" for name in functions))
            (plugin / "plugin/test_driver.py").write_text(
                "import subprocess, sys\n"
                "def test_driver():\n"
                "    assert subprocess.call([sys.executable, '-m', 'pytest', "
                "'tests/plugin/plugin/triton_dispatch_numerics_cases.py', "
                "'--junitxml=inner.xml', '-q']) == 0\n"
            )
            optimizer = plugin / "backend/flagos/test_optimizer_numerical.py"
            optimizer.write_text(
                "import pytest\n"
                "@pytest.mark.parametrize('case', [0, 1], ids=['False-0.0-0-0', 'False-0.0-0-1'])\n"
                "def test_multi_tensor_adam_matches_formula(case): pass\n"
            )
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("TE_CI_") and key not in ("PYTEST_PLUGINS", "PYTEST_ADDOPTS")
            }
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            expected_ignored = {
                "cuda": set(),
                "hygon": {functions[1]},
                "metax": {functions[1], functions[5]},
                "iluvatar": {functions[5]},
                "ppu": {functions[6]},
            }
            for platform, ignored in expected_ignored.items():
                with self.subTest(platform=platform):
                    env["PLATFORM"] = platform
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            "tests/plugin/",
                            "-q",
                            "-s",
                            "--junitxml=outer.xml",
                        ],
                        cwd=root,
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    inner = {
                        case.get("name")
                        for case in ET.parse(root / "inner.xml").findall(".//testcase")
                    }
                    self.assertEqual(inner, set(functions) - ignored)
                    outer = {
                        case.get("name")
                        for case in ET.parse(root / "outer.xml").findall(".//testcase")
                    }
                    expected = {
                        "test_driver",
                        "test_multi_tensor_adam_matches_formula[False-0.0-0-1]",
                    }
                    if platform != "ppu":
                        expected.add("test_multi_tensor_adam_matches_formula[False-0.0-0-0]")
                    self.assertEqual(outer, expected)
                    if ignored:
                        self.assertIn("[IGNORE][", result.stdout)

    def test_script_runner_passes_selection_to_child_pytest(self):
        spec = importlib.util.spec_from_file_location(
            "ci_selection_runner", ROOT / "tests/test_utils/run_ci_test_group.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_selected.py").write_text("def test_pass(): assert True\n")
            (root / "test_excluded.py").write_text("raise RuntimeError('must not import')\n")
            script = root / "suite.sh"
            script.write_text(
                '"$PYTHON_BIN" -m pytest -q test_excluded.py --junitxml=excluded.xml || exit $?\n'
                '"$PYTHON_BIN" -m pytest -q test_selected.py --junitxml=selected.xml\n'
            )
            group = {
                "name": "selected-script",
                "path": "suite.sh",
                "test_targets": ["test_selected.py"],
                "group_test_targets": {"selected": ["test_selected.py"]},
                "env": {"PYTHON_BIN": sys.executable},
            }
            with patch.object(runner, "REPO_ROOT", root), patch.dict(
                os.environ,
                GITHUB_WORKSPACE=str(root),
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                PYTEST_PLUGINS="",
            ):
                self.assertEqual(runner._run_script(group), 0)
            self.assertEqual(ET.parse(root / "excluded.xml").findall(".//testcase"), [])
            self.assertEqual(len(ET.parse(root / "selected.xml").findall(".//testcase")), 1)


if __name__ == "__main__":
    unittest.main()
