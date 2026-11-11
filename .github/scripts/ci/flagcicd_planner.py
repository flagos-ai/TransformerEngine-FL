#!/usr/bin/env python3
"""FlagCICD plan generator."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
import generate_matrix as gm
import flagcicd_contract as contract

LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
EVIDENCE_TYPES = ("junit", "exit_code")
# The plan travels through job outputs and environment variables; keep path
# lists bounded so a large PR cannot exceed the single-variable size limit.
MAX_LISTED_PATHS = 20


def load_project(path: Path) -> dict[str, Any]:
    project = yaml.safe_load(path.read_text())
    if project.get("schema_version") != "flagcicd.project/v1":
        raise ValueError("project.yaml must be flagcicd.project/v1")
    validate_test_groups(project)
    return project


def load_platforms(configs_dir: Path) -> dict[str, Any]:
    """Load platform configs with hashes."""
    platforms = {}
    for path in sorted(configs_dir.glob("*.yml")):
        if path.stem in gm._NON_PLATFORM_STEMS:
            continue
        config_bytes = path.read_bytes()
        config = yaml.safe_load(config_bytes)
        enabled = config.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{path.name}: enabled must be a boolean")
        if not enabled:
            continue
        nproc = config.get("nproc_per_node")
        if isinstance(nproc, bool) or not isinstance(nproc, int) or nproc < 1:
            raise ValueError(f"{path.name}: nproc_per_node must be a positive integer")
        platforms[path.stem] = {
            "config": config,
            "config_hash": hashlib.sha256(config_bytes).hexdigest(),
            "config_path": path.relative_to(configs_dir.parent).as_posix(),
        }
    return platforms


def _capped(paths: list[str], prefix: str) -> list[str]:
    reasons = [f"{prefix}{path}" for path in paths[:MAX_LISTED_PATHS]]
    if len(paths) > MAX_LISTED_PATHS:
        reasons.append(f"{prefix}... {len(paths) - MAX_LISTED_PATHS} more")
    return reasons


def select_platforms(
    stage: str,
    platforms: dict,
    changed: list[str],
    labels: list[str],
    requested: list[str],
) -> tuple[list[str], bool, dict[str, list[str]], list[str], list[str]]:
    """Select platforms.

    Returns: (selected, lint_only, per-platform reasons, common path triggers,
    plan-level routing reasons)
    """
    all_names = sorted(platforms.keys())

    if stage == "pr":
        watch = {n: platforms[n]["config"].get("paths_watch", []) for n in all_names}
        selected, routing = gm.select_platforms(watch, changed, "pull_request", labels)
        lint_only = len(selected) == 0 and len(changed) > 0
        code_paths = [path for path in changed if not gm._is_documentation(path)]
        # Paths no platform owns are the ones that force the full matrix.
        common = [
            path
            for path in code_paths
            if not any(gm._matches(path, entry) for entries in watch.values() for entry in entries)
        ]
        labelled = any(label.startswith("platform:") for label in labels)
        reasons = {}
        for name in selected:
            owned = [
                path
                for path in code_paths
                if any(gm._matches(path, entry) for entry in watch[name])
            ]
            path_reasons = _capped(common, "common path: ") + _capped(owned, "platform path: ")
            reasons[name] = routing[:1] if labelled or not path_reasons else path_reasons
        return (
            selected,
            lint_only,
            reasons,
            common[:MAX_LISTED_PATHS],
            routing[:MAX_LISTED_PATHS],
        )

    if stage in ("branch", "nightly", "release"):
        reason = f"{stage} stage: full matrix"
        return all_names, False, {name: [reason] for name in all_names}, [], [reason]

    if stage == "manual":
        if not requested:
            raise ValueError("manual stage requires requested platforms")
        unknown = set(requested) - set(all_names)
        if unknown:
            raise ValueError(f"Unknown platforms: {', '.join(sorted(unknown))}")
        selected = sorted(set(requested))
        reason = "manual stage: requested platform"
        return selected, False, {name: [reason] for name in selected}, [], [reason]

    raise ValueError(f"Unknown stage: {stage}")


def _checked_order(value: str, order: dict[str, int], kind: str) -> str:
    if value not in order:
        raise ValueError(f"Unknown test-level rule {kind}: {value}")
    return value


def match_level_rules(changed: list[str], level_rules: dict) -> str:
    """Return the highest minimum test level required by changed code paths.

    Each file uses the highest matching rule, or unknown_level when unmatched.
    A lower-level match on one file cannot hide another unmatched file.
    """
    rules = level_rules.get("rules", [])
    unknown_level = _checked_order(level_rules.get("unknown_level", "L2"), LEVEL_ORDER, "level")
    max_level = "L0"
    for path in changed:
        if gm._is_documentation(path):
            continue
        matched = [
            rule
            for rule in rules
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule.get("paths", []))
        ]
        level = max(
            (
                _checked_order(rule.get("minimum_level", "L0"), LEVEL_ORDER, "level")
                for rule in matched
            ),
            key=LEVEL_ORDER.__getitem__,
            default=unknown_level,
        )
        max_level = max(max_level, level, key=LEVEL_ORDER.__getitem__)
    return max_level


def resolve_level(
    stage_def: dict,
    requested: str,
    project: dict,
    changed: list[str],
    labels: list[str],
) -> str:
    """Resolve the test level from stage defaults, path rules and labels."""
    if stage_def.get("fixed_level"):
        return stage_def["default_level"]

    # Start with stage default
    test_level = stage_def["default_level"] if requested == "auto" else requested
    if test_level not in LEVEL_ORDER:
        raise ValueError(f"Unknown requested test level: {requested}")

    # Apply path-based test level rules.
    if changed and "test_level_rules" in project:
        rule_level = match_level_rules(changed, project["test_level_rules"])
        if LEVEL_ORDER[rule_level] > LEVEL_ORDER[test_level]:
            test_level = rule_level

    # Label-based escalation
    for label in labels:
        if label.startswith("level:"):
            requested_level = label.split(":", 1)[1].upper()
            if requested_level not in LEVEL_ORDER:
                raise ValueError(f"Unknown level label: {label}")
            if LEVEL_ORDER[requested_level] > LEVEL_ORDER[test_level]:
                test_level = requested_level

    return test_level


def _path_matches(path: str, pattern: str) -> bool:
    # "/"-terminated entries are directory prefixes; others are exact paths or globs.
    return path.startswith(pattern) if pattern.endswith("/") else fnmatch.fnmatchcase(path, pattern)


def validate_test_groups(project: dict) -> None:
    """Validate the single functional test-group catalog and dependency graph."""
    if "functional_groups" in project:
        raise ValueError("test_groups is the only test-group catalog")
    groups = project.get("test_groups", [])
    catalog = {group["name"]: group for group in groups}
    if not groups or len(catalog) != len(groups):
        raise ValueError("Missing or duplicate test group names")
    visiting, visited = set(), set()

    def visit(name):
        if name not in catalog:
            raise ValueError(f"Unknown test-group dependency: {name}")
        if name in visiting:
            raise ValueError(f"Test-group dependency cycle at {name}")
        if name in visited:
            return
        visiting.add(name)
        group = catalog[name]
        if group.get("level") not in LEVEL_ORDER:
            raise ValueError(f"Invalid test-group level: {name}")
        if group.get("resources", {}).get("class") not in project["resource_classes"]:
            raise ValueError(f"Unknown resource class for test group {name}")
        if not group.get("blocked"):
            _group_evidence(group)
            execution = group.get("execution", {})
            if name != "preflight" and not all(
                isinstance(execution.get(key), str) and execution[key]
                for key in ("matrix", "entry")
            ):
                raise ValueError(f"Test group {name} requires a platform execution mapping")
            if execution.get("entry") == "pytorch_unit" and not group.get("test_targets"):
                raise ValueError(f"Test group {name} requires test_targets")
        targets = group.get("test_targets", [])
        if not isinstance(targets, list):
            raise ValueError(f"Invalid test_targets for {name}")
        for target in targets:
            try:
                contract.split_test_target(target)
            except ValueError:
                raise ValueError(f"Invalid test target for {name}: {target!r}")
        for dependency in group.get("depends_on", []):
            visit(dependency)
            if LEVEL_ORDER[catalog[dependency]["level"]] > LEVEL_ORDER[group["level"]]:
                raise ValueError(f"Higher-level test-group dependency: {name} -> {dependency}")
        visiting.remove(name)
        visited.add(name)

    for name in catalog:
        visit(name)
    for definition in project.get("levels", {}).values():
        for name in definition.get("required_groups", []):
            if name not in catalog:
                raise ValueError(f"Unknown required test group: {name}")


def select_test_groups(project: dict, changed: list[str], test_level: str, stage: str) -> dict:
    """Match test groups, add required baselines and recursively include dependencies."""
    catalog = {group["name"]: group for group in project["test_groups"]}
    eligible = {
        name
        for name, group in catalog.items()
        if LEVEL_ORDER[group["level"]] <= LEVEL_ORDER[test_level]
    }
    code = [path for path in changed if not gm._is_documentation(path)]
    routing = project.get("routing", {})
    broad = routing.get("shared_paths", []) + routing.get("impact_paths", [])
    direct, unmapped, reasons = set(), [], []
    for path in code:
        matches = {
            name
            for name, group in catalog.items()
            if any(_path_matches(path, pattern) for pattern in group.get("paths", []))
        }
        direct.update(matches)
        if not matches and not any(_path_matches(path, pattern) for pattern in broad):
            unmapped.append(path)
    shared = [path for path in code if any(_path_matches(path, pattern) for pattern in broad)]
    full = stage != "pr" or not code or bool(shared or unmapped)
    if stage != "pr" or not code:
        reasons.append(f"{stage}: complete in-level test scope")
    reasons.extend(_capped(shared, "shared/impact path: "))
    reasons.extend(_capped(unmapped, "unmapped path: "))
    required = {
        name
        for level, definition in project.get("levels", {}).items()
        if LEVEL_ORDER[level] <= LEVEL_ORDER[test_level]
        for name in definition.get("required_groups", [])
    }
    selected, visited = set(eligible if full else direct) | required, set()

    def add_dependencies(name):
        if name not in eligible:
            raise ValueError(f"Test group {name} requires level {catalog[name]['level']}")
        if name in visited:
            return
        visited.add(name)
        for dependency in catalog[name].get("depends_on", []):
            selected.add(dependency)
            add_dependencies(dependency)

    for name in sorted(selected):
        add_dependencies(name)
    return {
        "groups": sorted(selected),
        "direct_groups": sorted(direct),
        "required_groups": sorted(required),
        "full_scope": full,
        "unmapped_paths": unmapped[:MAX_LISTED_PATHS],
        "reasons": reasons or ["affected test groups and dependencies"],
    }


def platform_test_groups(project: dict, config: dict, selection: dict) -> tuple[list, list]:
    """Record unavailable platform test groups and propagate dependency gaps."""
    if "unsupported_test_groups" in config:
        raise ValueError("unsupported_test_groups was renamed to ignored_test_groups")
    ignored = config.get("ignored_test_groups", {})
    catalog = {group["name"]: group for group in project["test_groups"]}
    if not isinstance(ignored, dict) or set(ignored) - catalog.keys():
        raise ValueError("Invalid ignored_test_groups in platform config")
    if any(not isinstance(reason, str) or not reason.strip() for reason in ignored.values()):
        raise ValueError("Every ignored test group requires a reason")
    unavailable = dict(ignored)
    while True:
        previous = len(unavailable)
        for name in selection["groups"]:
            missing = set(catalog[name].get("depends_on", [])) & unavailable.keys()
            if missing and name not in unavailable:
                unavailable[name] = (
                    f"unavailable test-group dependencies: {', '.join(sorted(missing))}"
                )
        if len(unavailable) == previous:
            break
    missing_required = set(selection["required_groups"]) & unavailable.keys()
    if missing_required:
        raise ValueError(f"Platform cannot run required test baseline: {sorted(missing_required)}")
    return (
        [name for name in selection["groups"] if name not in unavailable],
        [
            {"group": name, "reason": unavailable[name]}
            for name in selection["groups"]
            if name in unavailable
        ],
    )


def _group_evidence(group_def: dict) -> str:
    evidence = group_def.get("evidence")
    if evidence not in EVIDENCE_TYPES:
        raise ValueError(
            f"test group {group_def['name']} must declare evidence as one of"
            f" {', '.join(EVIDENCE_TYPES)}"
        )
    return evidence


def _execution_entry(group_def: dict, platform: str, config: dict) -> dict | None:
    """Resolve the platform adapter named by the logical test group."""
    if group_def["name"] == "preflight":
        return {}
    mapping = group_def.get("execution", {})
    entries = [
        entry
        for entry in config.get(mapping.get("matrix", ""), [])
        if entry.get("name") == mapping.get("entry")
    ]
    if len(entries) > 1:
        raise ValueError(
            f"{platform} has duplicate {mapping.get('entry')} in {mapping.get('matrix')}"
        )
    return dict(entries[0]) if entries else None


def build_test_groups(
    project: dict,
    platforms: dict,
    selected_platforms: list[str],
    test_level: str,
    changed: list[str],
    blocked: list[dict[str, Any]] | None = None,
    ignored: list[dict[str, Any]] | None = None,
    selection: dict | None = None,
) -> list[dict]:
    """Resolve selected logical groups, retaining platform exclusions and blockers."""
    selection = selection or select_test_groups(
        project, changed, test_level, "pr" if changed else "branch"
    )
    groups = []
    for platform in selected_platforms:
        config = platforms[platform]["config"]
        supported, unavailable = platform_test_groups(project, config, selection)
        unavailable = {entry["group"]: entry["reason"] for entry in unavailable}
        for definition in project["test_groups"]:
            name = definition["name"]
            if LEVEL_ORDER[definition["level"]] > LEVEL_ORDER[test_level]:
                continue
            reason = unavailable.get(name)
            if name not in selection["groups"]:
                reason = "test group not selected"
            adapter = None
            blocker = definition.get("blocked")
            if not reason and not blocker:
                adapter = _execution_entry(definition, platform, config)
                if adapter is None:
                    mapping = definition["execution"]
                    if definition.get("required"):
                        raise ValueError(
                            f"{platform} missing required group {name}: {mapping['entry']} in"
                            f" {mapping['matrix']}"
                        )
                    reason = f"no {mapping['matrix']} entry {mapping['entry']}"
                else:
                    blocker = adapter.get("blocked")
            for device in config.get("device_types", []):
                ledger = {
                    "platform": platform,
                    "device": device,
                    "group": name,
                    "kind": definition["kind"],
                    "level": definition["level"],
                    "required": definition.get("required", False),
                }
                if reason:
                    if ignored is not None:
                        ignored.append({**ledger, "reason": reason})
                    continue
                if blocker:
                    if blocked is not None:
                        blocked.append({**ledger, "reason": str(blocker)})
                    continue
                class_name = definition["resources"]["class"]
                class_def = project["resource_classes"][class_name]
                devices = (
                    config["nproc_per_node"]
                    if class_def["devices"] == "nproc_per_node"
                    else class_def["devices"]
                )
                timeout = adapter.get(
                    "timeout_minutes",
                    definition["resources"].get("timeout_minutes", 90),
                )
                if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
                    raise ValueError(
                        f"{platform} {name} timeout_minutes must be a positive integer"
                    )
                groups.append(
                    {
                        **ledger,
                        "execution_group": definition.get("execution", {}).get("entry", name),
                        "lane": (
                            "preflight"
                            if name == "preflight"
                            else ("serial" if class_def["exclusive"] or devices > 1 else "parallel")
                        ),
                        "evidence": _group_evidence(definition),
                        "resources": {
                            "class": class_name,
                            "devices": devices,
                            "exclusive": class_def["exclusive"],
                            "timeout_minutes": timeout,
                        },
                        "coverage": bool(
                            definition.get("coverage") and config.get("coverage", {}).get("enabled")
                        ),
                        "test_targets": definition.get("test_targets", []),
                    }
                )
    return groups


def _expand_target_candidates(root: Path, target: str) -> list[str]:
    """Expand a directory or file glob target to concrete repository-relative targets."""
    target_path, target_node = contract.split_test_target(target)
    path = root / target_path
    if target_path.endswith("/") and path.is_dir():
        files = [
            candidate.relative_to(root).as_posix()
            for candidate in sorted(path.rglob("test_*.py"))
            if candidate.is_file()
        ]
        if files:
            return [f"{file}::{target_node}" if target_node else file for file in files]
    matched = [
        candidate.relative_to(root).as_posix()
        for candidate in sorted(root.glob(target_path))
        if candidate.is_file()
    ]
    if matched:
        return [f"{file}::{target_node}" if target_node else file for file in matched]
    return [target]


def _target_suppressed_by_ignore(candidate: str, ignored_target: str) -> bool:
    """Return True when ignored_target suppresses candidate during pytest collection."""
    cand_path, cand_node = contract.split_test_target(candidate)
    ig_path, ig_node = contract.split_test_target(ignored_target)
    path_matches = (
        cand_path.startswith(ig_path)
        if ig_path.endswith("/")
        else fnmatch.fnmatchcase(cand_path, ig_path)
    )
    if not path_matches:
        return False
    if not ig_node:
        return True
    if not cand_node:
        return False
    return contract.test_target_matches(candidate, ignored_target)


def build_execution_groups(
    project: dict,
    platforms: dict,
    logical_groups: list[dict],
    full_scope: bool,
    source_root: Path | None = None,
) -> list[dict]:
    """Coalesce selected groups sharing one platform adapter into one job."""
    definitions = {group["name"]: group for group in project["test_groups"]}
    executions = {}
    mappings = {}
    for logical in logical_groups:
        key = (logical["platform"], logical["device"], logical["execution_group"])
        definition = definitions[logical["group"]]
        adapter = _execution_entry(
            definition, logical["platform"], platforms[logical["platform"]]["config"]
        )
        for field in ("name", "timeout_minutes", "blocked"):
            adapter.pop(field, None)
        if key not in executions:
            mappings[key] = definition.get("execution")
            executions[key] = {
                **{
                    field: logical[field]
                    for field in (
                        "platform",
                        "device",
                        "kind",
                        "level",
                        "required",
                        "lane",
                        "evidence",
                        "coverage",
                    )
                },
                "group": logical["execution_group"],
                "resources": dict(logical["resources"]),
                "test_groups": [],
                "exec": adapter if logical["group"] != "preflight" else None,
            }
        execution = executions[key]
        if mappings[key] != definition.get("execution") or (execution["exec"] or {}) != adapter:
            raise ValueError(f"Incompatible execution mapping for {key}")
        if any(
            execution[field] != logical[field]
            for field in ("kind", "evidence", "lane", "required", "coverage")
        ) or any(
            execution["resources"][field] != logical["resources"][field]
            for field in ("class", "devices", "exclusive")
        ):
            raise ValueError(f"Incompatible test groups share execution {key}")
        execution["test_groups"].append(logical["group"])
        execution["required"] |= logical["required"]
        execution["coverage"] |= logical["coverage"]
        execution["level"] = max(execution["level"], logical["level"], key=LEVEL_ORDER.__getitem__)
        execution["resources"]["timeout_minutes"] = max(
            execution["resources"]["timeout_minutes"],
            logical["resources"]["timeout_minutes"],
        )
    for execution in executions.values():
        targets = {
            name: definitions[name]["test_targets"]
            for name in execution["test_groups"]
            if definitions[name].get("test_targets")
        }
        if targets:
            if len(targets) != len(execution["test_groups"]):
                raise ValueError(
                    f"Filtered and unfiltered test groups share execution {execution['group']}"
                )
            execution["exec"]["group_test_targets"] = targets
            execution["exec"]["test_targets"] = (
                ["tests/"]
                if full_scope
                else sorted({target for entries in targets.values() for target in entries})
            )
    catalog_targets = {
        target for group in definitions.values() for target in group.get("test_targets", [])
    }
    root = source_root if source_root is not None else Path(__file__).resolve().parents[3]
    for execution in executions.values():
        if not execution["exec"]:
            continue
        # Full directory discovery must not re-enable a platform's ignored
        # groups through another group's overlapping test_targets.
        _, ignored_groups = platform_test_groups(
            project,
            platforms[execution["platform"]]["config"],
            {"groups": list(definitions), "required_groups": []},
        )
        ignored_targets = {
            target
            for group in ignored_groups
            for target in definitions[group["group"]].get("test_targets", [])
        }
        if execution["platform"] != "ascend" and "tests/plugin/backend/npu/" in catalog_targets:
            ignored_targets.add("tests/plugin/backend/npu/")
        if execution["platform"] == "ascend":
            # Keep the plan aligned with the standalone plugin-unit preset:
            # NPU compute-scale CUDA-semantic parity is not wired in this CI.
            ignored_targets.add("tests/plugin/accuracy/test_compute_scale.py")
        execution["exec"]["ignored_test_targets"] = sorted(ignored_targets)
        group_targets = execution["exec"].get("group_test_targets", {})
        if group_targets and ignored_targets:
            overlaps = {}
            for group_name, entries in group_targets.items():
                suppressed = sorted(
                    {
                        candidate
                        for entry in entries
                        for candidate in _expand_target_candidates(root, entry)
                        if any(
                            _target_suppressed_by_ignore(candidate, ignored_target)
                            for ignored_target in execution["exec"]["ignored_test_targets"]
                        )
                    }
                )
                if suppressed:
                    overlaps[group_name] = suppressed
            if overlaps:
                execution["exec"]["ignored_group_test_targets"] = overlaps
    return list(executions.values())


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build the plan."""
    source_root = Path(args.source_root).resolve()
    project_path = source_root / args.project
    configs_dir = source_root / args.configs_dir

    project = load_project(project_path)
    platforms = load_platforms(configs_dir)

    # Resolve stage
    stage_name = args.stage
    if stage_name not in project["stages"]:
        raise ValueError(f"Unknown stage: {stage_name}")
    stage_def = project["stages"][stage_name]

    # Get source SHA
    result = subprocess.run(
        ["git", "rev-parse", args.source_sha or "HEAD"],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=True,
    )
    source_sha = result.stdout.strip()

    # Get base SHA for PR
    base_sha = None
    if stage_name == "pr" and args.base_ref:
        result = subprocess.run(
            ["git", "merge-base", f"origin/{args.base_ref}", "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            base_sha = result.stdout.strip()

    # Get changed files
    changed = []
    if args.changed_files:
        changed = json.loads(args.changed_files)
    elif stage_name == "pr" and args.base_ref:
        try:
            changed = gm._changed_files(args.base_ref, cwd=source_root)
        except subprocess.CalledProcessError:
            pass

    # Get labels
    labels = []
    if args.labels_from_event and stage_name == "pr":
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path and Path(event_path).exists():
            try:
                event = json.loads(Path(event_path).read_text())
                labels = [l["name"] for l in event.get("pull_request", {}).get("labels", [])]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                pass

    # Parse requested platforms
    requested_platforms = []
    if args.platforms:
        requested_platforms = (
            json.loads(args.platforms) if args.platforms.startswith("[") else [args.platforms]
        )

    # Select platforms
    selected, lint_only, platform_reasons, common_triggers, routing = select_platforms(
        stage_name, platforms, changed, labels, requested_platforms
    )

    # Resolve test level
    test_level = resolve_level(stage_def, args.requested_level, project, changed, labels)
    group_selection = select_test_groups(project, changed, test_level, stage_name)
    if lint_only:
        group_selection.update(
            groups=[],
            direct_groups=[],
            required_groups=[],
            full_scope=False,
            reasons=["documentation-only: no hardware tests"],
        )
    blocked: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    test_groups = (
        build_test_groups(
            project,
            platforms,
            selected,
            test_level,
            changed,
            blocked,
            ignored,
            group_selection,
        )
        if not lint_only
        else []
    )
    execution_groups = build_execution_groups(
        project,
        platforms,
        test_groups,
        group_selection["full_scope"],
        source_root=source_root,
    )
    platform_entries = []
    for pname in selected:
        p = platforms[pname]
        _, ignored_groups = platform_test_groups(project, p["config"], group_selection)
        # Coverage completeness is measured against the platform's supported
        # catalog, independent of a PR's partial group selection.
        coverage_supported, _ = platform_test_groups(
            project,
            p["config"],
            {
                "groups": [group["name"] for group in project["test_groups"]],
                "required_groups": [],
            },
        )
        platform_entries.append(
            {
                "name": pname,
                "config": p["config_path"],
                "config_hash": p["config_hash"],
                "device_types": p["config"].get("device_types", []),
                "nproc_per_node": p["config"]["nproc_per_node"],
                "coverage_enabled": bool(p["config"].get("coverage", {}).get("enabled", False)),
                "coverage_required": bool(p["config"].get("coverage", {}).get("required", False)),
                "coverage_full_scope": group_selection["full_scope"],
                "selected_test_groups": sorted(
                    group["group"]
                    for group in test_groups
                    if group["platform"] == pname
                    and group["device"] == p["config"].get("device_types", [None])[0]
                ),
                "ignored_test_groups": ignored_groups,
                "coverage_groups": sorted(
                    {
                        group["execution"]["entry"]
                        for group in project["test_groups"]
                        if group.get("coverage")
                        and group["name"] in coverage_supported
                        and not group.get("blocked")
                        and p["config"].get("coverage", {}).get("enabled", False)
                        and any(
                            entry.get("name") == group["execution"]["entry"]
                            and not entry.get("blocked")
                            for entry in p["config"].get(group["execution"]["matrix"], [])
                        )
                    }
                ),
                "reasons": platform_reasons[pname],
            }
        )

    if stage_name == "pr":
        coverage_policy = {"method": "absolute"}
    elif stage_name == "nightly":
        coverage_policy = {
            "method": "statistical",
            "sigma": 3.0,
            "window_runs": 14,
            "minimum_history": 8,
        }
    else:
        coverage_policy = {"method": "none"}

    # Build plan
    plan = {
        "schema_version": contract.PLAN_SCHEMA,
        "repository": project["repository"],
        "stage": stage_name,
        "selection": stage_def["selection"],
        "lint_only": lint_only,
        "source_sha": source_sha,
        "base_sha": base_sha,
        "test_level": test_level,
        "test_level_rule_version": project.get("test_level_rules", {}).get("version", "v1"),
        "changed_files": {
            "count": len(changed),
            "documentation_only": (
                all(gm._is_documentation(f) for f in changed) if changed else False
            ),
        },
        "common_path_triggers": common_triggers,
        "platforms": platform_entries,
        "group_selection": group_selection,
        "test_groups": test_groups,
        "execution_groups": execution_groups,
        "ignored": ignored,
        "blocked": blocked,
        "reasons": [
            f"{stage_name} stage with {len(selected)} platforms, level {test_level}",
            *routing,
        ],
        "coverage_policy": coverage_policy,
    }

    # Compute plan hash
    plan["plan_hash"] = contract.compute_plan_hash(plan)

    return plan


def write_summary(path: Path, plan: dict) -> None:
    """Render routing and test scope together, using the TE plan contract."""

    def escape(value):
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("|", "&#124;")
            .replace("\n", " ")
        )

    platforms = plan["platforms"]
    executions = plan["execution_groups"]
    selection = plan["group_selection"]
    gaps = [
        f"{p['name']}/{gap['group']}: {gap['reason']}"
        for p in platforms
        for gap in p["ignored_test_groups"]
    ] + [
        f"{job['platform']}/{group} ignored targets: {', '.join(targets)}"
        for job in executions
        for group, targets in (job.get("exec") or {}).get("ignored_group_test_targets", {}).items()
    ]
    lines = [
        "### Platform selection",
        "",
        "Selected: " + (", ".join(p["name"] for p in platforms) or "none (lint only)"),
        "",
    ]
    lines.extend(f"- {escape(reason)}" for reason in plan["reasons"])
    lines.extend(
        [
            "",
            "### Unit test plan",
            "",
            f"- Stage: `{plan['stage']}`",
            f"- Selection: `{plan['selection']}`",
            (
                "- Scope:"
                f" `{'full in-level scope' if selection['full_scope'] else 'affected groups plus baseline'}`"
            ),
            f"- Level: `{plan['test_level']}`",
            f"- Required blocked groups: {sum(bool(g.get('required')) for g in plan['blocked'])}",
            (
                "- Gate policy: required groups must execute and pass; declared platform gaps are"
                " recorded separately"
            ),
            f"- Selected logical groups: {len(plan['test_groups'])} platform/device entries",
            f"- Selected execution groups: {len(executions)} jobs",
            f"- Changed files: {plan['changed_files']['count']}",
            f"- Ignored entries: {len(plan['ignored'])}; blocked entries: {len(plan['blocked'])}",
            "- Selected test groups: " + (", ".join(selection["groups"]) or "none"),
            f"- Plan hash: `{plan['plan_hash']}`",
            "",
            "| Platform | Unit jobs | Integration jobs | Other jobs | Declared coverage gaps |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for platform in platforms:
        groups = [g for g in executions if g["platform"] == platform["name"]]
        units = sum(g["kind"] == "unit" for g in groups)
        integrations = sum(g["kind"] == "integration" for g in groups)
        lines.append(
            f"| {escape(platform['name'])} | {units} | {integrations} |"
            f" {len(groups)-units-integrations} | {len(platform['ignored_test_groups'])} |"
        )
    lines.extend(["", "Coverage gaps and ignored targets are not passing-test evidence.", ""])
    if gaps:
        lines.extend(["<details><summary>Declared platform coverage gaps</summary>", ""])
        lines.extend(f"- {escape(gap)}" for gap in gaps)
        lines.extend(["", "</details>", ""])
    if plan["blocked"]:
        lines.extend(["<details><summary>Blocked test groups</summary>", ""])
        lines.extend(
            f"- {escape(g['platform'])}/{escape(g['group'])}: {escape(g.get('reason', 'blocked'))}"
            for g in plan["blocked"]
        )
        lines.extend(["", "</details>", ""])
    with path.open("a") as output:
        output.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["pr", "branch", "nightly", "manual", "release"],
    )
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--configs-dir", default=".github/configs")
    parser.add_argument("--project", default=".flagcicd/project.yaml")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--changed-files", default="")
    parser.add_argument("--platforms", default="")
    parser.add_argument("--requested-level", default="auto")
    parser.add_argument("--labels-from-event", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--github-output", action="store_true")

    args = parser.parse_args()

    try:
        plan = build_plan(args)

        # Write plan file
        if args.output:
            args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

        # Write GitHub outputs
        if args.github_output:
            gh_out = os.environ.get("GITHUB_OUTPUT")
            if gh_out:
                with open(gh_out, "a") as f:
                    # Compact JSON for output
                    plan_json = contract.canonical_json(plan)
                    f.write(f"plan={plan_json}\n")
                    f.write(f"plan_hash={plan['plan_hash']}\n")
                    f.write(f"platforms={json.dumps([p['name'] for p in plan['platforms']])}\n")
                    f.write(f"source_sha={plan['source_sha']}\n")
                    f.write(f"test_level={plan['test_level']}\n")
                    f.write(f"stage={plan['stage']}\n")
                    f.write(f"lint_only={json.dumps(plan['lint_only'])}\n")
        else:
            # Default: write to stdout if no output options specified
            print(json.dumps(plan, indent=2, sort_keys=True))

        # Write the plan only in the real planning job, not self-test subprocesses.
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            write_summary(Path(summary), plan)

        return 0

    except (OSError, ValueError, subprocess.CalledProcessError, yaml.YAMLError) as exc:
        print(f"::error::Plan generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
