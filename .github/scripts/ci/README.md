# TransformerEngine-FL CI

The active entries are `ci.yml` (PR, push to main, manual dispatch),
`nightly.yml` (stage `nightly`, 00:00 Asia/Shanghai, or manual dispatch),
`ci-requested.yml` (stage `manual`) and `release-acceptance.yml` (stage
`release`). Each generates a plan and directly calls `all_tests_common.yml` for its
platform matrix; test jobs use `test_groups_common.yml`. `CI Complete`, `Nightly Complete`, `CI Requested Complete` and
`Release Acceptance Complete` gate the results.

## Execution plan transport

The planning job uploads the full `plan.json` as `flagcicd-plan-<stage>`.
Entry workflows export only its artifact name and small routing/identity outputs;
reusable workflows and the final gate download that artifact and pass its local
path as `FLAGCICD_PLAN_FILE`. This keeps both plan loading and `toJson(needs)`
below Linux's per-environment-string limit when all nine platforms are selected.
Do not export the full plan through job outputs or a `FLAGCICD_PLAN` step env.

Every plan file goes through the existing schema/hash checks; slice still checks
source SHA and platform configuration identity. A configured missing/corrupt file
fails rather than falling back to other data. The JSON-env reader remains only
for compatibility with direct script callers and unit fixtures.

## Automatic plugin coverage

The unified CI catalog covers the `tests/plugin/` tree with separate owners:
`plugin_core` selects `tests/plugin/plugin/`, `reference_backend` selects
`tests/plugin/backend/reference/` and `tests/plugin/accuracy/`, and
`vendor_backend` selects `tests/plugin/backend/npu/` and
`tests/plugin/backend/flagos/`. Source routing follows those same boundaries.
Core dispatch evidence must come from core tests, not a passing backend test.

Unified CI owns automatic plugin execution. The independent
`te-plugin-tests.yml` workflow is removed because its non-NPU test scope is
already covered by the CUDA unit job. Manual runs remain available through the
unified CI entry points. Runner labels come from the platform configuration
(`nv-8g-cicd-te` for CUDA), avoiding a separate runner definition.

Platform unit launchers that run plugin suites discover all `test_*.py`
recursively through the shared `TE_TEST_GROUP_PRESET=plugin-unit` preset in
`tests/test_utils/run_ci_test_group.py`. The preset supplies the discovery
paths, non-Ascend NPU ignore prefix and path-scoped environment cleanup;
platform scripts do not duplicate inline JSON or `TE_FL_*` unset lists. Inner
pytest groups disable retries so the outer platform job owns retry policy.
PPU retains its per-file timeout/summary and Iluvatar its per-file ignore audit,
while both query `--list-targets` and `--target-env` from the shared preset.
MUSA and Kunlun keep their existing unit suites unchanged while `plugin_core`,
`reference_backend` and `vendor_backend` remain in `ignored_test_groups`.
The old per-file backend calls are removed to avoid duplicate execution.
Distinct relative paths produce distinct JUnit filenames even when test files
share a basename. Numerics helper files are invoked by their driver rather
than discovered twice. Core mock contracts clear vendor policy overrides;
backend/accuracy tests retain real platform policy. CI test selection and
coverage hooks remain in effect in all child processes.

Non-Ascend plans and the shared preset ignore `tests/plugin/backend/npu/` before
starting pytest.
The common discovery root is `tests/plugin/`; platforms do not maintain lists
of individual plugin test files. All original platform group ignores, their
reasons, and existing file/case ignore rules are preserved. Expanding discovery
does not establish hardware support and does not remove those ignores.

The planner expands ignored groups (and non-Ascend NPU exclusions) into
`ignored_test_targets`, including dependency-derived ignores, and records any
resulting cross-group target suppressions in `ignored_group_test_targets` and
the plan summary. The runner passes `ignored_test_targets` to every pytest child.
Whole-file/directory ignores apply before import; node selectors apply after
collection. Ignores take precedence over positive directory selections, even
when another selected group overlaps the ignored files and even in full Nightly
runs. Ignored tests produce no passing evidence. Standard capability-based
runtime skips remain separate. When plan deselection and platform `-k`/`-m`
filters leave an invocation with zero selected nodes, the selection plugin exits
cleanly without recording passing evidence; group gates still require actual
passing testcases.

## Ignore naming and include scopes

Repository-controlled test omissions use `ignore` names: platform group reasons
are configured in `ignored_test_groups`, whole-file and case filters use
`*_IGNORE_*`, the plan records an `ignored` ledger, and launcher messages use
`[IGNORE]`. PPU records whole-file ignores in `ignored-tests.tsv` and reports
those rows as `ignored`. Reasons, owners and restore conditions are retained.
The former `unsupported_test_groups` key is rejected with a migration error,
rather than silently enabling tests from an old ignore list. These entries
never count as passing evidence. A required baseline still cannot
be ignored; ignoring a dependency also makes its dependent groups unavailable.

`coverage.include` chooses source files counted in coverage; it does not choose
which tests run. GitHub Actions matrix `include` enumerates jobs. Neither is an
ignore list. `test_targets` defines the positive test selection for a group.

Pytest/JUnit `skipped`, pytest `--deselect` and collection hooks, GitHub Actions
`skipped` results, and coverage.py `include`/`omit` are native interface names.
They remain unchanged, as do build/dependency setup switches such as
`TE_FL_SKIP_CUDA` and `HYGON_SKIP_DEP_INSTALL`. Capability-driven runtime skips
remain distinct from repository-configured ignores.

## Selection and levels

PR routing uses platform `paths_watch`. Vendor paths select their platforms;
shared/unknown paths and unavailable diffs select all platforms. Documentation
only changes skip hardware. `platform:<name>` labels override routing, so a
narrowed shared-code run requires reviewer judgment. `level:L0` through
`level:L4` labels can escalate the inferred level. Path-based test level rules also raise it.

PR defaults to L1; branch/CI dispatch defaults to L2; Nightly is fixed at L2.
Each changed file takes the highest level of the test level rules it matches, or
`unknown_level` when it matches none; the plan takes the maximum over files.
Levels are cumulative, subject to group path filters and platform support.
Nightly has no diff filters and runs all currently executable configured L0-L2 groups.
L1 includes debug and portable unit modules. L2 adds distributed, ONNX and
MCore integration. Their source/test/launcher paths explicitly escalate to L2;
shared TE source also has an L2 floor. Level is an eligibility ceiling: affected
PRs select modules within it, while full plans run every configured suite in it.

`test_groups` in `.flagcicd/project.yaml` is the single test catalog. Each
entry declares changed paths, level, status, `depends_on`, evidence, resources
and `execution: {matrix, entry}`. Current functional groups are core, plugin
core, reference/vendor backend, normalization, GEMM, attention, quantization,
optimizer, debug, distributed, ONNX and MCore. Preflight and blocked L3/L4
acceptance entries use the same catalog. There is no separate module catalog.
The planner adds `levels.*.required_groups` (preflight and core) and recursively
closes `depends_on`: selecting reference_backend also selects plugin_core and core.
Dependencies expand selection; they do not impose job execution order.

`routing.shared_paths`, declared impact paths, unavailable diffs and unmapped
code select the complete in-level scope. Platform routing remains independent.
Platform `ignored_test_groups` records group -> reason gaps in the existing
QA coverage. They appear in plans, workflow summaries and platform reports, and
cannot be claimed as tested. A group with an unavailable dependency is also
unavailable. These documented gaps do not block runs on other supported groups;
an unavailable required core baseline is an error. This is a description of wired tests, not hardware capability detection.

The plan separates selected test requirements from their runtime jobs:
`group_selection` records selection and dependency closure, `test_groups`
contains logical groups per platform/device, and `execution_groups` deduplicates
runtime jobs. Groups whose `execution` points to `unit_test_matrix/pytorch_unit`
share the existing unit job and platform QA script without running the launcher
repeatedly. Shared jobs require
the same adapter, required/evidence/coverage policy and device resources;
incompatible declarations fail planning rather than weakening a group's gate.
The planner passes the union
of `test_targets` plus `group_test_targets` through that job's execution data;
`run_ci_test_group.py` enables `ci_pytest_selection.py` for child pytest calls.
It filters test files before collection and selected nodes after collection,
while preserving original node IDs, `-k` and platform exclusions. `test_targets`
accepts files/directories/globs and `file.py::Class::test_*` node selectors.
Bare function/class selectors include their parameter combinations; an explicit
parameter ID matches literally. Mixed sanity/numerics/fusible/NPU files use
node selectors so a linear testcase cannot certify normalization or attention.
The pytest plugin imports the same stdlib target matcher from flagcicd_contract
as the planner/gate; the runner adds both helper directories to PYTHONPATH.
Each selected unit group needs at least one matching passing testcase and no
matching failure/error, identified by the canonical JUnit `flagcicd_test_nodeid`
property. The file-only property is diagnostic and cannot prove a node ran.

Full plans retain the original QA scope; no unconfigured vendor tests are added.
The four execution suites retain multi-device serial isolation and their
platform-specific timeouts. FA compatibility and benchmarks are L3 and remain
blocked by runner/baseline requirements; L4 acceptance retains explicit gaps.
Plans record selection, platform support, execution groups, ignored and blocked
ledgers. See `.flagcicd/project.yaml`.

Requested runs: a PR comment `/ci run <platforms|all>` or `/test <platforms|all>`
is accepted from anyone. Users with write/maintain/admin permission run tests
immediately; other requests remain pending without checking out PR code or
running device tests. The workflow summary shows `/ci approve <comment-id>`;
a maintainer posts that command on the same PR to approve. Approval revalidates
the request and runs the PR head at approval time. Edited comments do not trigger
runs; create a new approval comment to run again. GitHub runs `issue_comment`
workflows from the default branch, so this behavior requires merging there.
`workflow_dispatch` on
`ci-requested.yml` takes platforms, a test level and a checkout ref.
`release-acceptance.yml` runs the fixed L4 release plan for one ref; required
blocked groups fail it. Weekly is not configured.

## Execution and evidence

The plan pins source SHA, platform config hashes, group commands, resources,
and coverage settings. Each platform slices `execution_groups` into parallel
and serial lanes.
Exclusive or multi-device groups run serially after parallel work finishes,
whatever its result; the platform gate judges both lanes. This only
orders jobs within a platform invocation; runner deployment must provide
physical device isolation across PR/Nightly runs.

Optional wheel artifacts carry identity metadata checked before setup: schema,
platform/source/config identity and the exact wheel names, sizes and SHA256 hashes.
These checks are mandatory whenever an artifact is used. Group
reports contain plan/source identity, status, execution and duration. Each runtime
artifact includes `test_group_reports` for the logical groups it executed. The
platform and global gates check those logical reports against `plan.test_groups`
as well as the shared runtime results. Matching node outcomes determine each
logical group's status and counts: core can pass while optimizer fails in the
same job. Missing/all-skipped matching evidence fails that group. Common setup,
identity, cancellation or a command failure without a failing JUnit case still
apply to all groups.
The shared job remains failed even if a failed testcase cannot be assigned to a
selected group. Required groups come from the plan and need execution evidence.
Missing, duplicate, mismatched or failed required reports fail the platform gate.
Each group declares `evidence`: `junit` groups need non-empty JUnit results,
`exit_code` groups (preflight, script integration) pass on exit code 0.
JUnit checks actual testcase outcomes and validates declared suite counts,
including nested suites. Missing, corrupt or inconsistent reports cannot pass.
Pytest execution entries retain steps, arguments, environment, retries and log
paths through plan slicing; report collection uses the same log directory.
Optional group failures or missing reports do not block the pipeline; their
available reports still undergo identity/consistency checks. Required groups
remain blocking at both the workflow and platform gate.
Preflight checks
repository structure/Python syntax; it does not certify device availability.
The global gate requires successful planning/pipeline results and complete
platform reports. Required blocked groups fail acceptance.

## Coverage

Coverage measures Python source, not C++ or device kernels. Group jobs collect
the platform `coverage.sources` with `--cov-append`, upload raw hidden fragments,
and `all_tests_common.yml` combines them, applying `coverage.include`/`omit`.
Platforms with `coverage.required: true` fail if collection/aggregation fails
or the coverage job is skipped. This requirement is separate from regression
thresholds, which are advisory. Regression checks only compare full configured
coverage scopes; module- or suite-subset coverage uses a `-partial` artifact name and cannot
become a full baseline. Optional collection/aggregation failures are
also non-blocking. These are fixed gate/workflow rules; project.yaml does not
expose inactive validation toggles or unimplemented image-cache declarations.

PR coverage compares against a main-branch artifact of the same platform/device.
Nightly uses a 14-run history with at least 8 samples and a lower bound of
mean minus three standard deviations. Threshold configuration lives in
`.flagcicd/golden_baseline.json`. Baselines are fetched with the Actions REST
API from Python, not `gh`; live baseline collection must be validated on
runners. FlagOps uploads are advisory and need
`FLAGCICD_API_TOKEN`; fork runs usually have no secret.

## Validation

```sh
python -m pip install PyYAML==6.0.3 pytest==8.4.2
python -m unittest discover -s .github/scripts/ci -p 'test_*.py'
python .github/scripts/ci/flagcicd_planner.py --stage nightly --output /tmp/plan.json
```

Local validation does not prove hardware compatibility or GitHub nested matrix
behavior. Validate docs-only/vendor/shared routing, deliberate test/setup
failure, absent reports, coverage failure and cancellation on Actions before
changing required checks. Configure `CI Complete` in branch protection manually.
Schedules require the workflow on the default branch and may be delayed.

## Scope and device counts

This pipeline uses static routing: changed paths, stage defaults, test level rules and
labels select platforms and test groups. Runtime-evidence collection, cached
result pruning, priority waves and reduced workloads are outside this change.
JUnit/exit-code execution evidence, artifact identity checks and all three report
levels remain required by the existing plan and gate contracts.

Keep `nproc_per_node` in each `.github/configs/<platform>.yml`: it is a positive
integer device budget per multi-device test job, not the total number of cards
installed on the host. The planner rejects missing or invalid counts rather than
assuming one device. `multi_device` and exclusive resource classes resolve this
platform value; `single_device` groups still request one device. Counts may differ
by platform (currently 2, 4 or 8) and do not require duplicated workflows.

The budget is a scheduling declaration, not automatic hardware detection or a
universal torchrun override. Keep it aligned with container-visible devices and
platform launchers; launchers can use fewer devices in individual phases. Changing
physical card allocation also requires checking the runner, device mounts and
visibility settings. Runtime device isolation remains a runner responsibility.


### Platform enablement and test ordering

Platform YAML files accept `enabled: true|false` (default: true for existing
configurations). Disabled platforms are excluded from automatic plans and cannot
be requested manually. Provision runners and preload images before enabling.

Integration runs after the unit lanes and their evidence report complete, even
when unit evidence fails. The final platform gate still rejects required failed
or missing tests. Unit reports are retained for 14 days.

Iluvatar quantization-only recipe/tensor/NVFP4 targets are omitted from routine
CI until recovery validation supplies passing evidence. The shared blockwise
GEMM target remains selected. Whole-file and partial exclusions are currently
recorded in the Iluvatar shell configuration; a per-target coverage-gap schema
and report is follow-up work, and passing group results do not imply complete
coverage of those ignored targets.

The Megatron checkout retry is shared by all platforms: partial-clone checkout
may download blobs from the promisor remote, so it now retries up to three times
with a five-second delay, just like clone and fetch. Exhausted retries still fail.

The unit evidence report checks unit/preflight groups only; compatibility,
benchmark and release groups remain subject to the final platform gate.
Iluvatar ignore rules are audited inline before pytest keyword filtering. Each
rule must match some but not all collected tests; matched node IDs are logged.
This detects missing selectors but does not prove historical quarantine membership
is unchanged. Review the logged node IDs after upstream parameter changes.

### Plugin discovery compatibility

The plugin-unit preset forces `TE_FL_PREFER=vendor` for NPU backend files before
imports. Every platform, including Ascend, runs
`tests/plugin/accuracy/test_compute_scale.py`; the NPU backend follows the TEX
CUDA scale semantics that test asserts.

Isolated pytest exit code 5 is accepted only when the newly generated JUnit
report contains skipped cases exclusively, with no failures or errors. Empty
collections remain failures; skipped cases do not provide passing gate evidence.
Each invocation logs its exit code and elapsed time. Unit jobs default to no
whole-suite retries; integration/compat jobs retain their existing retry policy.
FlagOps upload is advisory and explicitly skips when its API token is absent;
coverage collection, artifacts and required coverage checks remain enabled.

Plugin tests are not quarantined in test code or launcher scripts. A failure
caused by this repository stays a failure until it is fixed. A defect owned by
an external toolchain or image (for example FlagTree or FlagGems) may be listed
in the platform config under `ignored_tests`, using the same convention as
Megatron-LM-FL: a list of `path` or `path::node` entries (coarse or fine), with
a YAML comment above each group giving the reason, owner and restore condition.
The planner merges these entries into each execution's ignored targets, lists
them on the plan's platform entry (`ignored_tests`) and in the plan summary's
coverage gaps. They never count as passing evidence; remove an entry as soon as
its restore condition holds. The shared
workflow passes its platform identity explicitly; the MetaX standalone
environment sets the same identity.
