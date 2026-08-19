# CI Testing Guide

This guide describes the configuration-driven TransformerEngine-FL test
workflows, the platform test-runner layout, and how to reproduce a CI test from
a development machine.

For the test source layout and naming basics, see [`README.md`](README.md). For
plugin test ownership, see [`plugin/README.md`](plugin/README.md).

## CI architecture

Each hardware platform is connected to CI through three files:

| Component | Location | Responsibility |
| --- | --- | --- |
| Platform workflow | `.github/workflows/all_tests_<platform>.yml` | Triggers the shared workflow and enables test classes |
| Platform config | `.github/configs/<platform>.yml` | Selects the image, runner, container, device, test matrices, and coverage |
| Platform setup | `.github/scripts/setup_<platform>.sh` | Validates the image/runtime and exports CI environment variables |

The common execution path is:

```text
all_tests_<platform>.yml
  -> all_tests_common.yml
     -> optional build_test_artifact
        -> build one installable artifact without accelerator devices
     -> unit_tests_common.yml
        -> setup_<platform>.sh
        -> tests/test_utils/run_ci_test_group.py
        -> platform run_unit_tests.sh or a shared QA/pytest target
     -> integration_tests_common.yml
        -> setup_<platform>.sh
        -> platform run_integration_tests.sh or a shared QA target
```

The shared workflows must remain hardware-neutral. Device imports, runtime
patches, environment overrides, and platform-specific exclusions belong in the
platform setup or `tests/plugin/backend/<platform>/`.

Unit tests complete before integration tests start. A unit-test failure blocks
the integration matrix for the same platform workflow.

## Platform test-runner layout

Platform-owned launchers and their support files live together:

```text
tests/plugin/backend/<platform>/
  run_unit_tests.sh
  run_integration_tests.sh
  config.sh                    # Optional test selection and exclusions
  set_env.sh                   # Optional sourceable runtime environment
  <platform support files>     # Optional narrowly scoped helpers
```

Use these names consistently:

- `run_unit_tests.sh` is the script entry point for `unit_test_matrix` groups.
- `run_integration_tests.sh` is the entry point for `integration_test_matrix`.
- `config.sh` contains test selection, exclusions, and runner knobs, not image
  or device allocation configuration.
- `set_env.sh` exports runtime variables needed by both setup and test runners.
- Support scripts use names that describe their operation, such as
  `patch_megatron_mccl.py`.

Do not create `tests/integration/<platform>/` for a platform launcher. Shared
integration test implementations remain under `qa/`; only the platform entry
point and platform-specific support files belong under the plugin backend.

An `__init__.py` is only needed when the directory is intentionally imported as
a Python package. Shell-only runner directories do not need one.

A platform that needs no runtime wrapper may point its matrix directly at a
shared QA script or use a declarative pytest group. Do not add an empty platform
launcher solely to satisfy the directory shape.

## Platform configuration contract

`.github/configs/<platform>.yml` is the source of truth for the CI job. Platform
metadata and fields consumed by the common workflows are:

| Field | Meaning |
| --- | --- |
| `hardware_name` | Stable hardware identifier used by the platform config |
| `display_name` | Human-readable platform name |
| `ci_image` | Validated container image containing the vendor runtime |
| `container_pull_policy` | Docker pull policy; defaults to `never` |
| `runner_labels` | Labels selecting the self-hosted runner |
| `container_volumes` | Host-to-container volume mappings |
| `container_options` | Runtime, device, IPC, memory, and security options |
| `device_types` | Device values used to expand the CI matrix |
| `checkout_submodules` | Checkout submodule mode; defaults to `false` |
| `setup_script` | Repository-relative platform setup script |
| `test_artifact` | Optional build-once artifact shared by all test jobs |
| `unit_test_matrix` | Unit-test job entries; normally one serial launcher |
| `unit_test_timeout_minutes` | Optional unit-job timeout; defaults to 90 minutes |
| `integration_test_matrix` | Named integration-test entries |
| `coverage` | Optional Python coverage configuration |

Keep the vendor PyTorch stack, accelerator runtime, communication libraries,
and expensive build dependencies in the image. Setup scripts should validate
that stack and activate it; they should not silently replace it with packages
from public package indexes.

### Build-once test artifacts

Platforms that compile the same source package in every matrix job should use
the optional `test_artifact` contract:

```yaml
test_artifact:
  enabled: true
  build_script: .github/scripts/build_example_test_artifact.sh
  path: ci-artifacts/example-wheel
  runner_labels:
    - example-build-runner
  container_options: >-
    --user root
```

`all_tests_common.yml` runs the build script once, uploads the files under
`path`, and makes the artifact available to every unit and integration job.
The reusable test workflows download it before invoking `setup_script` and set
`TE_CI_ARTIFACT_DIR` to the repository-relative download directory.

The build script and setup script own the artifact format. For a Python wheel,
the build script should produce exactly the wheel needed by the configured
image, while setup should install that wheel instead of rebuilding the source.
Keep the artifact key scoped to the workflow's checked-out commit and do not
reuse it across incompatible Python, framework, toolkit, ABI, or build flags.

When compilation only needs the vendor toolkit, omit accelerator device
options from `test_artifact.container_options`. `runner_labels` may select a
CPU build runner when it can run the same image. If only the accelerator host
has that image and toolchain, the build can still run there without exposing
devices; building once still removes repeated compilation from the test matrix.

### Unit-test matrix

A platform normally defines one unit-test job and lets its standard launcher
run the debug, PyTorch, distributed, and ONNX suites serially:

```yaml
unit_test_matrix:
  - name: pytorch_unittest
    runner: script
    path: tests/plugin/backend/example/run_unit_tests.sh
```

`tests/test_utils/run_ci_test_group.py` supports two runner types:

- `script`: executes `path` with optional `args` and `env`.
- `pytest`: executes declarative `steps`, targets, arguments, environment, and
  optional JUnit output without a platform shell wrapper.

Use `pytest` for hardware-neutral collections that only need a different pytest
command. Use `script` when the platform needs multiple commands, runtime
preflight, conditional groups, or platform-owned failure aggregation.

Keep a single `pytorch_unittest` entry unless the platform can actually run
independent entries concurrently. The launcher should run all supported suites
when called without arguments, retain optional suite arguments for local
debugging, aggregate failures, and write each suite's JUnit files to a separate
log subdirectory. Script paths are repository-relative and must remain under
`tests/plugin/backend/<platform>/` for platform-owned runners.

### Integration-test matrix

An integration entry names one executable platform launcher:

```yaml
integration_test_matrix:
  - name: pytorch_mcore_integration
    path: tests/plugin/backend/example/run_integration_tests.sh
```

The launcher prepares platform-specific variables and then delegates to the
shared integration implementation, for example:

```bash
exec bash "$TE_PATH/qa/L1_pytorch_mcore_integration/test.sh"
```

Keep model construction, training, checkpoint, and common success criteria in
the shared QA script. Keep device selection, collective backend checks, and
temporary platform compatibility helpers in the platform launcher.

When a pinned MCore checkout needs a temporary platform preparation step, the
launcher may export `MCORE_PREPARE_SCRIPT`. The shared MCore QA invokes that
Python script with `MCORE_PATH` after checkout. Keep the helper next to the
platform launcher and remove it when the pinned MCore ref contains the fix.

## Environment setup contract

The platform setup script runs before every unit and integration matrix entry.
It should:

1. Locate and export the vendor runtime paths.
2. Validate the configured accelerator is visible.
3. Validate required Python packages and backend implementations.
4. Preserve the image-provided PyTorch and vendor libraries.
5. Write required variables to `GITHUB_ENV` when that variable is available.

GitHub Actions invokes setup with `bash`, so environment needed by later steps
must be written to `GITHUB_ENV`. During local reproduction, use `source` so the
same exports remain in the current shell.

If several launchers need the same exports, place them in
`tests/plugin/backend/<platform>/set_env.sh` and source that file from both the
setup script and the launchers.

## Reproducing CI locally

Use the exact image, mounts, and container options from the platform config.
Host Python is not a CI-equivalent environment unless it already matches the
vendor runtime and package versions in that image.

At the repository root, inspect the selected configuration:

```bash
export PLATFORM=enflame
export CONFIG_FILE=".github/configs/${PLATFORM}.yml"

yq '.ci_image, .runner_labels, .container_volumes, .container_options' \
  "$CONFIG_FILE"
```

Start the configured image with every `container_volumes` and
`container_options` value, then mount this checkout and use it as the working
directory:

```bash
docker run --rm -it \
  <container options from the platform config> \
  <volume options from the platform config> \
  -v "$PWD:/workspace/TransformerEngine-FL" \
  -w /workspace/TransformerEngine-FL \
  <ci_image from the platform config> \
  bash
```

Do not replace vendor device options with `--gpus all`. Copy the configured
runtime and device mappings exactly.

### Run the unit-test job

Inside the container:

```bash
export GITHUB_WORKSPACE="$PWD"
export PLATFORM=enflame
export GROUP=pytorch_unittest
export CONFIG_FILE=".github/configs/${PLATFORM}.yml"

SETUP_SCRIPT="$(yq -r '.setup_script' "$CONFIG_FILE")"
source "$SETUP_SCRIPT"

export TE_TEST_GROUP_JSON="$(
  yq -o=json -I=0 \
    '.unit_test_matrix[] | select(.name == strenv(GROUP))' \
    "$CONFIG_FILE"
)"
test -n "$TE_TEST_GROUP_JSON"

python3 tests/test_utils/run_ci_test_group.py
```

This uses the same group definition and dispatcher as CI. A direct pytest
command is useful for debugging, but it does not prove that the configured
runner, environment, exclusions, JUnit output, and coverage behavior work.

To reproduce only one suite while debugging a platform launcher, pass its
short name directly:

```bash
bash tests/plugin/backend/enflame/run_unit_tests.sh debug
```

### Run one integration test

Start with a fresh shell or container, then run:

```bash
export GITHUB_WORKSPACE="$PWD"
export PLATFORM=enflame
export GROUP=pytorch_mcore_integration
export CONFIG_FILE=".github/configs/${PLATFORM}.yml"

SETUP_SCRIPT="$(yq -r '.setup_script' "$CONFIG_FILE")"
source "$SETUP_SCRIPT"

INTEGRATION_SCRIPT="$(
  yq -r \
    '.integration_test_matrix[] | select(.name == strenv(GROUP)) | .path' \
    "$CONFIG_FILE"
)"
test -f "$INTEGRATION_SCRIPT"

bash "$INTEGRATION_SCRIPT"
```

Integration tests may clone or update another FlagOS repository. Keep its ref
pinned, and distinguish a network checkout failure from a model or backend
failure.

## Adding a test

For a hardware-neutral plugin contract:

1. Add the test below `tests/plugin/plugin/`, `tests/plugin/backend/reference/`,
   or `tests/plugin/backend/flagos/` according to ownership.
2. Add it to an existing declarative pytest group or platform launcher.
3. Confirm the CI runner actually collects it.
4. Run every platform group whose shared behavior changed.

For platform-specific runtime behavior:

1. Put the test or support helper below `tests/plugin/backend/<platform>/`.
2. Invoke it from `run_unit_tests.sh` or `run_integration_tests.sh`.
3. Keep the common workflows and shared TE source free of vendor conditionals.

## Adding a platform

1. Add `.github/configs/<platform>.yml` with image, runner, container, device,
   setup, and test matrices.
2. Add `.github/scripts/setup_<platform>.sh` for image and device validation.
3. Add `tests/plugin/backend/<platform>/run_unit_tests.sh` when unit execution
   needs a platform wrapper; otherwise use a shared QA or declarative pytest
   group.
4. Add `run_integration_tests.sh` when integration coverage needs a platform
   wrapper; otherwise point to the shared QA entry.
5. Add `.github/workflows/all_tests_<platform>.yml` calling
   `all_tests_common.yml`.
6. Validate the serial unit entry and one integration entry before enabling the
   platform workflow.

A normal platform addition must not require a hardware branch in
`all_tests_common.yml`, `unit_tests_common.yml`, or
`integration_tests_common.yml`.

## Exclusions and temporary workarounds

An exclusion is acceptable only when the platform cannot support the behavior
or a tracked runtime limitation makes the test unsafe. Keep it narrow and
record the reason near the exclusion.

- Prefer a pytest node or named expression over skipping a whole file.
- Keep platform exclusions in `config.sh` or the platform unit launcher.
- Do not change common test expectations to make one accelerator pass.
- Do not report a reference-backend fallback as vendor-backend coverage.
- Remove temporary patches when the pinned dependency contains the fix.

## Logs, coverage, and failure diagnosis

Unit launchers should emit one JUnit XML file per meaningful step under
`XML_LOG_DIR`. The common unit workflow uploads logs and, when configured,
collects Python coverage for the selected source and include patterns.

Classify failures by the stage that actually ran:

| Symptom | First place to inspect |
| --- | --- |
| Job remains queued | Exact runner labels and runner availability |
| Container does not start | Image pull policy, runtime, devices, mounts, and permissions |
| Setup fails | Image package versions, device visibility, and exported paths |
| Configured path is missing | Platform matrix path and runner layout |
| Pytest collection fails | Plugin imports, required modules, and pytest arguments |
| One unit sub-step fails | Its JUnit XML and the first Python exception |
| Distributed timeout | First failed rank and collective backend preflight |
| Integration checkout fails | Network access, repository URL, and pinned ref |
| Model build or optimizer fails | Selected platform/backend and parameter device type |
| Coverage is missing | Coverage dependencies, raw fragments, and include/omit filters |

Warnings about an optional dependency are not a failure unless the selected
test requires that dependency. Use the process exit code and the first failing
test step as the success boundary.

## Submission checklist

- Platform files follow the standard plugin backend layout and runner names.
- Every matrix path exists and is executable by `bash`.
- The setup script validates the configured image without replacing its vendor
  runtime.
- Unit groups run through `tests/test_utils/run_ci_test_group.py`.
- Integration entries run through their platform launcher.
- Exclusions are narrow, justified, and do not weaken other platforms.
- Shell syntax, YAML parsing, and repository path checks pass.
- At least one CI-equivalent unit group and each changed integration entry were
  validated on the target hardware.
- The complete platform workflow passes before merge.
