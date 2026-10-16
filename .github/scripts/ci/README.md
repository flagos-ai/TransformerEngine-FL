# Platform selection and CI gate

For pull requests to `main` and `devops`, `ci.yml` runs lint, computes the
platform matrix, runs the selected platform pipelines, and reports `CI Complete`.

Each platform config defines `paths_watch` using exact repository paths or
directory prefixes ending in `/`. Paths are case-sensitive; glob patterns are
not supported. Vendor aliases include `npu` for Ascend and `kunlunxin` for Kunlun.

- A platform-owned change selects that platform. Multiple platforms are combined.
- Every non-documentation path must match a rule. Shared or unknown paths run all
  platforms, including in PRs that also contain a platform-owned change.
- Changes under `docs/`, or to `.md` and `.rst` files, require no hardware tests.
  Lint and `CI Complete` still run for documentation-only PRs.
  This is an explicit routing policy, not a guarantee that these files can never
  affect runtime behavior. If tests or builds start consuming files in this
  category as inputs, update `_is_documentation` and its regression tests before
  relying on hardware-test skipping for those paths.
- Missing diff information or missing watch rules run all platforms. Invalid
  platform configurations fail matrix generation.
- Renames are treated as deletion plus addition to include both affected paths.
- Pushes that trigger CI, schedules, and manual runs select all platforms.

### Optional routing and retries

- `platform:<config-name>` PR labels override path routing; multiple labels take
  their union, and `platform:all` selects all platforms. Unknown platform labels
  fail generation. Adding/removing labels reruns CI; removing all restores path
  routing. Non-PR runs are unaffected. Narrow shared-code changes only after review.
- Unit/integration groups retry up to twice, five seconds apart, within the same
  job timeout. Each attempt is logged; exhaustion fails the job. Signal exits are
  not retried. A group can set `max_retries: 0`, `1`, or `2`; outside CI the default
  is zero. Setup, builds and lint are not retried.
- Writers/maintainers/admins can comment `/ci run ascend`, `/ci run cuda,ascend`,
  `/ci run all`, or `/test musa` on an open PR. `ci-requested.yml` checks live
  permissions and passes the current head SHA through all configuration/build/
  test/coverage checkouts, including fork PRs. Authorization uses trusted base code
  and a read-only token; requested PR code executes on selected self-hosted hardware.
  Invalid commands/platforms, issue/edited comments, closed PRs and unauthorized
  users cannot start tests. `Requested CI Complete` is supplemental to required
  PR CI. Comment triggering requires this workflow on the default branch and uses
  the GitHub CLI bundled with ubuntu-latest; existing manual workflows are unchanged.

Keep shared dependencies out of individual platform watch lists: shared code
must continue to trigger all platforms. A new platform without watch rules
temporarily disables filtering rather than risking a missed test.

`CI Complete` requires successful lint and matrix generation. A nonempty matrix
requires successful platform pipelines. An empty matrix requires the platform
pipeline to be skipped. Failure, cancellation, or an unexpected skip fails the
gate. Inside each platform, tests may be skipped only when their workflow input
explicitly disables them; artifact builds may be skipped only when disabled in
the platform config.

After verifying a real PR run, configure the protected branch to require the
`CI Complete` status check instead of individual platform test checks. This is a
repository setting and is not changed by these files. Existing independent lint
and other workflow requirements should be reviewed separately.

Run the hardware-independent regression checks with:

```sh
python -m pip install PyYAML==6.0.3
python -m unittest discover -s .github/scripts/ci -p 'test_*.py'
```

Inspect routing without running hardware tests:

```sh
python .github/scripts/ci/generate_matrix.py \
  --event-name pull_request \
  --changed-files tests/plugin/backend/ascend/run_unit_tests.sh
```

## FlagCICD coverage reports

Coverage-enabled platforms collect Python coverage per unit-test group, combine
raw data for groups named `pytorch_*`, and upload
`coverage-<platform>-<device>-pytorch-unittest.json` as a GitHub artifact and
to FlagCICD via `flagos-ai/FlagOps/actions/post-pytest-report@v2`. This does
not measure C++ or device-kernel source coverage.

The upload step runs inside the reusable `unit_tests_common.yml`, so the org
secret `FLAGCICD_API_TOKEN` is forwarded explicitly at every `uses:` level
(`ci.yml`, `_platform_pipeline.yml`, `ci-requested.yml`, `all_tests_*.yml`,
`all_tests_common.yml`). Uploads are non-blocking (`continue-on-error`,
`fail_on_error: 'false'`). Pull requests from forks receive no secrets, so their
upload step fails with an empty `api_token` and is ignored; reports are sent
from runs in this repository (push, schedule, dispatch, `/ci run`).

## Validation on GitHub Actions

Local checks validate routing and execute the gate commands, including failure,
cancellation, and skip combinations. Running these checks on an accelerator
machine still does not validate how GitHub propagates results through nested
reusable workflows and matrices. Verify the following scenarios on GitHub before
changing required checks.

Use a test base branch that already contains this CI implementation, such as
`devops` (which is included in the PR trigger). Create separate scenario branches
and PRs against that base. The PR introducing the CI files itself selects the
full matrix, so it cannot demonstrate documentation-only or vendor-only routing.

| Scenario | Changes or action | Expected result |
| --- | --- | --- |
| Documentation only | Change only `README.rst` or a `.md` file | Lint and matrix generation succeed; matrix is `[]`; platform pipeline is skipped; `CI Complete` succeeds. |
| Vendor only | Change a watched vendor test script, with lint passing | Only the corresponding platform runs; its summary and `CI Complete` succeed. |
| Mixed vendor and shared code | Change a vendor file and a shared runtime file | All configured platforms are selected. |
| Test failure | Temporarily add `exit 1` to a watched vendor test script in the scenario branch | The test, its enabled test summary, platform workflow, and `CI Complete` fail. |
| Lint failure | Introduce a change rejected by an existing pre-commit hook | Lint fails; matrix and platform jobs are skipped; `CI Complete` runs and fails. |
| Cancellation | Cancel a running selected test job or the workflow run | The cancelled work must not produce a successful `CI Complete`; the gate fails if it runs, or remains cancelled if the whole run is cancelled. |
| Label routing | Add/remove `platform:ascend`, combine vendor labels, then use `platform:all` | Each label change reruns PR CI with the documented override; removing the labels restores path routing. |
| Retry recovery/exhaustion | Use a disposable test group that fails once then passes; separately make it always fail | Attempt counts are visible; recovery succeeds, while exhaustion fails summaries and the final gate. |
| Authorized comment | A writer comments `/ci run ascend` on an open PR | Only Ascend runs; every checkout uses the captured PR SHA; `Requested CI Complete` reports the outcome. |
| Rejected comment | Use a read-only account, an invalid platform, an issue comment, or a closed PR | No hardware tests start. |
| Pinned/fork PR request | Update the PR after resolution; separately request a fork PR | Every checkout still tests the captured SHA using base-repository workflows. |

For each run, inspect the platform-selection summary, matrix jobs, nested test
summaries, and the final check on the PR's current commit. Keep intentional
failures in disposable scenario branches; do not merge them into the test base.
Only switch branch protection to `CI Complete` after these scenarios have been
observed. No live Actions results are claimed by the local regression suite.
