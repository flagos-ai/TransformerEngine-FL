# FlagCICD contract

The authoritative configuration is `project.yaml`; runner/image/command details
remain in `.github/configs/*.yml`. See
[the execution guide](../.github/scripts/ci/README.md) for active entry points,
validation, coverage and current limitations.

All internal schemas currently use v1: project, plan, group-report,
platform-report, gate and artifact-identity. `flagcicd_contract.py` provides
canonical plan hashing and validation; `flagcicd_planner.py` generates plans;
`flagcicd_gate.py` implements slice, group-report, platform and gate commands.

The catalog contains preflight (L0), pytorch_unittest (L1),
pytorch_mcore_integration (L2), fa_versions and pytorch_benchmark (L3), and five
L4 acceptance groups: cross_repo_integration, model_validation,
release_wheel_validation, backward_compatibility, golden_baseline_acceptance.
Blocked entries stay in `plan.blocked` rather than executable matrices. A
required blocked entry prevents a successful global acceptance gate.

A plan describes implemented execution scope, not proof that L3/L4 acceptance
is complete. Dedicated manual/release workflows, performance/numerical goldens
and the declared L4 acceptance jobs remain future work. Python coverage
thresholds in `golden_baseline.json` are not numerical accuracy goldens.
