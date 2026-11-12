# CI/CD Audit Rules

Check uses: ./path, workflow_call workflows, shell commands, and config paths. A path existing locally but not in the target commit is stale.

Record vendor coverage by backend identity, not workflow filename. Separate unsupported hardware from omitted tests. Inspect continue-on-error, || true, if: always(), and upload-only jobs for failure masking.

Record permissions and secrets because pull_request and pull_request_target have different trust boundaries.
