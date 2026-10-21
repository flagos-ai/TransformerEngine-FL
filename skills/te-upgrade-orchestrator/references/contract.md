# Orchestration Contract

Each phase consumes the prior phase's JSON and emits a stable artifact. The orchestrator must fail closed when a file is missing, refs differ, a status is unknown, or a blocker has no owner. A user approval is a state transition and must be recorded with date, scope, and exact authorized action.

The only allowed hardware assumption for this upgrade is the authorized NVIDIA host and its conda environment; non-NVIDIA native tests remain blocked unless separately authorized.
