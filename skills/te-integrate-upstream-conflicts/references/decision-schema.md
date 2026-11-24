# Conflict Decision Schema

A decision is complete only when it answers:

| Field | Requirement |
|---|---|
| path | Exact repository-relative path |
| priority | P0/P1/P2 with evidence |
| fork invariant | Behavior that must survive |
| upstream change | Behavior introduced after base |
| strategy | Manual merge, fork-preserve, upstream-preserve, or redesign |
| owner | Named person/agent or next skill |
| acceptance | Exact command or test |
| status | Proposed, approved, resolved, or blocked |
| evidence | Diff, log, or test artifact under the temp directory |

Clean Git merges still require a semantic decision when both sides changed the path.
