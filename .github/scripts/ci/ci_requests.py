"""Resolve public CI requests and maintainer approvals to executable requests."""

import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

PLATFORM_LIST = r"[a-z0-9_-]+(?:[ ,]+[a-z0-9_-]+)*"


class AvailablePlatforms(set):
    def __init__(self, configs_dir):
        import generate_matrix as matrix

        enabled = matrix._load_watch_rules(configs_dir)
        super().__init__(enabled)
        configured = {
            p.stem for p in configs_dir.glob("*.yml") if p.stem not in matrix._NON_PLATFORM_STEMS
        }
        self.disabled = configured - self


TEST_LEVELS = frozenset(("auto", "L0", "L1", "L2", "L3", "L4"))


def select_platforms(requested, available):
    if not re.fullmatch(PLATFORM_LIST, requested):
        raise ValueError("Expected comma/space-separated platforms or 'all'")
    selected = set(re.split(r"[ ,]+", requested))
    disabled = selected & getattr(available, "disabled", set())
    if disabled:
        raise ValueError(f"Platform disabled: {', '.join(sorted(disabled))}")
    if not available or selected - available - {"all"}:
        raise ValueError("Unknown platform or no configured platforms")
    return sorted(available if "all" in selected else selected)


def resolve(event, api, available):
    if event.get("action") != "created" or not event.get("issue", {}).get("pull_request"):
        raise ValueError("Expected a newly created PR comment")
    comment = event["comment"]
    number = event["issue"]["number"]
    if number <= 0:
        raise ValueError("Expected a positive PR number")
    body = comment["body"].strip()
    approval = re.fullmatch(r"/ci approve ([1-9][0-9]*)", body)
    request = re.fullmatch(r"/(?:ci run|test) ([^\r\n]+)", body)
    if not approval and not request:
        raise ValueError(
            "Expected /ci run <platforms|all>, /test <platforms|all> or /ci approve <comment-id>"
        )

    login = comment["user"]["login"]
    permission = api(f"collaborators/{quote(login, safe='')}/permission")["permission"]
    approved = permission in {"write", "maintain", "admin"}
    if approval:
        if not approved:
            raise PermissionError("Approving CI requires repository write permission")
        original = api(f"issues/comments/{approval[1]}")
        issue_url = event["issue"].get("url")
        if not issue_url or original.get("issue_url") != issue_url:
            raise ValueError("Approval must reference a request on this PR")
        request = re.fullmatch(r"/(?:ci run|test) ([^\r\n]+)", original["body"].strip())
        if not request:
            raise ValueError("Referenced comment is not a CI request")

    platforms = select_platforms(request[1], available)
    pr = api(f"pulls/{number}")
    sha = pr["head"]["sha"]
    if pr["state"] != "open" or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected an open PR with a valid head SHA")
    return platforms, sha, approved


def resolve_dispatch(inputs, available, default_ref):
    """Resolve workflow_dispatch inputs; dispatching already requires write access."""
    platforms = select_platforms(str(inputs.get("platforms", "")).strip(), available)
    level = inputs.get("test_level") or "auto"
    if level not in TEST_LEVELS:
        raise ValueError(f"Unknown test level: {level}")
    ref = str(inputs.get("checkout_ref", "")).strip() or default_ref
    if not re.fullmatch(r"\S+", ref):
        raise ValueError("Expected a checkout ref without whitespace")
    return platforms, ref, level


def main():
    def api(path):
        return json.loads(
            subprocess.check_output(
                ["gh", "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/{path}"], text=True
            )
        )

    available = AvailablePlatforms(Path(".github/configs"))
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        platforms, ref, level = resolve_dispatch(
            event.get("inputs") or {}, available, os.environ["GITHUB_SHA"]
        )
        approved = True
    else:
        (platforms, ref, approved), level = resolve(event, api, available), "auto"
    if not approved:
        message = (
            "### CI request awaiting maintainer approval\n\nRequested platforms:"
            f" {', '.join(platforms)}\n\nPR head at request time: `{ref}`\n\nA maintainer can"
            f" approve on this PR with `/ci approve {event['comment']['id']}`.\n\nApproval runs the"
            " PR head at approval time. No PR code or device tests run while pending.\n"
        )
        print(message)
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary, "a") as output:
                output.write(message)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(
            f"platforms={json.dumps(platforms)}\ncheckout_ref={ref}\ntest_level={level}\n"
            f"approved={json.dumps(approved)}\n"
        )


if __name__ == "__main__":
    main()
