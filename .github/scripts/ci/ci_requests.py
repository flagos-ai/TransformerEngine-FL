"""Resolve authorized CI requests to platforms, a fixed ref and a test level."""

import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

PLATFORM_LIST = r"[a-z0-9_-]+(?:[ ,]+[a-z0-9_-]+)*"
TEST_LEVELS = frozenset(("auto", "L0", "L1", "L2", "L3", "L4"))


def select_platforms(requested, available):
    if not re.fullmatch(PLATFORM_LIST, requested):
        raise ValueError("Expected comma/space-separated platforms or 'all'")
    selected = set(re.split(r"[ ,]+", requested))
    if not available or selected - available - {"all"}:
        raise ValueError("Unknown platform or no configured platforms")
    return sorted(available if "all" in selected else selected)


def resolve(event, api, available):
    if event.get("action") != "created" or not event.get("issue", {}).get("pull_request"):
        raise ValueError("Expected a newly created PR comment")
    comment = event["comment"]
    match = re.fullmatch(r"/(?:ci run|test) ([^\r\n]+)", comment["body"].strip())
    if not match:
        raise ValueError("Expected /ci run <platforms|all> or /test <platforms|all>")
    number, requested, login = event["issue"]["number"], match[1], comment["user"]["login"]
    permission = api(f"collaborators/{quote(login, safe='')}/permission")["permission"]
    if permission not in {"write", "maintain", "admin"}:
        raise PermissionError("Requesting CI requires repository write permission")
    if number <= 0:
        raise ValueError("Expected a positive PR number")
    platforms = select_platforms(requested, available)
    pr = api(f"pulls/{number}")
    sha = pr["head"]["sha"]
    if pr["state"] != "open" or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected an open PR with a valid head SHA")
    return platforms, sha


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

    available = {
        p.stem
        for p in Path(".github/configs").glob("*.yml")
        if p.stem not in {"template", "image_sources", "platforms"}
    }
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        platforms, ref, level = resolve_dispatch(
            event.get("inputs") or {}, available, os.environ["GITHUB_SHA"]
        )
    else:
        (platforms, ref), level = resolve(event, api, available), "auto"
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"platforms={json.dumps(platforms)}\ncheckout_ref={ref}\ntest_level={level}\n")


if __name__ == "__main__":
    main()
