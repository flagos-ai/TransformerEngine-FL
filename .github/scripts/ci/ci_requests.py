"""Resolve authorized PR comments to platforms and a fixed PR SHA."""

import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote


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
    if number <= 0 or not re.fullmatch(r"[a-z0-9_-]+(?:[ ,]+[a-z0-9_-]+)*", requested):
        raise ValueError("Expected a positive PR number and comma/space-separated platforms")
    selected = set(re.split(r"[ ,]+", requested))
    if not available or selected - available - {"all"}:
        raise ValueError("Unknown platform or no configured platforms")
    pr = api(f"pulls/{number}")
    sha = pr["head"]["sha"]
    if pr["state"] != "open" or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected an open PR with a valid head SHA")
    return sorted(available if "all" in selected else selected), sha


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
    platforms, sha = resolve(
        json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()), api, available
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"platforms={json.dumps(platforms)}\ncheckout_ref={sha}\n")


if __name__ == "__main__":
    main()
