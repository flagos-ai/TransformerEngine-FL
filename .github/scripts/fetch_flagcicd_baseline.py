#!/usr/bin/env python3
"""Fetch the newest successful main-branch FlagCICD report artifact.

This helper intentionally uses the GitHub REST API instead of the ``gh`` CLI.
The shared test workflow runs inside vendor images, which do not consistently
ship with GitHub CLI installed.

Exit codes:
  0: artifact downloaded; stdout contains the source run id
  2: no matching artifact exists
  3: GitHub API or archive error
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


API_ROOT = "https://api.github.com"


class GitHubApiError(RuntimeError):
    """An authenticated GitHub API request failed."""


def request(url: str, token: str) -> bytes:
    request_obj = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "flagcicd-baseline-fetcher",
        },
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise GitHubApiError(f"GitHub API returned HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise GitHubApiError(f"GitHub API request failed: {exc.reason}") from exc


def get_json(url: str, token: str) -> dict[str, Any]:
    payload = json.loads(request(url, token).decode("utf-8"))
    if not isinstance(payload, dict):
        raise GitHubApiError(f"unexpected JSON response from {url}")
    return payload


def safe_extract(archive: bytes, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
        for member in zip_file.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise GitHubApiError(f"unsafe artifact member path: {member.filename}")
            zip_file.extract(member, root)


def fetch(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise GitHubApiError("GH_TOKEN or GITHUB_TOKEN is required")

    repository = args.repository.strip("/")
    encoded_repository = "/".join(
        urllib.parse.quote(part, safe="") for part in repository.split("/")
    )
    query = urllib.parse.urlencode(
        {"branch": args.branch, "status": "completed", "per_page": str(args.limit)}
    )
    runs_url = f"{API_ROOT}/repos/{encoded_repository}/actions/runs?{query}"
    runs = get_json(runs_url, token).get("workflow_runs", [])
    if not isinstance(runs, list):
        raise GitHubApiError("workflow_runs is not an array")

    for run in runs:
        if not isinstance(run, dict) or run.get("conclusion") != "success":
            continue
        run_id = run.get("id")
        artifacts_url = run.get("artifacts_url")
        if not run_id or not artifacts_url:
            continue
        artifact_query = "per_page=100"
        separator = "&" if "?" in artifacts_url else "?"
        artifacts = get_json(f"{artifacts_url}{separator}{artifact_query}", token).get(
            "artifacts", []
        )
        if not isinstance(artifacts, list):
            continue
        artifact = next(
            (
                item
                for item in artifacts
                if isinstance(item, dict)
                and item.get("name") == args.artifact_name
                and not item.get("expired", False)
            ),
            None,
        )
        if artifact is None:
            continue
        archive_url = artifact.get("archive_download_url")
        if not archive_url:
            raise GitHubApiError(f"artifact {args.artifact_name} has no download URL")
        safe_extract(request(archive_url, token), args.output_dir)
        print(run_id)
        print(f"Downloaded {args.artifact_name} from successful main run {run_id}", file=sys.stderr)
        return 0

    print(
        f"No non-expired artifact named {args.artifact_name!r} found on {args.branch}",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token", default="")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    try:
        return fetch(args)
    except (GitHubApiError, json.JSONDecodeError, OSError, zipfile.BadZipFile) as exc:
        print(f"Baseline fetch failed: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
