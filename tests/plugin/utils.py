"""Small shared helpers for plugin tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def run_in_fresh_process(command: Sequence[str], *, cwd: Path = REPOSITORY_ROOT) -> int:
    """Run a test command without leaking imported plugin modules between suites."""
    return subprocess.run(list(command), cwd=cwd, check=False).returncode
