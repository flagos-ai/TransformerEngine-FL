"""Small shared helpers for plugin tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def run_in_fresh_process(command: Sequence[str], *, cwd: Path = REPOSITORY_ROOT) -> int:
    """Run a test command without leaking imported plugin modules between suites."""
    return subprocess.run(list(command), cwd=cwd, check=False).returncode


def triton_device_unavailable_reason() -> str | None:
    """Return why TE's Triton kernels cannot launch here, or None if they can.

    ``torch.cuda.is_available()`` alone is not a capability check: device
    shims such as torch_gcu's transfer_to_gcu report True while the installed
    Triton only has the NVIDIA backend and no libcuda, so every launch would
    fail. Probe the active Triton driver the same way a kernel launch does.
    """
    import torch

    if not torch.cuda.is_available():
        return "CUDA is required"
    try:
        from triton.runtime import driver

        driver.active.get_current_device()
    except Exception as error:  # pylint: disable=broad-except
        return f"Triton driver unavailable: {type(error).__name__}: {error}"
    return None
