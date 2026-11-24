# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Driver for the Triton dispatch numerics cases.

The comparisons themselves live in ``triton_dispatch_numerics_cases.py`` and
run in a fresh interpreter. They allocate CUDA tensors, and sibling plugin
suites install ``flag_gems`` MagicMocks that make a CUDA graph capture fail
mid-flight; an aborted capture leaves the process unable to allocate at all
("Offset increment outside graph capture encountered unexpectedly"), which is
not recoverable by synchronize, empty_cache, or restoring generator state.
Running in a subprocess keeps these checks executing instead of skipping,
whatever ran before them.
"""

import sys

import pytest

from plugin.utils import REPOSITORY_ROOT, run_in_fresh_process, triton_device_unavailable_reason


CASES = "tests/plugin/plugin/triton_dispatch_numerics_cases.py"


def test_numerics_cases_pass_in_a_clean_process():
    unavailable = triton_device_unavailable_reason()
    if unavailable is not None:
        pytest.skip(unavailable)
    command = [sys.executable, "-m", "pytest", CASES, "-q", "-p", "no:randomly"]
    returncode = run_in_fresh_process(command, cwd=REPOSITORY_ROOT)

    assert (
        returncode == 0
    ), f"Triton dispatch numerics cases failed; rerun directly for detail:\n  {' '.join(command)}"
