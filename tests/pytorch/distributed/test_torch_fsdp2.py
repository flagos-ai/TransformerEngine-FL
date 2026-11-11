# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import shlex
import sys
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import run_distributed

import pytest
import torch

import transformer_engine.pytorch as te

NUM_PROCS: int = torch.cuda.device_count()
_FSDP2_DIR = Path(__file__).parent.resolve() / "fsdp2_tests"


def _nested_test_env(*, isolate_coverage: bool = False) -> dict[str, str]:
    """Optionally prevent outer pytest-cov settings from leaking into nested pytest."""
    env = os.environ.copy()
    if not isolate_coverage:
        return env

    for key in (
        "COVERAGE_FILE",
        "COVERAGE_PROCESS_START",
        "COVERAGE_RCFILE",
        "COV_CORE_SOURCE",
        "COV_CORE_CONFIG",
        "COV_CORE_DATAFILE",
    ):
        env.pop(key, None)

    pytest_addopts = shlex.split(env.get("PYTEST_ADDOPTS", ""))
    coverage_flags = {"--cov-branch", "--cov-append"}
    coverage_options_with_values = {"--cov-report", "--cov-config", "--cov-fail-under"}
    filtered_addopts = []
    index = 0
    while index < len(pytest_addopts):
        arg = pytest_addopts[index]
        if arg in coverage_flags:
            index += 1
            continue
        if arg in coverage_options_with_values:
            index += 2
            continue
        if any(arg.startswith(f"{option}=") for option in coverage_options_with_values):
            index += 1
            continue
        if arg == "--cov" or arg == "--no-cov" or arg.startswith("--cov="):
            index += 1
            if arg == "--cov" and index < len(pytest_addopts):
                if not pytest_addopts[index].startswith("-"):
                    index += 1
            continue
        filtered_addopts.append(arg)
        index += 1
    if filtered_addopts:
        env["PYTEST_ADDOPTS"] = shlex.join(filtered_addopts)
    else:
        env.pop("PYTEST_ADDOPTS", None)
    return env


# Import some utilities from PyTest-owned conftest.py.
sys.path.insert(0, str(_FSDP2_DIR))
from conftest import _parametrize_recipes

sys.path.pop(0)


@pytest.mark.skip(
    reason=(
        "Test fails with exitcode 3 in CI environment. "
        "Root cause: All FP8 recipes are skipped due to insufficient GPU compute capability, "
        "but torchrun multi-process pytest collection fails with internal error (exitcode 3) "
        "instead of gracefully handling all-skipped scenario. "
        "This is a known issue with nested pytest runs under torchrun when all tests are skipped."
    )
)
@pytest.mark.skipif(NUM_PROCS % 2 != 0, reason="Requires even number of GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
def test_fsdp2_model_tests():
    """All FSDP2 model tests (parametrized internally by recipe, fp8_init, sharding, layer)."""
    test_path = _FSDP2_DIR / "run_fsdp2_model.py"
    run_distributed(
        [
            "torchrun",
            f"--nproc_per_node={NUM_PROCS}",
            "--local-ranks-filter=0",
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "-s",
            "--tb=short",
        ],
        valid_returncodes=(0, 5),
        env=_nested_test_env(isolate_coverage=True),
        timeout=600,
    )


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
@pytest.mark.skip(
    reason="FSDP2 FusedAdam nested pytest exits with code 3 in the current CUDA CI environment"
)
def test_fsdp2_fused_adam_tests():
    """All FSDP2 FusedAdam tests (parametrized internally by recipe, test variant)."""
    test_path = _FSDP2_DIR / "run_fsdp2_fused_adam.py"
    nproc = min(NUM_PROCS, 2)
    run_distributed(
        [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "--local-ranks-filter=0",
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "-s",
            "--tb=short",
            # The following 2 tests need to be run in sequence,
            # as they depend on each other.
            "-k",
            "not dcp_resharding_save and not dcp_resharding_load",
        ],
        valid_returncodes=(0, 5),
        env=_nested_test_env(isolate_coverage=True),
        timeout=600,
    )


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
def test_fsdp2_mem_leak_tests():
    """Run FSDP2 memory leak tests."""
    test_path = _FSDP2_DIR / "run_fsdp2_mem_leak.py"
    nproc = min(NUM_PROCS, 2)
    result = subprocess.run(
        [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "--local-ranks-filter=0",
            "-m",
            "pytest",
            str(test_path),
            "-v",
            "-s",
            "--tb=short",
        ],
        env=_nested_test_env(isolate_coverage=True),
        timeout=600,
    )
    assert result.returncode in (0, 5), f"Inner pytest failed with exit code {result.returncode}"


@pytest.mark.skipif(NUM_PROCS < 4, reason="Requires 4+ GPUs for DP4→DP2 resharding test")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
@pytest.mark.parametrize("recipe", _parametrize_recipes())
def test_fsdp2_fused_adam_dcp_resharding(recipe):
    """DCP checkpoint saved with DP4 loads correctly into DP2 (cross-topology resharding).

    Runs two sequential torchrun invocations against run_fsdp2_fused_adam.py:
      1. nproc=4  →  dcp_resharding_save  (train + write checkpoint + ref output)
      2. nproc=2  →  dcp_resharding_load  (load checkpoint, assert output parity)
    """
    if recipe == "MXFP8BlockScaling":
        pytest.xfail(
            "MXFP8BlockScaling: FusedAdam CUDA kernel does not support "
            "MXFP8 quantized tensors, causing illegal memory access. "
            "Fixed by https://github.com/NVIDIA/TransformerEngine/pull/2789."
        )
    if recipe == "NVFP4BlockScaling":
        pytest.xfail(
            "NVFP4BlockScaling: DCP load_state_dict triggers reset_sharded_param() "
            "which calls data_ptr() on NVFP4Tensor wrapper subclass with invalid storage"
        )
    if recipe == "Float8BlockScaling":
        pytest.xfail(
            "Float8BlockScaling doesnt work for DCP resharding with scale inv padding "
            "not being handled correctly for slice ops"
        )

    test_path = _FSDP2_DIR / "run_fsdp2_fused_adam.py"

    # Phase 1: save checkpoint with 4 ranks.
    nested_env = _nested_test_env()
    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=4",
            "--local-ranks-filter=0",
            str(test_path),
            "--test",
            "dcp_resharding_save",
            "--recipe",
            recipe,
        ],
        env=nested_env,
        timeout=300,
    )
    assert result.returncode == 0, f"DCP resharding save phase failed: {result.returncode}"

    # Phase 2: load checkpoint with 2 ranks (different topology).
    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=2",
            "--local-ranks-filter=0",
            str(test_path),
            "--test",
            "dcp_resharding_load",
            "--recipe",
            recipe,
        ],
        env=nested_env,
        timeout=300,
    )
    assert result.returncode == 0, f"DCP resharding load phase failed: {result.returncode}"


def test_dummy() -> None:
    """Dummy test

    pytest returns exit code 5 if all tests are skipped.

    """
    pass
