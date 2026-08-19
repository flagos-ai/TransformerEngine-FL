#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Temporarily patch Megatron-LM-FL for the MUSA mccl integration test."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <MCORE_PATH>")

    mcore_path = Path(sys.argv[1])
    config = mcore_path / "megatron/training/config/common_config.py"
    text = config.read_text()
    old = '    distributed_backend: Literal["nccl", "gloo"] = "nccl"\n'
    new = '    distributed_backend: Literal["nccl", "gloo", "mccl"] = "mccl"\n'
    if old in text:
        config.write_text(text.replace(old, new, 1))
        print("Patched Megatron distributed_backend to accept mccl")
    elif new not in text:
        raise SystemExit("expected distributed_backend definition not found")
    else:
        print("Megatron distributed_backend already accepts mccl")

    platform_manager = mcore_path / "megatron/plugin/platform/platform_manager.py"
    text = platform_manager.read_text()
    old = """    if "cuda" in PLATFORMS.keys() and PLATFORMS["cuda"].is_available():
"""
    new = """    requested_platform = os.environ.get("PLATFORM", "").lower()
    if requested_platform in {"mthreads", "musa"}:
        if "musa" not in PLATFORMS or not PLATFORMS["musa"].is_available():
            raise ValueError("MUSA platform was requested but is not available")
        cur_platform = PLATFORMS["musa"]
        print("Megatron-LM-FL Platform: musa Selected")
        return cur_platform

    if "cuda" in PLATFORMS.keys() and PLATFORMS["cuda"].is_available():
"""
    if old in text:
        platform_manager.write_text(text.replace(old, new, 1))
        print("Patched Megatron platform selection to honor MUSA PLATFORM")
    elif new not in text:
        raise SystemExit("expected Megatron platform selection block not found")
    else:
        print("Megatron platform selection already honors MUSA PLATFORM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
