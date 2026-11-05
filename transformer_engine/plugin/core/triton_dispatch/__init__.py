# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import sys

from .hook import install_finder


def install() -> None:
    """Install Triton API dispatch interception."""
    install_finder()


def is_installed() -> bool:
    """Return whether this process has an installed Triton dispatch finder."""
    return any(getattr(finder, "_te_fl_triton_dispatch_finder", False) for finder in sys.meta_path)


__all__ = ["install", "is_installed"]
