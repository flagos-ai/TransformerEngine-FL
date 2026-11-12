# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Source-level launch conventions that keep TE's Triton kernels portable.

Vendor Triton forks (MACA, CoreX, FlagTree) lag upstream in a few launch
APIs. These checks parse the sources instead of compiling, so they run on
every platform, including ones without a Triton driver, and fail as soon as
an upstream sync reintroduces a non-portable form.
"""

from __future__ import annotations

import ast
from pathlib import Path

KERNEL_DIR = Path("transformer_engine/common/triton")
LAUNCHER_DIR = Path("transformer_engine/pytorch/triton")


def _reset_to_zero_kernels(root: Path) -> dict[str, tuple[list[str], list[str]]]:
    kernels = {}
    for path in sorted((root / KERNEL_DIR).glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and getattr(decorator.func, "attr", None) == "autotune"
                ):
                    continue
                for keyword in decorator.keywords:
                    if keyword.arg == "reset_to_zero":
                        reset = [ast.literal_eval(item) for item in keyword.value.elts]
                        kernels[node.name] = ([arg.arg for arg in node.args.args], reset)
    return kernels


def test_reset_to_zero_arguments_are_passed_positionally(repository_root):
    """Older Triton autotuners zero ``args[reset_idx]`` from positional args only.

    MACA Triton raised ``IndexError: tuple index out of range`` in
    ``Autotuner._pre_hook`` because every mHC launch used keywords, leaving
    ``args == ()``. Positional passing works on every Triton version.
    """
    kernels = _reset_to_zero_kernels(repository_root)
    assert kernels, "expected autotuned kernels with reset_to_zero"
    launches = 0
    for path in sorted((repository_root / LAUNCHER_DIR).glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in kernels
            ):
                continue
            launches += 1
            signature, reset = kernels[node.func.value.id]
            positional = signature[: len(node.args)]
            missing = [name for name in reset if name not in positional]
            assert not missing, (
                f"{path.name}:{node.lineno} launches {node.func.value.id} with "
                f"reset_to_zero arguments {missing} by keyword"
            )
    assert launches >= len(kernels)


def test_permutation_dims_are_passed_as_varargs(repository_root):
    """``tl.trans(x, (1, 0))`` relies on upstream unwrapping a dims tuple.

    CoreX Triton forwards the tuple as a single dimension, raising
    ``ValueError: permute dims must have the same length as input shape``.
    ``tl.trans(x, 1, 0)`` is the same permutation on every Triton version.
    """
    offenders = []
    for path in sorted((repository_root / KERNEL_DIR).glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("trans", "permute")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tl"
                and len(node.args) == 2
                and isinstance(node.args[1], (ast.Tuple, ast.List))
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"tuple-form tl.trans/tl.permute dims: {offenders}"


def test_kernels_call_assume_through_the_compat_alias(repository_root):
    """``tl.assume`` is missing from older Triton builds (MACA Triton 3.0).

    It only feeds optimisation hints, so kernels call a module-level alias
    that is ``tl.assume`` when available and a no-op otherwise.
    """
    offenders = []
    for path in sorted((repository_root / KERNEL_DIR).glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "assume"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tl"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"direct tl.assume calls: {offenders}"
