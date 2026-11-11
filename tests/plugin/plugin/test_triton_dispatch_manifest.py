# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Manifest integrity checks for the Triton dispatch API list.

The manifest is maintained by hand (there is no module boundary to hijack
wholesale, unlike ``transformer_engine_torch``), so a missing entry would
otherwise be a silent gap. ``test_manifest_matches_source_public_functions``
is the check that makes an upstream-added public function a hard failure: it
re-derives the public function list from the Triton sources with the same
three rules the import hook applies, and diffs it against the constants.
"""

import ast
from pathlib import Path

import pytest

from transformer_engine.plugin.core.triton_dispatch.manifest import (
    EXCLUDED_PUBLIC_FUNCTIONS,
    TARGET_APIS,
    TARGET_MODULES,
    TARGET_OPS,
)


_TRITON_PACKAGE = "transformer_engine/pytorch/triton"


def _module_path(repository_root: Path, module_name: str) -> Path:
    return repository_root / f"{module_name.replace('.', '/')}.py"


# Decorators that replace the function object with something
# ``inspect.isfunction`` rejects, so the import hook never sees a function to
# wrap. ``triton.jit``/``autotune``/``heuristics`` all produce ``JITFunction``
# or ``Autotuner`` instances. Every other decorator returns a function and is
# therefore still a replacement target -- matching on the decorator's
# attribute name keeps this list the only approximation in the check.
_NON_FUNCTION_DECORATORS = frozenset({"jit", "autotune", "heuristics"})


def _decorator_name(node: ast.expr) -> str:
    """Return the attribute/callee name of a decorator expression."""
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _source_public_functions(path: Path) -> set[str]:
    """Enumerate public functions the import hook would consider replaceable.

    Mirrors ``hook._public_module_functions`` on the AST instead of a live
    module: top-level ``def`` only (so ``__module__`` equals the module), no
    leading underscore, and excluding only the decorators that turn a ``def``
    into a non-function object. A plain decorator such as ``@torch.no_grad()``
    still yields a function, so such a def must stay visible here -- otherwise
    an upstream addition passes CI and only fails at import time, where
    ``transformer_engine/__init__.py`` swallows the ImportError.
    """
    tree = ast.parse(path.read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
        and not any(
            _decorator_name(decorator) in _NON_FUNCTION_DECORATORS
            for decorator in node.decorator_list
        )
    }


def test_manifest_contains_five_modules_and_sixteen_apis():
    assert len(TARGET_MODULES) == 5
    assert sum(map(len, TARGET_APIS.values())) == 16
    assert len(TARGET_OPS) == 16
    assert len(set(TARGET_OPS.values())) == 16


def test_manifest_op_names_are_prefixed():
    for (module_name, api_name), op_name in TARGET_OPS.items():
        assert module_name in TARGET_MODULES
        assert api_name in TARGET_APIS[module_name]
        assert op_name == f"triton_{api_name}"


def test_check_deterministic_is_explicitly_excluded():
    module_name = "transformer_engine.pytorch.triton.mhc"
    assert EXCLUDED_PUBLIC_FUNCTIONS[module_name] == frozenset({"check_deterministic"})
    assert "check_deterministic" not in TARGET_APIS[module_name]


def test_manifest_covers_every_triton_source_module(repository_root):
    """A new ``pytorch/triton/*.py`` module must be added to the manifest."""
    package = repository_root / _TRITON_PACKAGE
    on_disk = {
        f"transformer_engine.pytorch.triton.{path.stem}"
        for path in package.glob("*.py")
        if path.stem != "__init__"
    }

    assert on_disk == set(TARGET_MODULES), (
        "Triton source modules and manifest disagree: "
        f"missing={sorted(on_disk - set(TARGET_MODULES))}, "
        f"unknown={sorted(set(TARGET_MODULES) - on_disk)}"
    )


@pytest.mark.parametrize("module_name", sorted(TARGET_MODULES))
def test_manifest_matches_source_public_functions(repository_root, module_name):
    """Every public function is either dispatched or explicitly excluded.

    This is the upstream-sync guard: adding a public function to a Triton
    module without listing it fails here, instead of surfacing later as an
    import-time error that ``transformer_engine/__init__.py`` swallows.
    """
    discovered = _source_public_functions(_module_path(repository_root, module_name))
    declared = set(TARGET_APIS[module_name])
    excluded = set(EXCLUDED_PUBLIC_FUNCTIONS.get(module_name, ()))

    assert discovered == declared | excluded, (
        f"Triton API manifest is stale for '{module_name}': "
        f"unlisted={sorted(discovered - declared - excluded)}, "
        f"absent={sorted((declared | excluded) - discovered)}. "
        "Add new APIs to TARGET_APIS, or to EXCLUDED_PUBLIC_FUNCTIONS if the "
        "function is not an operator."
    )


def test_excluded_functions_are_declared_for_known_modules():
    assert set(EXCLUDED_PUBLIC_FUNCTIONS).issubset(set(TARGET_MODULES))
    for module_name, excluded in EXCLUDED_PUBLIC_FUNCTIONS.items():
        assert not excluded & set(TARGET_APIS[module_name])
