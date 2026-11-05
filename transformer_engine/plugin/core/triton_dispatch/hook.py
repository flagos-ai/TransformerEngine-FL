# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Import hook that swaps TE's Triton APIs for dispatch entries.

Each API becomes an independent dispatch point, so a vendor implementation can
replace TE's Triton kernel for one function while the rest keep running TE's.

Replacement happens in a loader wrapper, after the module body defines the
functions and before any consumer reads them. A ``setattr`` pass after import
would be too late for the ``from X import fn`` consumers (``distributed.py``
and the three grouped_linear modules), which hold the original from import
time onwards.

Installed from ``plugin/core/__init__.py`` -- the one window where none of the
five target modules is loaded yet.
"""

from __future__ import annotations

import inspect
import sys
import threading
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Optional, Sequence

from ..manager import get_default_manager
from .entry import (
    ensure_native_registered,
    forget_natives,
    make_entry,
    remember_natives,
)
from .manifest import EXCLUDED_PUBLIC_FUNCTIONS, TARGET_APIS, TARGET_MODULES, TARGET_OPS


_PROCESSED_MARKER = "__te_fl_triton_dispatch_processed__"


class TritonDispatchLoader(Loader):
    """Loader wrapper that replaces selected Triton APIs after module execution."""

    def __init__(self, wrapped: Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: ModuleSpec):
        create_module = getattr(self._wrapped, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        # Defensive: the finder already rejects reload. This catches a reload
        # that reuses the module object without going through find_spec.
        if getattr(module, _PROCESSED_MARKER, False):
            raise ImportError(
                f"Reload is not supported for Triton dispatch module '{module.__name__}'"
            )

        exec_module = getattr(self._wrapped, "exec_module", None)
        if exec_module is None:
            raise ImportError(
                f"Loader for Triton dispatch module '{module.__name__}' has no exec_module()"
            )
        exec_module(module)

        # Order matters: the setattr pass runs only after registration has
        # succeeded. Otherwise a failure would leave the module exporting
        # entries whose op has no registered impl.
        replacements = _build_replacements(module)
        natives = {op_name: native for _, op_name, native, _ in replacements}
        remember_natives(natives)
        try:
            ensure_native_registered(get_default_manager())
        except Exception:
            forget_natives(natives)
            raise
        for api_name, _, _, entry in replacements:
            setattr(module, api_name, entry)
        setattr(module, _PROCESSED_MARKER, True)

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


def _public_module_functions(module: ModuleType) -> set[str]:
    """Names the hook considers replaceable, by name and origin only.

    ``isfunction`` excludes the ``@triton.jit`` kernels -- they are
    ``JITFunction`` instances, and wrapping one breaks the
    ``kernel[grid](...)`` subscript syntax. ``__module__`` excludes names
    imported into the module rather than defined by it.
    """
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and inspect.isfunction(value)
        and getattr(value, "__module__", None) == module.__name__
    }


def _build_replacements(module: ModuleType):
    """Validate the module against the manifest, then build its entries.

    Only names listed in the manifest are replaced; the scan is a cross-check.
    An unlisted public function raises rather than being hijacked unreviewed,
    since upstream may add a helper that is not an operator.
    """
    expected = set(TARGET_APIS[module.__name__])
    excluded = set(EXCLUDED_PUBLIC_FUNCTIONS.get(module.__name__, ()))
    discovered = _public_module_functions(module)
    if discovered != expected | excluded:
        missing = sorted(expected - discovered)
        unexpected = sorted(discovered - expected - excluded)
        raise ImportError(
            f"Triton API manifest mismatch for '{module.__name__}': "
            f"missing={missing}, unexpected={unexpected}"
        )

    replacements = []
    for api_name in TARGET_APIS[module.__name__]:
        native = getattr(module, api_name)
        op_name = TARGET_OPS[(module.__name__, api_name)]
        replacements.append((api_name, op_name, native, make_entry(op_name, native)))
    return replacements


class TritonDispatchFinder(MetaPathFinder):
    """Finder that wraps the loader for each manifest target module."""

    # Identifies an installed finder across module reloads of this file,
    # where an isinstance() check against a fresh class object would miss.
    _te_fl_triton_dispatch_finder = True

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        if fullname not in TARGET_MODULES:
            return None
        # Reject reload before delegating or running the original loader. A
        # reload defines fresh function objects, and "skip if already
        # captured" would pin te.native to the stale ones. Supporting it needs
        # replace/upsert semantics in the registry.
        if target is not None:
            raise ImportError(f"Reload is not supported for Triton dispatch module '{fullname}'")

        # Delegate to the finders after this one rather than calling
        # PathFinder directly: under an editable install the target lives
        # behind a custom finder and is not on sys.path, so hardcoding
        # PathFinder would break imports that work without the hook.
        for finder in _finders_after(self):
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is None or getattr(spec.loader, "exec_module", None) is None:
                raise ImportError(f"Loader for Triton dispatch module '{fullname}' is unsupported")
            if isinstance(spec.loader, TritonDispatchLoader):
                raise ImportError(f"Triton dispatch loader already installed for '{fullname}'")
            spec.loader = TritonDispatchLoader(spec.loader)
            return spec
        return None


def _finders_after(finder: MetaPathFinder):
    """Return the ``sys.meta_path`` entries after ``finder``, located by
    identity rather than equality."""
    finders = tuple(sys.meta_path)
    try:
        index = next(index for index, item in enumerate(finders) if item is finder)
    except StopIteration:
        return ()
    return finders[index + 1 :]


_install_lock = threading.RLock()


def install_finder() -> TritonDispatchFinder:
    """Install the finder, or return the one already installed.

    Raises if a target module is already in ``sys.modules``, which would mean
    hijacking only the not-yet-imported ones. That happens if upstream
    reorders the import chain reaching ``plugin.core``, so it is reported
    rather than left silent.
    """
    with _install_lock:
        existing = [
            finder
            for finder in sys.meta_path
            if getattr(finder, "_te_fl_triton_dispatch_finder", False)
        ]
        if existing:
            return existing[0]

        preloaded = sorted(name for name in TARGET_MODULES if name in sys.modules)
        if preloaded:
            raise ImportError(
                "Cannot install Triton dispatch after target modules were loaded: "
                + ", ".join(preloaded)
            )

        finder = TritonDispatchFinder()
        sys.meta_path.insert(0, finder)
        return finder
