# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Dispatch entries for the Triton APIs, plus registration of TE's originals.

``te.native`` does not come from ``register_ops.py`` like other impls: that
runs inside ``OpManager.ensure_initialized()``, which has already completed by
the time a ``pytorch/triton`` module loads. The import hook captures the
originals instead, and they are registered here on whichever manager is
current when an entry is first called.

Registration is lazy because the default manager is not a process-lifetime
object -- ``reset_default_manager()`` drops it and the Triton modules are not
re-imported, so a replacement manager would otherwise have no ``te.native``.
"""

from __future__ import annotations

import functools
import os
import threading
import weakref
from typing import Callable, Dict

from ..manager import OpManager, get_default_manager
from ..policy import get_policy, get_policy_epoch
from ..types import BackendImplKind, OpImpl


_NATIVE_IMPL_ID = "te.native"
_lock = threading.RLock()
_native_functions: Dict[str, Callable] = {}
# Bumped on every capture. The five target modules load in separate batches,
# so a manager registered against an older generation still needs the rest.
_native_generation = 0
# op_name -> (fn, native_gen, registry_gen, policy_gen, policy_fp). Keyed by
# str so the hot path stays Dynamo-traceable. The fingerprint is required, not
# redundant with policy_gen: a policy context is a contextvar (per-thread) but
# policy_gen is a single global counter, so two threads inside different
# contexts share an epoch and would otherwise read each other's selection.
_fast_path: Dict[str, tuple] = {}
# manager -> (native generation, registry generation) it is registered for.
# Weak so a manager dropped by reset_default_manager() can be collected.
_registered_managers: "weakref.WeakKeyDictionary[OpManager, tuple[int, int]]" = (
    weakref.WeakKeyDictionary()
)


def _reset_after_fork() -> None:
    # The captured natives stay valid -- the modules are still loaded in the
    # child, so the function objects are the same. Only the lock (which may
    # have been held at fork time) and the per-manager bookkeeping are reset,
    # forcing the child to re-verify registration against its own registry.
    global _lock, _registered_managers, _fast_path

    _lock = threading.RLock()
    _registered_managers = weakref.WeakKeyDictionary()
    # Required, not just hygiene: fork leaves the registry, the captured
    # natives and the global policy epoch untouched, so every field a cached
    # entry validates against still matches and the child would take the fast
    # path without ever re-verifying registration on its own manager.
    _fast_path = {}


try:
    os.register_at_fork(after_in_child=_reset_after_fork)
except AttributeError:
    pass


def remember_natives(natives: Dict[str, Callable]) -> None:
    """Remember native TE functions captured after a target module loads."""
    global _native_generation

    with _lock:
        duplicates = set(natives).intersection(_native_functions)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise RuntimeError(f"Triton native functions already captured: {names}")
        _native_functions.update(natives)
        _native_generation += 1


def ensure_native_registered(manager: OpManager) -> None:
    """Register every captured native function on ``manager``, once.

    Called on every dispatch, so the hit path is a single dict lookup. The
    slow path is serialised because ``register_impl`` rejects duplicate
    ``impl_id``s: idempotence comes from the lock and the explicit check
    below, so a genuine conflict still raises.
    """
    registry_generation = manager.registry.generation
    if _registered_managers.get(manager) == (_native_generation, registry_generation):
        return

    with _lock:
        # Re-read inside the lock: another thread may have registered while
        # this one waited, which also moves the registry generation.
        registry_generation = manager.registry.generation
        if _registered_managers.get(manager) == (
            _native_generation,
            registry_generation,
        ):
            return

        missing = []
        for op_name, native in _native_functions.items():
            existing = manager.registry.get_implementation(op_name, _NATIVE_IMPL_ID)
            if existing is None:
                # REFERENCE puts the TE original last in the default
                # flagos -> vendor -> reference order, so a vendor impl wins
                # whenever one is registered. It shares that tier with
                # reference.torch; the differing impl_id keeps them distinct.
                missing.append(
                    OpImpl(
                        op_name=op_name,
                        impl_id=_NATIVE_IMPL_ID,
                        kind=BackendImplKind.REFERENCE,
                        fn=native,
                    )
                )
            elif existing.fn is not native:
                raise RuntimeError(
                    f"Unexpected {_NATIVE_IMPL_ID} implementation already registered for "
                    f"op='{op_name}'"
                )

        # Atomic: register_many pre-checks the whole batch under the registry
        # lock, so a conflict cannot leave half the natives registered.
        manager.registry.register_many(missing)
        _registered_managers[manager] = (
            _native_generation,
            manager.registry.generation,
        )


def forget_natives(natives: Dict[str, Callable]) -> None:
    """Roll back a capture when module post-processing fails."""
    global _native_generation

    with _lock:
        for op_name, native in natives.items():
            if _native_functions.get(op_name) is not native:
                raise RuntimeError(f"Cannot roll back Triton native function '{op_name}'")
        for op_name in natives:
            del _native_functions[op_name]
        _native_generation += 1


def make_entry(op_name: str, native: Callable) -> Callable:
    """Build the function that replaces ``native`` in its TE module.

    Uses ``resolve()``, not ``call()``: ``call()`` retries on the next
    candidate when an impl raises, which would double-apply the APIs that
    mutate their inputs (``cross_entropy_forward`` rewrites ``_input`` in
    place; ``compute_grouped_dbias(_dscales)`` accumulates into
    ``dbias``/``dscales``) and return a plausible wrong number.

    ``functools.wraps`` keeps ``__name__`` and the signature, and sets
    ``__wrapped__`` so ``inspect.unwrap()`` recovers the original.
    """

    @functools.wraps(native)
    def entry(*args, **kwargs):
        manager = get_default_manager()
        cached = _fast_path.get(op_name)
        if cached is not None:
            fn, native_gen, registry_gen, policy_gen, policy_fp = cached
            if (
                native_gen == _native_generation
                and registry_gen == manager.registry.generation
                and policy_gen == get_policy_epoch()
                and policy_fp == get_policy().fingerprint()
            ):
                return fn(*args, **kwargs)
        ensure_native_registered(manager)
        fn = manager.resolve(op_name)
        _fast_path[op_name] = (
            fn,
            _native_generation,
            manager.registry.generation,
            get_policy_epoch(),
            get_policy().fingerprint(),
        )
        return fn(*args, **kwargs)

    return entry


def captured_natives() -> Dict[str, Callable]:
    """Return a copy of the captured native functions."""
    with _lock:
        return dict(_native_functions)
