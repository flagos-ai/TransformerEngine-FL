# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Opt-in function and class overrides using the existing TE-FL plugin policy."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

from .manager import get_default_manager
from .registry import OpRegistry
from .types import BackendImplKind, OpImpl

_T = TypeVar("_T")


def override(
    target: str,
    *,
    vendor: str | None = None,
    kind: BackendImplKind = BackendImplKind.VENDOR,
    impl_id: str | None = None,
    priority: int = 100,
    is_available: Callable[[], bool] | None = None,
    supports: Callable[..., bool] | None = None,
) -> Callable[[_T], _T]:
    """Describe an override without importing or initializing a backend manager.

    Args:
        target: Fully qualified module and qualified name of the decorated target.
        vendor: Vendor name used by the existing allow/deny policy, e.g. ``HYGON``.
        kind: Existing TE-FL implementation category.
        impl_id: Implementation ID, defaulting to ``vendor.<vendor>`` for vendors.
        priority: Priority within the existing policy order.
        is_available: Optional dynamic availability check, taking no arguments.
        supports: Optional predicate receiving the actual call arguments. A false
            result tries the next candidate or the original implementation.

    Register the returned implementation with ``register_overrides`` from the
    backend's existing ``register_builtins(registry)`` entry point.
    """
    if not isinstance(target, str) or not target:
        raise ValueError("An override requires a fully qualified target name")
    identifier = impl_id or (f"vendor.{vendor.lower()}" if vendor else str(kind))

    def decorate(implementation: _T) -> _T:
        if not callable(implementation):
            raise TypeError("An override must be a function or class")
        if target == f"{implementation.__module__}.{implementation.__qualname__}":
            raise ValueError("An implementation cannot override itself")
        if is_available is not None:
            implementation._is_available = (
                staticmethod(is_available) if inspect.isclass(implementation) else is_available
            )
        implementation.__tefl_supports__ = (
            staticmethod(supports) if inspect.isclass(implementation) and supports else supports
        )
        implementation.__tefl_override__ = OpImpl(
            op_name=target,
            impl_id=identifier,
            kind=kind,
            fn=implementation,
            vendor=vendor,
            priority=priority,
        )
        return implementation

    return decorate


def register_overrides(registry: OpRegistry, *implementations: Callable[..., Any]) -> None:
    """Register implementations described by ``@override`` in a supplied registry."""
    for implementation in implementations:
        spec = getattr(implementation, "__tefl_override__", None)
        if spec is None or spec.fn is not implementation:
            raise TypeError("register_overrides expects implementations decorated with @override")
        registry.register_impl(spec)


def _select_override(original: Callable[..., Any], args: tuple, kwargs: dict):
    key = f"{original.__module__}.{original.__qualname__}"
    candidates = get_default_manager().resolve_candidates(key, optional=True)
    for candidate in candidates:
        implementation = candidate.fn
        supports = getattr(implementation, "__tefl_supports__", None)
        if supports is not None and not supports(*args, **kwargs):
            continue
        if inspect.isclass(implementation) != inspect.isclass(original):
            raise TypeError(f"Override '{candidate.impl_id}' has the wrong target kind for '{key}'")
        if implementation is original or inspect.unwrap(implementation) is original:
            raise ValueError(f"Recursive override for '{key}'")
        return implementation
    return None


def _overridable_class(original: type) -> type:
    # Use the original metaclass so custom metaclasses and ordinary constructors
    # retain their initialization behavior. In particular, do not allocate an
    # uninitialized vendor instance from __new__.
    class OverrideMeta(type(original)):
        def __call__(cls, *args, **kwargs):
            if cls is proxy:
                implementation = _select_override(original, args, kwargs)
                if implementation is not None:
                    return implementation(*args, **kwargs)
            return super().__call__(*args, **kwargs)

    proxy = OverrideMeta(original.__name__, (original,), {"__module__": original.__module__})
    functools.update_wrapper(proxy, original, updated=())

    import torch

    if issubclass(original, torch.autograd.Function):
        # Dispatch the whole Function at .apply(), not its forward/backward
        # independently. PyTorch then pins the selected backward to this graph.
        @classmethod
        def apply(cls, *args, **kwargs):
            if cls is proxy:
                implementation = _select_override(original, args, kwargs)
                if implementation is not None:
                    if not issubclass(implementation, torch.autograd.Function):
                        raise TypeError("An autograd Function override must also be a Function")
                    return implementation.apply(*args, **kwargs)
            return super(proxy, cls).apply(*args, **kwargs)

        proxy.apply = apply
    return proxy


def overridable(target: _T) -> _T:
    """Enable optional plugin overrides for a function, method or class.

    The default key is ``<module>.<qualname>``. No matching implementation means
    the original runs. Selection follows the existing TE-FL policy; availability
    and input predicates are checked on every call. Selected implementation
    failures propagate instead of silently retrying another backend.

    Function signatures are exposed through ``__wrapped__``. Ordinary classes
    dispatch construction; autograd Functions dispatch ``apply`` as a unit.
    Subclasses of a decorated class keep their own construction/apply behavior.
    """
    if isinstance(target, staticmethod):
        return staticmethod(overridable(target.__func__))
    if isinstance(target, classmethod):
        return classmethod(overridable(target.__func__))
    if inspect.isclass(target):
        return _overridable_class(target)
    if not inspect.isfunction(target):
        raise TypeError("@overridable expects a function, method or class")

    if inspect.iscoroutinefunction(target):

        @functools.wraps(target)
        async def async_wrapper(*args, **kwargs):
            implementation = _select_override(target, args, kwargs)
            return await (implementation or target)(*args, **kwargs)

        return async_wrapper

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        implementation = _select_override(target, args, kwargs)
        return (implementation or target)(*args, **kwargs)

    return wrapper
