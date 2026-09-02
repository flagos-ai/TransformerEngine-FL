from __future__ import annotations

from transformer_engine.plugin.core import get_manager


_IMPLEMENTATION_PREFIXES = ("default.flagos", "reference.torch", "vendor.")


def available_implementations(op_name: str):
    """Return available FlagOS, Reference, and current-vendor implementations."""
    manager = get_manager()
    manager.ensure_initialized()
    implementations = [
        impl
        for impl in manager.registry.get_implementations(op_name)
        if impl.impl_id.startswith(_IMPLEMENTATION_PREFIXES) and impl.is_available()
    ]
    return sorted(implementations, key=lambda impl: impl.impl_id)
