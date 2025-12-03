# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import importlib
from typing import Any, Optional

# Safety import cache to avoid circular imports and improve performance
_import_cache: dict[str, Any] = {}


class _LazyImport:
    """Lazy import proxy that defers actual import until first use."""
    
    def __init__(self, module_path: str, name: Optional[str] = None):
        self._module_path = module_path
        self._name = name
        self._cache_key = f"{module_path}.{name}" if name else module_path
        self._imported = None
    
    def _import(self):
        """Perform the actual import."""
        if self._imported is None:
            if self._cache_key in _import_cache:
                self._imported = _import_cache[self._cache_key]
            else:
                module = importlib.import_module(self._module_path)
                if self._name:
                    self._imported = getattr(module, self._name)
                else:
                    self._imported = module
                _import_cache[self._cache_key] = self._imported
        return self._imported
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the imported object."""
        return getattr(self._import(), name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Allow calling if the imported object is callable."""
        return self._import()(*args, **kwargs)
    
    def __repr__(self) -> str:
        """String representation."""
        if self._imported is None:
            return f"<LazyImport: {self._cache_key} (not loaded)>"
        return repr(self._imported)


def safety_import(module_path: str, name: Optional[str] = None, lazy: bool = False) -> Any:
    """
    Safely import a module or attribute with lazy loading and caching.
    
    This function helps avoid circular imports by deferring imports until
    they are actually needed, and caches the result for performance.
    
    Args:
        module_path: Full module path
        name: Optional attribute name to import from the module (e.g., 'FLAttention')
              If None, returns the module itself.
        lazy: If True, returns a lazy proxy that defers import until first use.
              If False (default), imports immediately but caches the result.
              Use lazy=True when there's a risk of circular imports.
    
    Returns:
        The imported module or attribute (or a lazy proxy if lazy=True).
    """
    cache_key = f"{module_path}.{name}" if name else module_path
    
    if lazy:
        # Return lazy proxy that defers import
        return _LazyImport(module_path, name)
    
    # Immediate import with caching
    if cache_key not in _import_cache:
        module = importlib.import_module(module_path)
        if name:
            _import_cache[cache_key] = getattr(module, name)
        else:
            _import_cache[cache_key] = module
    
    return _import_cache[cache_key]