# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .register import get_backend, get_selected_backend, register_backend
from .logger import get_logger
logger = get_logger()

from .import_utils import have_flag_gems, have_native_backend

HAVE_FLAG_GEMS = have_flag_gems()
HAVE_NATIVE_BACKEND = have_native_backend()

class BackendDispatch:
    """
    Transformer Engine Backend that routes operations to appropriate implementations.
    
    Uses caching to avoid repeated flag checks and backend lookups for the same operation.
    """
    
    def __init__(self):
        """Initialize the backend with an empty implementation cache."""
        # Cache for operation implementations: {operation: impl}
        self._impl_cache: Dict[str, Any] = {}
    
    def _get_impl(self, operation: str):
        """
        Get the implementation for an operation based on flags.
        Falls back to native if the selected backend doesn't have the operation.
        Uses caching to avoid repeated lookups.

        Args:
            operation: Name of the operation (e.g., "gemm", "rmsnorm_fwd")

        Returns:
            The implementation function/class to use

        Raises:
            RuntimeError: If no backend has the operation
        """
        # Check cache first
        if operation in self._impl_cache:
            return self._impl_cache[operation]

        # Get selected backend based on global environment variable
        selected_backend = get_selected_backend()

        # Try to get implementation from selected backend
        impl = selected_backend.get(operation)

        # If not found in selected backend and native is available, try native
        if impl is None and HAVE_NATIVE_BACKEND:
            native_backend = get_backend("native")
            if native_backend is not None:
                logger.debug(
                    f"Backend '{selected_backend.name}' doesn't have '{operation}', "
                    f"falling back to native"
                )
                impl = native_backend.get(operation)

        if impl is None:
            available_ops = sorted(selected_backend._implementations.keys()) if selected_backend else []
            raise RuntimeError(
                f"Operation '{operation}' is not registered in any available backend. "
                f"Available operations in '{selected_backend.name if selected_backend else 'none'}': {available_ops}. "
                f"Native backend available: {HAVE_NATIVE_BACKEND}"
            )

        # Cache the implementation for future use
        logger.info(f"Backend '{selected_backend.name}' use implementation of '{operation}' for training")
        self._impl_cache[operation] = impl

        return impl

    def clear_cache(self):
        """Clear the implementation cache. Useful if flags change at runtime."""
        self._impl_cache.clear()
        logger.debug("Cleared implementation cache")

    def _fallback_to_native(self, operation: str, args, kwargs, error: Exception, trim_eps: bool = False):
        """
        Attempt to fallback to native backend for an operation.

        Args:
            operation: Name of the operation
            args: Original args
            kwargs: Original kwargs
            error: The original exception
            trim_eps: If True, removes the last argument (used for rmsnorm_bwd)

        Returns:
            Result from native backend if available

        Raises:
            The original error if native backend is not available
        """
        if not HAVE_NATIVE_BACKEND:
            logger.error(f"{operation} implementation failed and native backend is not available: {error}")
            raise error

        logger.warning(f"{operation} implementation failed, falling back to native: {error}")
        native_backend = get_backend("native")
        if native_backend is None:
            raise error

        native_impl = native_backend.get(operation)
        if native_impl is None:
            raise error

        if trim_eps:
            args = args[:-1]  # cut eps for rmsnorm_bwd
        return native_impl(*args, **kwargs)

    def gemm(self, *args, **kwargs):
        """GEMM operation with automatic fallback to native."""
        impl = self._get_impl("gemm")
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            return self._fallback_to_native("gemm", args, kwargs, e)

    def apply_normalization(self, *args, **kwargs):
        """Apply normalization with automatic fallback to native."""
        impl = self._get_impl("apply_normalization")
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            return self._fallback_to_native("apply_normalization", args, kwargs, e)

    def rmsnorm_fwd(self, *args, **kwargs):
        """RMSNorm forward pass with automatic fallback to native."""
        impl = self._get_impl("rmsnorm_fwd")
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            return self._fallback_to_native("rmsnorm_fwd", args, kwargs, e)

    def rmsnorm_bwd(self, *args, **kwargs):
        """RMSNorm backward pass with automatic fallback to native."""
        impl = self._get_impl("rmsnorm_bwd")
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            return self._fallback_to_native("rmsnorm_bwd", args, kwargs, e, trim_eps=True)

    def multi_tensor_adam(self):
        """Multi-tensor Adam optimizer with automatic fallback to native."""
        impl = self._get_impl("adam")
        try:
            return impl
        except Exception as e:
            if not HAVE_NATIVE_BACKEND:
                logger.error(f"Adam implementation failed and native backend is not available: {e}")
                raise e
            logger.warning(f"Adam implementation failed, falling back to native: {e}")
            native_backend = get_backend("native")
            return native_backend.get("adam")

    def flash_attention(self, *args, **kwargs):
        """Flash Attention with automatic fallback to native."""
        impl = self._get_impl("flash_attention")
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            return self._fallback_to_native("flash_attention", args, kwargs, e)


# Backend initialization state
_backends_initialized = False
_backend_instance = None

def _initialize_backends():
    """
    Initialize all backend registrations.
    This function is called automatically on first use.
    """
    global _backends_initialized, _backend_instance

    if _backends_initialized:
        return

    # Register native backend only if CUDA extensions are available
    if HAVE_NATIVE_BACKEND:
        from .backend_native import register_backend_native
        register_backend_native()
        logger.info("Native (CUDA) backend registered")
    else:
        logger.info("Skipping native backend registration (CUDA extensions not available)")

    # Register FL backend if flag_gems is available
    if HAVE_FLAG_GEMS:
        from .backend_fl import register_backend_fl
        register_backend_fl()
        logger.info("FL (Flag-Gems/Triton) backend registered")

    # Verify at least one backend is available
    if not HAVE_NATIVE_BACKEND and not HAVE_FLAG_GEMS:
        logger.warning(
            "No backends available! Neither native (CUDA) nor FL (Flag-Gems) backend is available. "
            "Install flag_gems for AMD/ROCm support, or rebuild with CUDA support."
        )

    _backend_instance = BackendDispatch()
    _backends_initialized = True

    logger.info("Backend system initialized successfully")

# Create backend instance on module import
_initialize_backends()
backend = _backend_instance
