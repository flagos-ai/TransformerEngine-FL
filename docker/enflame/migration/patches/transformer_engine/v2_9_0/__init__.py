"""Enflame migration extension bridge for the TE-FL 2.17 image.

The bundled GCU extension is from the vendor's TE 2.9 migration package. Its
scale operator has no GCU implementation, so scale uses the TE-FL PyTorch
reference implementation. Other operations are forwarded to the vendor module.
"""

import ctypes
import importlib.util
from pathlib import Path
import warnings

_LIB = Path(__file__).parent / "lib"
ctypes.CDLL(str(_LIB / "libtransformer_engine.so"), mode=ctypes.RTLD_GLOBAL)
_spec = importlib.util.spec_from_file_location(
    "transformer_engine_torch_gcu",
    next(_LIB.glob("transformer_engine_torch_gcu*.so")),
)
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)
__version__ = "2.9.0+tefl.scale.compat"
_warned = False


def __getattr__(name):
    return getattr(_native, name)


def multi_tensor_compute_scale_and_scale_inv(
    chunk_size, noop_flag, tensor_lists, max_fp8, force_pow_2_scales, epsilon
):
    """Use TE-FL's reference implementation for the unsupported vendor GCU op."""
    global _warned
    if not _warned:
        warnings.warn(
            "Enflame vendor compute_scale has no GCU implementation; using the "
            "TE-FL PyTorch reference implementation. CPU fallback may execute "
            "unsupported GCU operations.",
            RuntimeWarning,
            stacklevel=2,
        )
        _warned = True
    from transformer_engine.plugin.core.backends.reference.impl.optimizer import (
        multi_tensor_compute_scale_and_scale_inv_torch,
    )

    return multi_tensor_compute_scale_and_scale_inv_torch(
        chunk_size, noop_flag, tensor_lists, max_fp8, force_pow_2_scales, epsilon
    )
