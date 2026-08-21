"""Compatibility helpers for vendor Transformer Engine extensions."""

import warnings


_WARNED_OPERATIONS = set()


def _is_argument_mismatch(error):
    message = str(error)
    return any(
        marker in message
        for marker in (
            "incompatible function arguments",
            "positional arguments but",
            "required positional argument",
            "takes exactly",
            "expected at most",
        )
    )


def _warn_vendor_upgrade(vendor, operation):
    key = (vendor, operation)
    if key in _WARNED_OPERATIONS:
        return
    _WARNED_OPERATIONS.add(key)
    warnings.warn(
        f"{vendor} Transformer Engine does not yet provide the upstream v2.17 {operation} ABI; the"
        " vendor TE is being upgraded; FL will use a semantics-preserving compatibility path when"
        " available.",
        RuntimeWarning,
        stacklevel=4,
    )


def _legacy_arguments(vendor, operation, args):
    if operation == "generic_gemm":
        if vendor == "Enflame":
            return ((*args[:12], args[14], *args[18:]),)
        return ()
    if operation in {"group_quantize", "bgrad_group_quantize"}:
        if args[-1] is not None:
            return ()
        if vendor in {"Hygon", "MetaX", "Iluvatar"}:
            return (args[:2],)
        if vendor in {"MUSA", "Enflame"}:
            return (args[:-1],)
        return ()
    if operation == "clamped_swiglu":
        return (args[:-1],) if args[-1] == 1.0 else ()
    if operation == "clamped_dswiglu":
        return (args[:-1],) if args[-1] == 1.0 else ()
    if operation in {
        "fused_topk_with_score_function_fwd",
        "fused_score_for_moe_aux_loss_fwd",
    }:
        return (args[:-1],) if args[-1] == 0 else ()
    if operation == "fused_topk_with_score_function_bwd" and args[-1] == 0:
        grad_logits = args[3]
        return ((grad_logits.shape[0], grad_logits.shape[1], *args[:-1]),)
    if operation == "fused_score_for_moe_aux_loss_bwd":
        grad_logits = args[2]
        return ((grad_logits.shape[0], grad_logits.shape[1], *args),)
    return ()


class VendorTECompat:
    """Prefer the v2.17 ABI and fall back to the last FL ABI when equivalent."""

    _UPGRADED_OPERATIONS = {
        "generic_gemm",
        "group_quantize",
        "bgrad_group_quantize",
        "clamped_swiglu",
        "clamped_dswiglu",
        "fused_attn_fwd",
        "fused_attn_bwd",
        "fused_topk_with_score_function_fwd",
        "fused_topk_with_score_function_bwd",
        "fused_score_for_moe_aux_loss_fwd",
        "fused_score_for_moe_aux_loss_bwd",
    }

    def __init__(self, extension, vendor):
        self._extension = extension
        self._vendor = vendor

    def __getattr__(self, name):
        attribute = getattr(self._extension, name)
        if name not in self._UPGRADED_OPERATIONS or not callable(attribute):
            return attribute

        def call(*args, **kwargs):
            try:
                return attribute(*args, **kwargs)
            except TypeError as error:
                if kwargs or not _is_argument_mismatch(error):
                    raise
                legacy_candidates = _legacy_arguments(self._vendor, name, args)
                _warn_vendor_upgrade(self._vendor, name)
                if not legacy_candidates:
                    raise
                for legacy_args in legacy_candidates:
                    try:
                        return attribute(*legacy_args)
                    except TypeError as legacy_error:
                        if not _is_argument_mismatch(legacy_error):
                            raise
                raise

        return call


def vendor_te_compat(extension, vendor):
    """Wrap a vendor extension with the f031cf87-to-v2.17 ABI bridge."""
    return VendorTECompat(extension, vendor)
