# Copyright (c) 2025, BAAI. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""AscendC operator wrappers and runtime validation for GDN."""

from __future__ import annotations

# API version for compatibility checking
GDN_API_VERSION = 1
GDN_PROVIDER = "te_npu_vendor"

# Required AscendC operators for complete GDN implementation
_REQUIRED_ASCENDC_OPS = (
    "chunk_fwd_o",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_gated_delta_rule_bwd_dhu",
    "chunk_bwd_dqkwg",
    "chunk_bwd_dv_local",
    "prepare_wy_repr_bwd_da",
    "prepare_wy_repr_bwd_full",
    "recompute_w_u_fwd",
    "solve_tri",
)


def validate_runtime() -> dict:
    """Validate that all required operators are loaded.

    Returns:
        dict: Runtime information including API version and provider

    Raises:
        RuntimeError: If any required operators are missing
    """
    try:
        import fla_npu.ops.ascendc as ascendc
    except ImportError as e:
        raise RuntimeError(f"fla_npu.ops.ascendc not available: {e}") from e

    missing = []
    for op_name in _REQUIRED_ASCENDC_OPS:
        if not callable(getattr(ascendc, op_name, None)):
            missing.append(op_name)

    if missing:
        raise RuntimeError(
            f"GDN runtime incomplete, missing operators in fla_npu.ops.ascendc: {', '.join(missing)}\n"
            f"Please ensure fla_npu is properly installed with all GDN operators."
        )

    import fla_npu.ops.triton as triton

    required_triton = (
        "autocast_custom_bwd",
        "autocast_custom_fwd",
        "chunk_local_cumsum",
        "chunk_scaled_dot_kkt_fwd",
        "input_guard",
        "l2norm_bwd",
        "l2norm_fwd",
        "solve_tril_npu",
    )
    missing = [name for name in required_triton if not callable(getattr(triton, name, None))]
    if missing:
        raise RuntimeError(f"GDN runtime incomplete, missing Triton operators: {missing}")

    return {
        "api_version": GDN_API_VERSION,
        "provider": GDN_PROVIDER,
        "operators": list(_REQUIRED_ASCENDC_OPS),
    }
