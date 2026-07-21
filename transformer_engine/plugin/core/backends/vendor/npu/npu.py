# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""NPU vendor backend for TE-FL plugin system.

Bridges Ascend NPU operations into the TE-FL unified plugin interface
by delegating to transformer_engine_npu (pip-installed from TransformerEngineNPU).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import os

import torch

from ....ops import TEFLBackendBase, NVTE_Fused_Attn_Backend, DType
from .flash_attention import NPUFlashAttention


_DTYPE_TO_TORCH = {
    0: torch.uint8,
    2: torch.int32,
    4: torch.float32,
    5: torch.float16,
    6: torch.bfloat16,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}


def _to_torch_dtype(dtype: Any) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype

    value = getattr(dtype, "value", dtype)
    try:
        return _DTYPE_TO_TORCH.get(int(value))
    except (TypeError, ValueError):
        return None


def _check_npu_available() -> bool:
    """Check if NPU hardware and torch_npu are available."""
    try:
        import torch_npu  # noqa: F401
        import transformer_engine_npu

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def _ensure_npu_libs():
    """Ensure torch_npu is imported (activates NPU device support in PyTorch)."""
    import torch_npu  # noqa: F401


def _get_torch_npu():
    """Lazy import of torch_npu."""
    _ensure_npu_libs()
    import torch_npu

    return torch_npu


def _get_tenpu_optimizers():
    """Get optimizers subpackage directly, bypassing transformer_engine_npu/__init__.py
    which triggers circular imports via pytorch/__init__.py -> module -> ops."""
    import transformer_engine_npu

    return transformer_engine_npu.pytorch.optimizers


def _get_tenpu_gemm():
    """Get GEMM ops subpackage."""
    _ensure_npu_libs()
    import transformer_engine_npu

    return transformer_engine_npu.pytorch.ops.gemm


def _get_tenpu_activations():
    """Get NPU activation functions (uses NPU-optimized kernels like npu_gelu, npu_swiglu)."""
    _ensure_npu_libs()
    from transformer_engine_npu.pytorch.ops.basic import npu_activation

    return npu_activation


class NPUBackend(TEFLBackendBase):
    """NPU backend delegating to transformer_engine_npu + torch_npu."""

    def is_available(self) -> bool:
        return _check_npu_available()

    # ===================== Attention =====================

    def get_attention_backend(self, attention_params=None):
        """Return NPU attention backend selection as a 6-tuple.

        The caller (dot_product_attention.py) expects:
            (use_flash_attention, flash_attention_backend,
             use_fused_attention, fused_attention_backend,
             use_unfused_attention, available_backends)
        TransformerEngineNPU only supports FlashAttention backend
        """
        from packaging.version import Version as PkgVersion
        from ....logger_manager import get_logger

        logger = get_logger()

        # Read environment variables to determine which backends to enable
        use_flash_attention = 1
        use_fused_attention = 0
        use_unfused_attention = 0

        # Log disabled backends
        logger.info_once("TransformerEngineNPU only supports FlashAttentionNPU backend")

        # Ascend only supports FlashAttention backend, and the FlashAttention version cannot be specified.
        flash_attention_backend = 0
        fused_attention_backend = NVTE_Fused_Attn_Backend.NVTE_No_Backend

        available_backends = [use_flash_attention, use_fused_attention, use_unfused_attention]

        return (
            use_flash_attention,
            flash_attention_backend,
            use_fused_attention,
            fused_attention_backend,
            use_unfused_attention,
            available_backends,
        )

    def get_flash_attention_class(self):
        """Return FlashAttention adapter class for NPU.

        Returns the adapter that bridges TE-FL's calling convention
        to NPU's FlashAttention interface.
        """
        return NPUFlashAttention

    # ===================== RMSNorm =====================

    def rmsnorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        ln_out: Any,
        quantizer: Any,
        otype: Any,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """RMSNorm forward using torch_npu.npu_rms_norm.

        TE-FL calls with: (input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma)
        NPU kernel: npu_rms_norm(input, gamma, epsilon=eps) → (output, rstd)

        NPU kernel requires 2D input [outer_dim, inner_dim]. We reshape accordingly.
        We ignore ln_out (pre-allocated output buffer), otype, sm_margin.
        """
        torch_npu = _get_torch_npu()

        if zero_centered_gamma:
            weight = weight + 1

        # NPU npu_rms_norm requires 2D input: [outer_dim, hidden_size]
        input_shape = input.shape
        inner_dim = weight.shape[0]
        x_2d = input.reshape(-1, inner_dim)

        out_2d, inv_rms = torch_npu.npu_rms_norm(x_2d, weight, epsilon=eps)

        # Reshape output back to original input shape
        out = out_2d.reshape(input_shape)

        if quantizer is not None and hasattr(quantizer, "quantize"):
            out = quantizer.quantize(out)

        # TE-FL expects (ln_out, mu, rsigma); mu is None for RMSNorm
        # inv_rms shape is [outer_dim, 1] from NPU kernel
        return out, None, inv_rms

    def rmsnorm_bwd(
        self,
        dz: torch.Tensor,
        x: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int,
        zero_centered_gamma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm backward using torch_npu.npu_rms_norm_backward.

        TE-FL calls with: (dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
        NPU kernel expects: npu_rms_norm_backward(dy, x, gamma, rstd)
        where rstd must be FP32 and x/dy must be 2D [outer_dim, inner_dim].

        NPU supported combo (BF16):
          dy(BF16) x(BF16) rstd(FP32) gamma(BF16) → dx(BF16) dgamma(FP32)
        """
        torch_npu = _get_torch_npu()

        if zero_centered_gamma:
            gamma = gamma + 1

        # NPU kernel requires 2D input
        input_shape = x.shape
        inner_dim = gamma.shape[0]
        x_2d = x.reshape(-1, inner_dim)
        dz_2d = dz.reshape(-1, inner_dim)

        # NPU kernel requires rstd in float32
        rsigma_fp32 = rsigma.float() if rsigma.dtype != torch.float32 else rsigma

        dx_2d, dw = torch_npu.npu_rms_norm_backward(dz_2d, x_2d, gamma, rsigma_fp32)

        # Reshape dx back to original input shape
        dx = dx_2d.reshape(input_shape)

        return dx, dw

    # ===================== Multi-tensor Optimizers =====================

    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ):
        """Multi-tensor scale."""
        opt = _get_tenpu_optimizers()
        opt.multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Multi-tensor L2 norm."""
        opt = _get_tenpu_optimizers()
        return opt.multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor)

    def multi_tensor_unscale_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        inv_scale: torch.Tensor,
        per_tensor: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Multi-tensor unscale + L2 norm."""
        opt = _get_tenpu_optimizers()
        return opt.multi_tensor_unscale_l2norm(
            chunk_size, noop_flag, tensor_lists, inv_scale, per_tensor
        )

    def multi_tensor_compute_scale_and_scale_inv(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        max_fp8: float,
        force_pow_2_scales: bool,
        epsilon: float,
    ):
        """Compute per-tensor FP8 scale and scale_inv."""
        if noop_flag.numel() > 0 and bool(noop_flag.item()):
            return

        opt = _get_tenpu_optimizers()
        opt.multi_tensor_compute_scale_and_scale_inv(
            chunk_size, noop_flag, tensor_lists, max_fp8, force_pow_2_scales, epsilon
        )

    def multi_tensor_compute_scale_inv_e8m0(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        block_len: int,
    ):
        """Compute scale_inv in e8m0 format for MXFP8."""
        opt = _get_tenpu_optimizers()
        opt.multi_tensor_compute_scale_inv_e8m0(chunk_size, noop_flag, tensor_lists)

    # ===================== GEMM =====================

    def generic_gemm(
        self,
        A: Any,
        transA: bool,
        B: Any,
        transB: bool,
        D: Any,
        quantizer: Any,
        output_dtype: Optional[Any],
        bias: Optional[torch.Tensor],
        bias_type: Any,
        gelu: bool,
        gelu_in: Optional[torch.Tensor],
        grad: bool,
        workspace: torch.Tensor,
        workspace_size: int,
        accumulate: bool,
        use_split_accumulator: bool,
        comm_overlap: Optional[Any] = None,
        comm_type: Optional[Any] = None,
        extra_output: Optional[torch.Tensor] = None,
        bulk_overlap: bool = False,
        alpha: float = 1.0,
        beta: Optional[float] = None,
    ) -> List[Any]:
        """General GEMM aligned with the generic_gemm interface.

        Computes out = B_comp @ A_comp (same as reference impl), where:
          B_comp = B.T if transB else B
          A_comp = A.T if transA else A

        Delegates to TransformerEngineNPU's general_gemm which computes:
          out = matmul(NPU_A, NPU_B) with usage-based transposition.

        Mapping: NPU_A=B, NPU_B=A, usage_a reflects transB, usage_b reflects transA.
        """
        import torch.nn.functional as F

        gemm_mod = _get_tenpu_gemm()

        # Map transA/transB to NPU TensorUsage strings
        # NPU general_gemm(A, B, usage_a, usage_b): transposes A if usage_a in USAGE_WITH_TRANS
        # We pass (B, A) as (NPU_A, NPU_B) so that NPU computes B_comp @ A_comp
        usage_a = "LT" if transB else "LN"  # controls transpose of NPU_A (which is our B)
        usage_b = "RT" if transA else "RN"  # controls transpose of NPU_B (which is our A)

        # Determine output dtype
        from ....ops import DType

        _DTYPE_TO_TORCH = {
            0: torch.uint8,
            2: torch.int32,
            4: torch.float32,
            5: torch.float16,
            6: torch.bfloat16,
            7: torch.float8_e4m3fn,
            8: torch.float8_e5m2,
        }
        torch_out_dtype = None
        if output_dtype is not None:
            if isinstance(output_dtype, torch.dtype):
                torch_out_dtype = output_dtype
            elif isinstance(output_dtype, int):
                torch_out_dtype = _DTYPE_TO_TORCH.get(output_dtype, None)
            elif hasattr(output_dtype, "value"):
                torch_out_dtype = _DTYPE_TO_TORCH.get(output_dtype.value, None)

        # Use the activation dtype of B as fallback for out_dtype
        if torch_out_dtype is None:
            torch_out_dtype = (
                B.dtype
                if B.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
                else torch.bfloat16
            )

        # Handle 3D tensors by flattening to 2D (matching reference semantics)
        original_B_shape = None
        if B.ndim == 3:
            original_B_shape = B.shape
            B = B.reshape(-1, B.shape[-1])
        if A.ndim == 3:
            A = A.reshape(-1, A.shape[-1])

        # Core GEMM: general_gemm(A, B, usage_a, usage_b, out_dtype, bias=None)
        # We pass bias=None here and handle bias/gelu ourselves to match reference semantics
        out = gemm_mod.general_gemm(B, A, usage_a, usage_b, torch_out_dtype, bias=None)

        # Restore 3D shape: a non-transposed B contributes its outer dimensions to the output
        if original_B_shape is not None and not transB:
            out = out.view(original_B_shape[0], original_B_shape[1], -1)

        if alpha != 1.0:
            out = out * alpha

        gelu_input_ret = None

        # Bias handling: in backward (grad=True), bias only requests fused BGRAD epilogue,
        # its value is NOT added to the GEMM result.
        if bias is not None and not grad:
            out = out + bias

        # GeLU handling
        if gelu:
            if grad:
                # Backward: compute dgelu(out, gelu_in)
                # out is the upstream gradient, gelu_in is the saved forward pre-activation
                if gelu_in is None:
                    raise ValueError("gelu_in must be provided for a backward GELU GEMM")
                x = gelu_in.detach().requires_grad_(True)
                with torch.enable_grad():
                    y = F.gelu(x, approximate="tanh")
                    y.backward(out)
                out = x.grad
            else:
                # Forward: save pre-gelu input and apply gelu
                if gelu_in is not None:
                    gelu_in.copy_(out)
                    gelu_input_ret = gelu_in
                else:
                    gelu_input_ret = out.clone()
                out = F.gelu(out, approximate="tanh")

        # Cast to output dtype if needed
        if torch_out_dtype is not None and out.dtype != torch_out_dtype:
            out = out.to(torch_out_dtype)

        # Accumulate into D if provided
        if D is not None:
            if accumulate:
                beta_val = beta if beta is not None else 1.0
                D.mul_(beta_val).add_(out)
                out = D
            else:
                D.copy_(out)
                out = D

        # Compute bias gradient in backward pass
        bias_grad = None
        if grad and bias is not None:
            # BGRADB epilogue: reduce over the batch/sequence dimension of B
            # At this point B is already 2D (flattened above), matching reference behavior
            bias_grad = B.sum(dim=0).to(dtype=out.dtype)

        extra_output_ret = None

        return out, bias_grad, gelu_input_ret, extra_output_ret

    def te_general_grouped_gemm(
        self,
        A: List[Any],
        transa: bool,
        B: List[Any],
        transb: bool,
        D: Optional[List[torch.Tensor]],
        D_type: DType,
        m_splits: List[int],
        bias: List[torch.Tensor],
        bias_type: DType,
        single_output: bool,
        pre_gelu_out: List[torch.Tensor],
        grad: bool,
        workspace: List[torch.Tensor],
        workspaceSizes: int,
        accumulate: bool,
        use_split_accumulator: bool,
        math_sm_count: int,
    ) -> Optional[List[torch.Tensor]]:
        """Grouped GEMM adapter for TransformerEngineNPU.

        TE-FL semantics for every group:

            D[i] = op(B[i], transb) @ op(A[i], transa)

        Native NPU mappings:

            Forward:
                layout="TN", group_type=0

            dgrad:
                layout="NN", group_type=0

            wgrad:
                layout="NT", group_type=2

        The group_type=2 path requires an Ascend A2/A3 device. Operations that
        require bgrad, GELU/dGELU, mixed per-group epilogues, unsupported dtypes,
        or non-standard transpose layouts fall back to per-group generic_gemm.
        """

        num_gemms = len(A)

        if len(B) != num_gemms:
            raise ValueError(
                "A and B must contain the same number of groups, "
                f"got len(A)={len(A)} and len(B)={len(B)}"
            )

        if num_gemms == 0:
            return bias

        if single_output and D is None:
            raise ValueError(
                "D must be provided when single_output=True"
            )

        def has_tensor(tensors, index: int) -> bool:
            return (
                tensors is not None
                and index < len(tensors)
                and tensors[index] is not None
                and tensors[index].numel() > 0
            )

        def matrix_shape(
            tensor: Any,
            transpose: bool,
        ) -> Tuple[int, int]:
            if tensor.ndim != 2:
                raise ValueError(
                    "te_general_grouped_gemm currently requires 2D tensors, "
                    f"got shape={tuple(tensor.shape)}"
                )

            if transpose:
                return int(tensor.shape[1]), int(tensor.shape[0])

            return int(tensor.shape[0]), int(tensor.shape[1])

        # TE-FL computes:
        #
        #     op(B, transb) @ op(A, transa)
        #
        # Calculate and validate each output shape first.
        output_shapes: List[Tuple[int, int]] = []

        for index in range(num_gemms):
            a_rows, a_cols = matrix_shape(A[index], transa)
            b_rows, b_cols = matrix_shape(B[index], transb)

            if b_cols != a_rows:
                raise ValueError(
                    f"Incompatible GEMM shapes for group {index}: "
                    f"B_comp=({b_rows}, {b_cols}), "
                    f"A_comp=({a_rows}, {a_cols})"
                )

            output_shapes.append((b_rows, a_cols))

        torch_out_dtype = _to_torch_dtype(D_type)

        if torch_out_dtype is None:
            if D is not None and len(D) > 0:
                torch_out_dtype = D[0].dtype
            else:
                torch_out_dtype = B[0].dtype

                if torch_out_dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ):
                    torch_out_dtype = torch.bfloat16

        # Allocate per-group output buffers when D is not provided.
        # single_output=True requires the caller-provided packed buffer.
        if D is None:
            D = [
                torch.empty(
                    shape,
                    dtype=torch_out_dtype,
                    device=B[index].device,
                )
                for index, shape in enumerate(output_shapes)
            ]

        if single_output:
            if len(D) != 1:
                raise ValueError(
                    "single_output=True requires exactly one D tensor, "
                    f"got {len(D)}"
                )

            output_widths = {
                shape[1]
                for shape in output_shapes
            }

            if len(output_widths) != 1:
                raise ValueError(
                    "single_output=True requires all grouped GEMMs "
                    "to have the same output width"
                )

            expected_shape = (
                sum(shape[0] for shape in output_shapes),
                output_shapes[0][1],
            )

            if tuple(D[0].shape) != expected_shape:
                raise ValueError(
                    "Invalid single output shape: "
                    f"expected {expected_shape}, "
                    f"got {tuple(D[0].shape)}"
                )
        else:
            if len(D) != num_gemms:
                raise ValueError(
                    f"Expected {num_gemms} output tensors, got {len(D)}"
                )

            for index, (destination, expected_shape) in enumerate(
                zip(D, output_shapes)
            ):
                if tuple(destination.shape) != expected_shape:
                    raise ValueError(
                        f"Invalid output shape for group {index}: "
                        f"expected {expected_shape}, "
                        f"got {tuple(destination.shape)}"
                    )

        bias_flags = [
            has_tensor(bias, index)
            for index in range(num_gemms)
        ]
        gelu_flags = [
            has_tensor(pre_gelu_out, index)
            for index in range(num_gemms)
        ]

        has_any_bias = any(bias_flags)
        has_all_bias = all(bias_flags)
        has_any_gelu = any(gelu_flags)

        all_dense_tensors = all(
            isinstance(tensor, torch.Tensor)
            for tensor in list(A) + list(B)
        )

        same_output_width = (
            len({shape[1] for shape in output_shapes}) == 1
        )

        # This dense wrapper does not pass quantization parameters to
        # npu_grouped_matmul, so only use the non-quantized floating-point path.
        native_dense_dtypes = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }

        native_dtype_supported = False
        native_device_supported = False

        if all_dense_tensors:
            input_dtypes = {
                tensor.dtype
                for tensor in list(A) + list(B)
            }

            native_dtype_supported = (
                len(input_dtypes) == 1
                and next(iter(input_dtypes)) in native_dense_dtypes
                and torch_out_dtype in native_dense_dtypes
            )

            input_devices = {
                tensor.device
                for tensor in list(A) + list(B)
            }
            native_device_supported = len(input_devices) == 1

        # group_type=0:
        #   The packed B input is not transposed. This covers forward and dgrad.
        #
        # group_type=2:
        #   The packed B input is transposed and split along its K dimension.
        #   This covers the standard TE wgrad layout="NT".
        regular_native_layout = not transb
        native_wgrad_layout = transb and not transa

        native_layout_supported = (
            regular_native_layout
            or native_wgrad_layout
        )

        # Forward bias can use the current TransformerEngineNPU wrapper.
        #
        # In backward mode the bias argument represents a bgrad destination,
        # not a forward bias tensor, so it must not be passed to the native
        # grouped GEMM as bias.
        native_bias_supported = (
            not has_any_bias
            or (
                regular_native_layout
                and not grad
                and has_all_bias
            )
        )

        # npu_grouped_matmul does not implement the TE dGELU/pre_gelu_out
        # contract. All GELU-related paths therefore use generic_gemm.
        native_epilogue_supported = (
            native_bias_supported
            and not has_any_gelu
        )

        use_native_path = (
            num_gemms > 1
            and all_dense_tensors
            and native_dtype_supported
            and native_device_supported
            and native_layout_supported
            and native_epilogue_supported
            and same_output_width
        )

        if use_native_path:
            if m_splits:
                split_sizes = [
                    int(size)
                    for size in m_splits
                ]

                if len(split_sizes) != num_gemms:
                    raise ValueError(
                        f"Expected {num_gemms} m_splits, "
                        f"got {len(split_sizes)}"
                    )
            else:
                split_sizes = [
                    int(tensor.shape[0])
                    for tensor in B
                ]

            # For both group_type=0 and the standard group_type=2 wgrad
            # mapping, m_splits describe the original row count of B[i].
            expected_splits = [
                int(tensor.shape[0])
                for tensor in B
            ]

            if split_sizes != expected_splits:
                raise ValueError(
                    "m_splits must match the row counts of B for the "
                    "native NPU grouped GEMM path: "
                    f"expected {expected_splits}, "
                    f"got {split_sizes}"
                )

            # group_list_type=1 means group_split contains per-group sizes,
            # rather than cumulative offsets.
            group_split = torch.tensor(
                split_sizes,
                dtype=torch.int64,
                device=B[0].device,
            )

            packed_input = torch.cat(
                B,
                dim=0,
            )

            layout = (
                ("T" if transa else "N")
                + ("T" if transb else "N")
            )

            group_type = (
                2 if native_wgrad_layout else 0
            )

            # Bias is only a forward group_type=0 epilogue here.
            native_use_bias = (
                regular_native_layout
                and not grad
                and has_all_bias
            )

            gemm_mod = _get_tenpu_gemm()

            packed_output = gemm_mod.general_grouped_gemm(
                A,                  # NPU B: RHS/weight tensor list
                packed_input,       # NPU A: packed LHS/input tensor
                group_split,
                layout=layout,
                use_bias=native_use_bias,
                biases=bias if native_use_bias else None,
                group_type=group_type,
                group_list_type=1,
                split_item=3,
                out_dtype=torch_out_dtype,
            )

            if packed_output is None:
                raise RuntimeError(
                    "TransformerEngineNPU general_grouped_gemm "
                    "returned None"
                )

            if not isinstance(packed_output, torch.Tensor):
                raise TypeError(
                    "TransformerEngineNPU general_grouped_gemm must "
                    "return a Tensor for split_item=3, "
                    f"got {type(packed_output)}"
                )

            expected_packed_shape = (
                sum(shape[0] for shape in output_shapes),
                output_shapes[0][1],
            )
            expected_packed_numel = (
                expected_packed_shape[0]
                * expected_packed_shape[1]
            )

            if packed_output.numel() != expected_packed_numel:
                raise RuntimeError(
                    "Unexpected native grouped GEMM output size: "
                    f"expected shape compatible with "
                    f"{expected_packed_shape} "
                    f"({expected_packed_numel} elements), "
                    f"got shape={tuple(packed_output.shape)} "
                    f"({packed_output.numel()} elements)"
                )

            # group_type=0 normally returns [sum(M_i), N].
            #
            # group_type=2 may return [num_groups, N, K] when all group
            # shapes are equal. Flattening the leading group dimensions
            # converts both cases to TE-FL's packed [sum(rows_i), width]
            # representation while preserving group order.
            packed_output = packed_output.reshape(
                expected_packed_shape
            )

            if single_output:
                source = packed_output.to(D[0].dtype)

                if accumulate:
                    D[0].add_(source)
                else:
                    D[0].copy_(source)
            else:
                output_chunks = torch.split(
                    packed_output,
                    [shape[0] for shape in output_shapes],
                    dim=0,
                )

                for destination, source in zip(D, output_chunks):
                    source = source.to(destination.dtype)

                    if accumulate:
                        destination.add_(source)
                    else:
                        destination.copy_(source)

            # TE-FL expects this function to return the bias/bgrad buffers;
            # GEMM results have already been written into D.
            return bias

        # Correctness fallback for:
        #
        #   - bgrad
        #   - GELU/dGELU
        #   - mixed per-group bias/GELU epilogues
        #   - transa=True and transb=True
        #   - unsupported dense dtypes
        #   - quantized/GroupedTensor inputs
        #   - heterogeneous output widths
        single_output_offset = 0

        for index in range(num_gemms):
            if single_output:
                output_rows = output_shapes[index][0]

                destination = D[0][
                    single_output_offset:
                    single_output_offset + output_rows
                ]

                single_output_offset += output_rows
            else:
                destination = D[index]

            bias_tensor = (
                bias[index]
                if bias_flags[index]
                else None
            )
            gelu_input = (
                pre_gelu_out[index]
                if gelu_flags[index]
                else None
            )

            if workspace:
                gemm_workspace = workspace[
                    min(index, len(workspace) - 1)
                ]
            else:
                gemm_workspace = torch.empty(
                    0,
                    dtype=torch.uint8,
                    device=B[index].device,
                )

            _, bias_grad, _, _ = self.generic_gemm(
                A=A[index],
                transA=transa,
                B=B[index],
                transB=transb,
                D=destination,
                quantizer=None,
                output_dtype=D_type,
                bias=bias_tensor,
                bias_type=bias_type,
                gelu=gelu_flags[index],
                gelu_in=gelu_input,
                grad=grad,
                workspace=gemm_workspace,
                workspace_size=workspaceSizes,
                accumulate=accumulate,
                use_split_accumulator=use_split_accumulator,
            )

            # generic_gemm returns the bias gradient, but does not necessarily
            # write it into the grouped GEMM bias/bgrad destination.
            if (
                grad
                and bias_flags[index]
                and bias_grad is not None
            ):
                bias_grad = bias_grad.to(
                    bias[index].dtype
                )

                if accumulate:
                    bias[index].add_(bias_grad)
                else:
                    bias[index].copy_(bias_grad)

        # math_sm_count is a CUDA-specific tuning parameter and has no
        # corresponding control in torch_npu.npu_grouped_matmul.
        _ = math_sm_count
        return bias
