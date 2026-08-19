# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""NPU vendor backend for TE-FL plugin system.

Bridges Ascend NPU operations into the TE-FL unified plugin interface
by delegating to transformer_engine_npu (pip-installed from TransformerEngineNPU).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from ....ops import TEFLBackendBase, NVTE_Fused_Attn_Backend, DType
from .flash_attention import NPUFlashAttention
from .permutation import NPUPermutation


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
        import transformer_engine_npu  # noqa: F401

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def _get_torch_npu():
    """Ensure torch_npu is imported (activates NPU device support in PyTorch)."""
    import torch_npu  # noqa: F401

    return torch_npu


def _get_tenpu_gemm():
    """Get GEMM ops subpackage."""
    import transformer_engine_npu

    return transformer_engine_npu.pytorch.ops.gemm


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

        available_backends = [
            use_flash_attention,
            use_fused_attention,
            use_unfused_attention,
        ]

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

    def get_permutation_class(self):
        """Return the NPU adapter for TE's public permutation API."""
        return NPUPermutation

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

        if zero_centered_gamma:
            weight = weight + 1

        # NPU npu_rms_norm requires 2D input: [outer_dim, hidden_size]
        input_shape = input.shape
        inner_dim = weight.shape[0]
        x_2d = input.reshape(-1, inner_dim)

        out_2d, inv_rms = _get_torch_npu().npu_rms_norm(x_2d, weight, epsilon=eps)

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

        if zero_centered_gamma:
            gamma = gamma + 1

        # NPU kernel requires 2D input
        input_shape = x.shape
        inner_dim = gamma.shape[0]
        x_2d = x.reshape(-1, inner_dim)
        dz_2d = dz.reshape(-1, inner_dim)

        # NPU kernel requires rstd in float32
        rsigma_fp32 = rsigma.float() if rsigma.dtype != torch.float32 else rsigma

        dx_2d, dw = _get_torch_npu().npu_rms_norm_backward(
            dz_2d, x_2d, gamma, rsigma_fp32
        )

        # Reshape dx back to original input shape
        dx = dx_2d.reshape(input_shape)

        return dx, dw

    # ===================== Grouped GEMM =====================

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
            Forward:  layout="TN", group_type=0
            dgrad:    layout="NN", group_type=0
            wgrad:    layout="NT", group_type=2

        The group_type=2 path requires an Ascend A2/A3 device. Unsupported
        combinations raise ``NotImplementedError`` so TE-FL can select its
        reference implementation for the complete grouped GEMM operation.
        """

        num_gemms = len(A)
        if len(B) != num_gemms:
            raise ValueError(
                f"A/B group count mismatch: len(A)={len(A)}, len(B)={len(B)}"
            )
        if num_gemms == 0:
            return bias

        def op_shape(tensor: Any, transpose: bool) -> Tuple[int, int]:
            if tensor.ndim != 2:
                raise ValueError(
                    f"Grouped GEMM requires 2D tensors, got {tuple(tensor.shape)}"
                )
            rows, cols = map(int, tensor.shape)
            return (cols, rows) if transpose else (rows, cols)

        def has_tensor(tensors, index: int) -> bool:
            return (
                tensors is not None
                and index < len(tensors)
                and tensors[index] is not None
                and tensors[index].numel() > 0
            )

        # 1. Validate GEMMs and prepare destinations.
        output_shapes: List[Tuple[int, int]] = []
        for index, (a_tensor, b_tensor) in enumerate(zip(A, B)):
            a_rows, a_cols = op_shape(a_tensor, transa)
            b_rows, b_cols = op_shape(b_tensor, transb)
            if b_cols != a_rows:
                raise ValueError(
                    f"Incompatible shapes for group {index}: "
                    f"op(B)=({b_rows}, {b_cols}), op(A)=({a_rows}, {a_cols})"
                )
            output_shapes.append((b_rows, a_cols))

        out_dtype = _to_torch_dtype(D_type)
        if out_dtype is None:
            out_dtype = D[0].dtype if D else B[0].dtype
            if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                out_dtype = torch.bfloat16

        if single_output:
            if D is None or len(D) != 1:
                raise ValueError("single_output=True requires exactly one D tensor")
            if len({shape[1] for shape in output_shapes}) != 1:
                raise ValueError("single_output=True requires a common output width")
            expected_shape = (
                sum(shape[0] for shape in output_shapes),
                output_shapes[0][1],
            )
            if tuple(D[0].shape) != expected_shape:
                raise ValueError(
                    f"Invalid D shape: expected {expected_shape}, got {tuple(D[0].shape)}"
                )
        else:
            if D is None:
                D = [
                    torch.empty(
                        shape,
                        dtype=out_dtype,
                        device=B[index].device,
                    )
                    for index, shape in enumerate(output_shapes)
                ]
            if len(D) != num_gemms:
                raise ValueError(f"Expected {num_gemms} output tensors, got {len(D)}")
            for index, (destination, expected_shape) in enumerate(
                zip(D, output_shapes)
            ):
                if tuple(destination.shape) != expected_shape:
                    raise ValueError(
                        f"Invalid D[{index}] shape: expected {expected_shape}, "
                        f"got {tuple(destination.shape)}"
                    )

        bias_flags = [has_tensor(bias, i) for i in range(num_gemms)]
        gelu_flags = [has_tensor(pre_gelu_out, i) for i in range(num_gemms)]

        # 2. Decide whether the official native wrapper can represent this call.
        if not transb:
            native_mode = "m_split"
        elif not transa:
            native_mode = "k_split"
        else:
            native_mode = None

        dense_tensors = all(isinstance(tensor, torch.Tensor) for tensor in (*A, *B))
        dtype_ok = False
        device_ok = False
        shape_ok = False

        if dense_tensors:
            input_dtypes = {tensor.dtype for tensor in (*A, *B)}
            input_dtype = next(iter(input_dtypes)) if len(input_dtypes) == 1 else None
            dtype_ok = (
                input_dtype
                in {
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                }
                and out_dtype == input_dtype
            )
            device_ok = len({tensor.device for tensor in (*A, *B)}) == 1

            if native_mode == "m_split":
                shape_ok = (
                    len({int(tensor.shape[1]) for tensor in B}) == 1
                    and len({shape[1] for shape in output_shapes}) == 1
                )
            elif native_mode == "k_split":
                # K-split packs both operands, so every group must produce the
                # same [M, N] shape.
                shape_ok = len(set(output_shapes)) == 1

        has_bias = any(bias_flags)
        epilogue_ok = not any(gelu_flags) and (
            not has_bias or (native_mode == "m_split" and not grad and all(bias_flags))
        )

        use_native = (
            1 < num_gemms <= 128
            and native_mode is not None
            and dense_tensors
            and dtype_ok
            and device_ok
            and shape_ok
            and epilogue_ok
        )

        # 3. Native M-split/K-split path.
        if use_native:
            expected_splits = [int(tensor.shape[0]) for tensor in B]
            split_sizes = (
                [int(size) for size in m_splits]
                if m_splits is not None and len(m_splits) > 0
                else expected_splits
            )
            if split_sizes != expected_splits:
                raise ValueError(
                    "m_splits must equal the original B row counts: "
                    f"expected {expected_splits}, got {split_sizes}"
                )

            # No kernel work is needed for an entirely empty token batch.
            if sum(split_sizes) == 0:
                if native_mode == "k_split" and not accumulate:
                    for destination in D:
                        destination.zero_()
                return bias

            group_split = torch.tensor(
                split_sizes,
                dtype=torch.int64,
                device=B[0].device,
            )
            packed_b = torch.cat(B, dim=0)

            if native_mode == "m_split":
                # Final NPU operands: x=[cat(B)], weight=A.
                npu_weight = A
                group_type = 0
            else:
                # layout="NT" turns cat(B) into the left operand:
                #
                #   x      = [cat(B).T]  -> [M, sum(K_i)]
                #   weight = [cat(A)]    -> [sum(K_i), N]
                #
                # Both lists therefore have length 1, as required by K-split.
                npu_weight = torch.cat(A, dim=0)
                group_type = 2

            layout = ("T" if transa else "N") + ("T" if transb else "N")
            use_forward_bias = native_mode == "m_split" and not grad and all(bias_flags)

            packed_output = _get_tenpu_gemm().general_grouped_gemm(
                npu_weight,
                packed_b,
                group_split,
                layout=layout,
                use_bias=use_forward_bias,
                biases=bias if use_forward_bias else None,
                group_type=group_type,
                group_list_type=1,
                split_item=3,
                out_dtype=out_dtype,
            )

            if not isinstance(packed_output, torch.Tensor):
                raise TypeError(
                    "general_grouped_gemm must return one Tensor "
                    f"for split_item=3, got {type(packed_output)}"
                )

            packed_shape = (
                sum(shape[0] for shape in output_shapes),
                output_shapes[0][1],
            )
            packed_numel = packed_shape[0] * packed_shape[1]
            if packed_output.numel() != packed_numel:
                raise RuntimeError(
                    "Unexpected grouped GEMM output: "
                    f"expected {packed_numel} elements, "
                    f"got shape={tuple(packed_output.shape)}"
                )

            # M-split is already 2D. K-split [G, M, N] is flattened to TE's
            # packed [G*M, N] representation.
            packed_output = packed_output.reshape(packed_shape)

            if single_output:
                outputs = [packed_output]
            else:
                outputs = torch.split(
                    packed_output,
                    [shape[0] for shape in output_shapes],
                    dim=0,
                )

            for destination, source in zip(D, outputs):
                source = source.to(destination.dtype)
                if accumulate:
                    destination.add_(source)
                else:
                    destination.copy_(source)

            return bias

        raise NotImplementedError(
            "NPU grouped GEMM only supports native M-split/K-split calls "
            "without unsupported epilogues"
        )
