# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
from packaging.version import Version as PkgVersion

import torch
from transformer_engine import te_device_type
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
)

from transformer_engine.pytorch.quantized_tensor import (
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.constants import (
    QKVLayouts,
    dist_group_type,
)

from transformer_engine.pytorch.distributed import get_distributed_world_size
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.attention.inference import InferenceParams

import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils

from transformer_engine.plugin.core.ops import FlashAttentionBase

import flag_gems


class AttnFuncFL(torch.autograd.Function):
    """FlagGems SDPA adapter for dense and packed, unpadded training inputs.

    Packed sequences are dispatched separately so attention never crosses a
    sequence boundary. FlagGems SDPA itself consumes contiguous BHSD tensors.
    This adapter supports zero-dropout, non-paged attention without sliding
    windows. THD inputs must have no padding between sequences.
    """

    @staticmethod
    def _to_bhsd(x, fmt):
        """Convert one dense batch or one packed sequence to the SDPA layout."""
        if fmt == "sbhd":
            return x.permute(1, 2, 0, 3).contiguous()
        if fmt == "bshd":
            return x.permute(0, 2, 1, 3).contiguous()
        return x.transpose(0, 1).unsqueeze(0).contiguous()

    @staticmethod
    def _from_bhsd(x, fmt):
        """Restore the caller's layout for outputs and Q/K/V gradients."""
        if fmt == "sbhd":
            return x.permute(2, 0, 1, 3).contiguous()
        if fmt == "bshd":
            return x.permute(0, 2, 1, 3).contiguous()
        return x.squeeze(0).transpose(0, 1).contiguous()

    @staticmethod
    def forward(
        ctx,
        is_training,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        page_table_k,
        page_table_v,
        q,
        k,
        v,
        attn_scale,
        dropout_p,
        qkv_layout,
        attn_mask_type,
        window_size,
        rng_gen,
        deterministic,
        layer_number,
    ):
        fmt, q_fmt, kv_fmt = dpa_utils.get_qkv_format(qkv_layout, None)
        layouts = {
            "".join(char for char in part if char.isalpha())
            for part in qkv_layout.replace("paged_kv_", "").split("_")
        }
        if fmt not in ("sbhd", "bshd", "thd") or q_fmt != kv_fmt or len(layouts) != 1:
            raise NotImplementedError(f"FlagGems SDPA layout: {qkv_layout}")
        expected_ndim = 3 if fmt == "thd" else 4
        if any(x.ndim != expected_ndim for x in (q, k, v)):
            raise ValueError(f"FlagGems SDPA {fmt} inputs must have {expected_ndim} dimensions")
        if k.shape != v.shape:
            raise ValueError("FlagGems SDPA K and V must have matching shapes")
        if dropout_p or page_table_k is not None or page_table_v is not None:
            raise NotImplementedError("FlagGems SDPA requires zero dropout and non-paged inputs")
        if attn_mask_type not in ("no_mask", "causal", "padding", "padding_causal"):
            raise NotImplementedError(f"FlagGems SDPA mask: {attn_mask_type}")
        causal = attn_mask_type in ("causal", "padding_causal")
        if window_size not in (None, (-1, -1), (-1, 0) if causal else (-1, -1)):
            raise NotImplementedError("FlagGems SDPA does not implement sliding windows")
        if fmt != "thd" and "padding" in attn_mask_type:
            raise NotImplementedError("FlagGems SDPA padding requires packed THD inputs")
        if fmt == "thd":
            if any(
                bounds is None or bounds.ndim != 1 or bounds.dtype not in (torch.int32, torch.int64)
                for bounds in (cu_seqlens_q, cu_seqlens_kv)
            ):
                raise ValueError("Packed cu_seqlens must be one-dimensional integer tensors")
            # FlagGems' dense SDPA has no cu_seqlens argument. Split at actual
            # sequence boundaries instead of treating packed tokens as one
            # sequence; the latter would allow attention across samples.
            # Reading these bounds synchronizes device metadata to the host.
            q_bounds = cu_seqlens_q.tolist()
            k_bounds = cu_seqlens_kv.tolist()
            if (
                len(q_bounds) != len(k_bounds)
                or len(q_bounds) < 2
                or q_bounds[0] != 0
                or k_bounds[0] != 0
                or q_bounds[-1] != q.shape[0]
                or k_bounds[-1] != k.shape[0]
            ):
                raise ValueError("Packed cu_seqlens must span unpadded Q/KV tensors")
            parts = []
            for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]):
                if qb <= qa or kb <= ka:
                    raise NotImplementedError("Empty packed attention sequences are unsupported")
                parts.append((q[qa:qb], k[ka:kb], v[ka:kb]))
        else:
            parts = [(q, k, v)]
        saved, outputs = [], []
        for qp, kp, vp in parts:
            qp, kp, vp = (AttnFuncFL._to_bhsd(x, fmt) for x in (qp, kp, vp))
            if causal and qp.shape[2] != kp.shape[2]:
                raise NotImplementedError("Non-square causal alignment is unsupported")
            with torch.cuda.nvtx.range("transformer_engine.AttnFuncFL.forward"):
                op, m = flag_gems.scaled_dot_product_attention_forward(
                    qp,
                    kp,
                    vp,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=causal,
                    scale=attn_scale,
                    enable_gqa=True,
                )
            saved.extend((qp, kp, vp, op, m))
            outputs.append(AttnFuncFL._from_bhsd(op, fmt))
        out = torch.cat(outputs, dim=0) if fmt == "thd" else outputs[0]
        from transformer_engine.pytorch.cpu_offload import (
            is_cpu_offload_enabled,
            mark_activation_offload,
        )

        if is_cpu_offload_enabled():
            mark_activation_offload(q, k, v, out, *saved)
        tensors, ctx.tensor_objects = prepare_for_saving(*saved)
        ctx.save_for_backward(*tensors)
        ctx.fmt, ctx.is_causal, ctx.attn_scale = fmt, causal, attn_scale
        return out

    @staticmethod
    def backward(ctx, d_out, *_args):
        saved = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        grads = [[], [], []]
        offset = 0
        for i in range(0, len(saved), 5):
            q, k, v, out, m = saved[i : i + 5]
            if ctx.fmt == "thd":
                dout = d_out[offset : offset + q.shape[2]]
                offset += q.shape[2]
            else:
                dout = d_out
            with torch.cuda.nvtx.range("transformer_engine.AttnFuncFL.backward"):
                dq, dk, dv = flag_gems.scaled_dot_product_attention_backward(
                    AttnFuncFL._to_bhsd(dout, ctx.fmt),
                    q,
                    k,
                    v,
                    out.contiguous(),
                    m.contiguous(),
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=ctx.is_causal,
                    scale=ctx.attn_scale,
                    enable_gqa=True,
                )
            for pieces, grad in zip(grads, (dq, dk, dv)):
                pieces.append(AttnFuncFL._from_bhsd(grad, ctx.fmt))
        dq, dk, dv = (torch.cat(g, dim=0) if ctx.fmt == "thd" else g[0] for g in grads)
        return (None,) * 7 + (dq, dk, dv) + (None,) * 8


class FlashAttentionFL(FlashAttentionBase):
    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = None,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__(
            softmax_scale=softmax_scale,
            attention_dropout=attention_dropout,
            attention_dropout_ctx=attention_dropout_ctx,
            attention_type=attention_type,
            layer_number=layer_number,
            deterministic=deterministic,
        )
        self.use_FAv2_bwd = os.getenv(
            "NVTE_FUSED_ATTN_USE_FAv2_BWD", "0"
        ) == "1" and get_device_compute_capability() == (9, 0)

        def remove_extra_states_check(self, incompatible_keys):
            for key in incompatible_keys.missing_keys:
                if "fused_attention._extra_state" in key:
                    incompatible_keys.missing_keys.remove(key)
            for key in incompatible_keys.unexpected_keys:
                if "fused_attention._extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)
                    warnings.warn(
                        "fused_attention._extra_state is not loaded from checkpoint. Please map "
                        "FusedAttention's _extra_state to DotProductAttention's _extra_state."
                    )

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    @property
    def backend_name(self) -> str:
        return "flagos"

    @no_torch_dynamo()
    def _forward_impl(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_layout: str = "sbh3d",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
        cp_global_ranks: List[int] = None,
        cp_stream: torch.cuda.Stream = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers=None,
        pad_between_seqs: Optional[bool] = False,
        inference_params: Optional[InferenceParams] = None,
        flash_attention_backend: Optional[PkgVersion] = PkgVersion("0"),
        fp8_output: bool = False,
        num_splits: Optional[int] = 1,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pad_between_seqs:
            raise NotImplementedError("FlagGems SDPA does not support padding between sequences")
        if alibi_slopes is not None:
            raise NotImplementedError("FlagGems SDPA does not support ALiBi")
        if (
            fp8
            or fp8_output
            or any(isinstance(x, Float8Tensor) for x in (query_layer, key_layer, value_layer))
        ):
            raise NotImplementedError("FlagGems SDPA supports FP16/BF16 inputs and outputs only")
        assert all(
            x.dtype in [torch.float16, torch.bfloat16] or isinstance(x, Float8Tensor)
            for x in [query_layer, key_layer, value_layer]
        ), "FLAttention only supports FP16 and BF16 data types, or Float8Tensors."
        assert (
            query_layer.device.type == te_device_type()
            and key_layer.device.type == te_device_type()
            and value_layer.device.type == te_device_type()
        ), f"FLAttention only supports {te_device_type()} tensors."
        assert qkv_layout in QKVLayouts, f"FLAttention does not support qkv_layout = {qkv_layout}!"

        cp_size = 1
        if isinstance(cp_group, dist_group_type):
            cp_size = get_distributed_world_size(cp_group)
        elif isinstance(cp_group, list):
            for group in cp_group:
                cp_size *= get_distributed_world_size(group)
        context_parallel = cp_size > 1
        assert not context_parallel, "FLAttention do not support context parallel now"

        qkv_format, q_format, kv_format = dpa_utils.get_qkv_format(qkv_layout, inference_params)

        if q_format in ["bshd", "sbhd"] or kv_format in ["bshd", "sbhd"]:
            batch_size = query_layer.shape[0] if q_format == "bshd" else query_layer.shape[1]
            if cu_seqlens_q is not None:
                cu_seqlens_q = cu_seqlens_q[: batch_size + 1]
            if cu_seqlens_kv is not None:
                cu_seqlens_kv = cu_seqlens_kv[: batch_size + 1]

        page_table = None
        if inference_params is None:
            if qkv_format in ["sbhd", "bshd"]:
                if qkv_format == "sbhd":
                    batch_size = query_layer.shape[1]
                    max_seqlen_q = query_layer.shape[0]
                    max_seqlen_kv = key_layer.shape[0]
                if qkv_format == "bshd":
                    batch_size = query_layer.shape[0]
                    max_seqlen_q = query_layer.shape[1]
                    max_seqlen_kv = key_layer.shape[1]
                max_seqlen_q *= cp_size
                max_seqlen_kv *= cp_size
                if "padding" in attn_mask_type:
                    assert (
                        not context_parallel
                    ), "Padding mask not supported with context parallelism!"
                    if cu_seqlens_q is None or cu_seqlens_kv is None:
                        if attention_mask is None:
                            raise RuntimeError(
                                "Please provide attention_mask or cu_seqlens for padding!"
                            )
                        if self.attention_type == "self":
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask)
                            cu_seqlens_kv = cu_seqlens_q
                        else:
                            cu_seqlens_q = dpa_utils.get_cu_seqlens(attention_mask[0])
                            cu_seqlens_kv = dpa_utils.get_cu_seqlens(attention_mask[1])
                else:
                    if cu_seqlens_q is None:
                        cu_seqlens_q = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_q,
                            query_layer.device,
                        )
                    if cu_seqlens_kv is None:
                        cu_seqlens_kv = dpa_utils.get_full_cu_seqlens(
                            batch_size,
                            max_seqlen_kv,
                            key_layer.device,
                        )
            if qkv_format == "thd":
                assert (
                    max_seqlen_q is not None
                    and max_seqlen_kv is not None
                    and cu_seqlens_q is not None
                    and cu_seqlens_kv is not None
                ), "max_seqlen_q/kv and cu_seqlens_q/kv can not be None when qkv_format is thd!"
        elif inference_params.is_paged:
            page_table = inference_params.cache_manager.page_table

        with self.attention_dropout_ctx():
            _attn_impl = AttnFuncFL
            output = _attn_impl.apply(
                self.training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                page_table,
                page_table,
                query_layer,
                key_layer,
                value_layer,
                self.softmax_scale,
                self.attention_dropout if self.training else 0.0,
                qkv_layout,
                attn_mask_type,
                window_size,
                None,
                self.deterministic,
                self.layer_number,
            )

        return output.view(*output.shape[:-2], -1)
