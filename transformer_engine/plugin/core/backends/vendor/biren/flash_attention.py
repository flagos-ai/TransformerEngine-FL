# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""FlashAttention implementation for Biren BR200 using flashattn_train/flashattn_infer."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from transformer_engine.plugin.core.ops import FlashAttentionBase


class FlashAttentionBiren(FlashAttentionBase):
    """Flash Attention implementation backed by Biren's flashattn_train library.

    Directly calls flashattn_train.flash_attn_interface APIs without depending
    on the NVIDIA flash-attn package or TE native FlashAttention code paths.
    """

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

    @property
    def backend_name(self) -> str:
        return "biren"

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
        cp_group: Optional[Any] = None,
        cp_global_ranks: Optional[List[int]] = None,
        cp_stream: Optional[torch.cuda.Stream] = None,
        cp_comm_type: str = "p2p",
        fp8: bool = False,
        fp8_meta: Optional[Dict[str, Any]] = None,
        quantizers: Optional[Any] = None,
        pad_between_seqs: Optional[bool] = False,
        inference_params: Optional[Any] = None,
        flash_attention_backend: Optional[Any] = None,
        fp8_output: bool = False,
        num_splits: Optional[int] = 1,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pad_between_seqs:
            raise NotImplementedError(
                "pad_between_seqs is not supported in FlashAttentionBiren yet"
            )

        from flashattn_train.flash_attn_interface import (
            flash_attn_func,
            flash_attn_varlen_func,
        )
        from transformer_engine.pytorch.attention.dot_product_attention.utils import (
            get_qkv_format,
            get_cu_seqlens_and_indices,
            get_full_cu_seqlens,
            get_indices,
            PackTensors,
            UnpackTensor,
        )

        # FP8 inputs: dequantize to BF16 before running the BF16 flash attention kernel.
        # Normally FP8 attention routes to FusedAttention (suDNN), but handle gracefully
        # if it arrives here (e.g. fallback path). Accept both Float8Tensor and
        # Float8TensorStorage (the container DPA actually passes on its FP8 branch).
        from transformer_engine.pytorch.tensor import Float8Tensor, Float8TensorStorage

        _fp8_like = (Float8Tensor, Float8TensorStorage)
        if fp8 or isinstance(query_layer, _fp8_like):
            if isinstance(query_layer, _fp8_like):
                query_layer = query_layer.dequantize(dtype=torch.bfloat16)
            if isinstance(key_layer, _fp8_like):
                key_layer = key_layer.dequantize(dtype=torch.bfloat16)
            if isinstance(value_layer, _fp8_like):
                value_layer = value_layer.dequantize(dtype=torch.bfloat16)

        assert all(
            x.dtype in [torch.float16, torch.bfloat16]
            for x in [query_layer, key_layer, value_layer]
        ), "FlashAttentionBiren only supports FP16 and BF16 data types."

        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes not supported in FlashAttentionBiren yet")

        if cp_group is not None:
            raise NotImplementedError(
                "Context parallelism not supported in FlashAttentionBiren yet"
            )

        # Biren kernel bwd is a no-op for head_dim=32 — fallback to torch SDPA
        head_dim = query_layer.shape[-1]
        if head_dim == 32 and self.training:
            from transformer_engine.plugin.core.backends.reference.flash_attention import (
                FlashAttentionTorch,
            )

            fallback = FlashAttentionTorch(
                softmax_scale=self.softmax_scale,
                attention_dropout=self.attention_dropout,
                attention_dropout_ctx=self.attention_dropout_ctx,
                attention_type=self.attention_type,
                layer_number=self.layer_number,
                deterministic=self.deterministic,
            )
            fallback.training = self.training
            return fallback(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask,
                qkv_layout=qkv_layout,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                attn_mask_type=attn_mask_type,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                cp_group=cp_group,
                cp_global_ranks=cp_global_ranks,
                cp_stream=cp_stream,
                cp_comm_type=cp_comm_type,
                fp8=fp8,
                fp8_meta=fp8_meta,
                quantizers=quantizers,
                inference_params=inference_params,
            )

        qkv_format, q_format, kv_format = get_qkv_format(qkv_layout, inference_params)

        # Layout conversion: sbhd → bshd
        if qkv_format == "sbhd":
            query_layer, key_layer, value_layer = [
                x.transpose(0, 1).contiguous() for x in (query_layer, key_layer, value_layer)
            ]
        elif q_format == "sbhd" and kv_format == "bshd":
            query_layer = query_layer.transpose(0, 1).contiguous()

        # Determine batch_size and sequence lengths
        indices_q = None
        use_varlen = False

        if inference_params is not None:
            import flashattn_br200_infer

            is_paged = "paged_kv_" in qkv_layout

            batch_size = query_layer.shape[0]
            context_len = query_layer.shape[1]

            cu_seqlens_q = cu_seqlens_q[: batch_size + 1]
            cu_seqlens_kv = cu_seqlens_kv[: batch_size + 1]

            # Single host transfer for all cu_seqlens offsets: every loop
            # iteration below reads plain Python ints instead of issuing a
            # device->host sync per .item() call.
            cu_q_host = cu_seqlens_q.detach().cpu().tolist()
            cu_kv_host = cu_seqlens_kv.detach().cpu().tolist()

            total_q = cu_q_host[batch_size]
            seqlens_q_host = [cu_q_host[i + 1] - cu_q_host[i] for i in range(batch_size)]
            max_seqlen_q_val = max(seqlens_q_host)

            # Pack Q from [B, S, H, D] to [T_q, H, D]
            if total_q == batch_size * context_len:
                query_layer = query_layer.reshape(
                    total_q, query_layer.shape[2], query_layer.shape[3]
                )
            else:
                nH, nD = query_layer.shape[2], query_layer.shape[3]
                q_packed = torch.empty(
                    total_q,
                    nH,
                    nD,
                    dtype=query_layer.dtype,
                    device=query_layer.device,
                )
                for i in range(batch_size):
                    s = cu_q_host[i]
                    e = cu_q_host[i + 1]
                    q_packed[s:e] = query_layer[i, : e - s]
                query_layer = q_packed

            causal = "causal" in attn_mask_type
            ws = window_size if window_size is not None else (-1, -1)

            if is_paged:
                # Paged KV: convert to non-paged, then use varlen_fwd
                key_layer, value_layer = inference_params.convert_paged_to_nonpaged(
                    self.layer_number
                )

            # Non-paged KV [B, max_seq, H, D]: pack into [T_k, H, D] and
            # use varlen_fwd which gives exact results.
            total_kv = cu_kv_host[batch_size]
            max_seqlen_kv_val = max(cu_kv_host[i + 1] - cu_kv_host[i] for i in range(batch_size))

            max_seq = key_layer.shape[1]
            if total_kv == batch_size * max_seq:
                key_packed = key_layer.reshape(total_kv, *key_layer.shape[2:])
                val_packed = value_layer.reshape(total_kv, *value_layer.shape[2:])
            else:
                key_packed = torch.empty(
                    total_kv,
                    *key_layer.shape[2:],
                    dtype=key_layer.dtype,
                    device=key_layer.device,
                )
                val_packed = torch.empty(
                    total_kv,
                    *value_layer.shape[2:],
                    dtype=value_layer.dtype,
                    device=value_layer.device,
                )
                for i in range(batch_size):
                    s = cu_kv_host[i]
                    e = cu_kv_host[i + 1]
                    key_packed[s:e] = key_layer[i, : e - s]
                    val_packed[s:e] = value_layer[i, : e - s]

            result = flashattn_br200_infer.varlen_fwd(
                query_layer.contiguous(),
                key_packed.contiguous(),
                val_packed.contiguous(),
                None,
                cu_seqlens_q.contiguous(),
                cu_seqlens_kv.contiguous(),
                None,
                None,
                max_seqlen_q_val,
                max_seqlen_kv_val,
                self.softmax_scale,
                causal,
                ws[0],
                ws[1],
            )
            output = result[0]

            if isinstance(output, (list, tuple)):
                output = output[0]

            # Unpack output [T_q, H, D] → [B, S, H*D]
            if total_q == batch_size * context_len:
                output = output.reshape(batch_size, context_len, -1)
            else:
                hidden = output.shape[1] * output.shape[2]
                out_bshd = torch.zeros(
                    batch_size,
                    context_len,
                    hidden,
                    dtype=output.dtype,
                    device=output.device,
                )
                for i in range(batch_size):
                    s = cu_q_host[i]
                    e = cu_q_host[i + 1]
                    out_bshd[i, : e - s] = output[s:e].reshape(e - s, hidden)
                output = out_bshd

            return output.contiguous()

        if qkv_format in ["sbhd", "bshd"]:
            batch_size = query_layer.shape[0]
            max_seqlen_q = query_layer.shape[1]
            max_seqlen_kv = key_layer.shape[1]

            if "padding" in attn_mask_type:
                use_varlen = True
                # Pack padded tensors to variable-length format
                query_layer, key_layer, value_layer = [
                    x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
                    for x in [query_layer, key_layer, value_layer]
                ]

                if self.attention_type == "self":
                    if cu_seqlens_q is None:
                        assert attention_mask is not None
                        cu_seqlens_q, indices_q = get_cu_seqlens_and_indices(attention_mask)
                    else:
                        indices_q = get_indices(max_seqlen_q, cu_seqlens_q)
                    cu_seqlens_kv = cu_seqlens_q
                    query_layer, key_layer, value_layer = PackTensors.apply(
                        indices_q, query_layer, key_layer, value_layer
                    )
                else:
                    if cu_seqlens_q is None or cu_seqlens_kv is None:
                        assert attention_mask is not None
                        cu_seqlens_q, indices_q = get_cu_seqlens_and_indices(attention_mask[0])
                        cu_seqlens_kv, _ = get_cu_seqlens_and_indices(attention_mask[1])
                    else:
                        indices_q = get_indices(max_seqlen_q, cu_seqlens_q)
                    query_layer = PackTensors.apply(indices_q, query_layer)
                    indices_kv = get_indices(max_seqlen_kv, cu_seqlens_kv)
                    key_layer, value_layer = PackTensors.apply(indices_kv, key_layer, value_layer)
            else:
                if cu_seqlens_q is None:
                    cu_seqlens_q = get_full_cu_seqlens(batch_size, max_seqlen_q, query_layer.device)
                if cu_seqlens_kv is None:
                    cu_seqlens_kv = get_full_cu_seqlens(batch_size, max_seqlen_kv, key_layer.device)

        elif qkv_format == "thd":
            use_varlen = True
            assert cu_seqlens_q is not None and cu_seqlens_kv is not None
            if max_seqlen_q is None:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                max_seqlen_q = seqlens_q.max().item()
            if max_seqlen_kv is None:
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                max_seqlen_kv = seqlens_kv.max().item()

        # Call flashattn_train
        causal = "causal" in attn_mask_type
        dropout_p = self.attention_dropout if self.training else 0.0
        ws = window_size if window_size is not None else (-1, -1)

        # Check head_dim support — Biren FA supports head_dim in {32, 64, 128}
        if query_layer.shape[-1] not in (32, 64, 128):
            raise NotImplementedError(
                f"FlashAttentionBiren does not support head_dim={query_layer.shape[-1]}. "
                "Supported: 32, 64, 128."
            )

        with self.attention_dropout_ctx():
            if not use_varlen and qkv_format in ["bshd", "sbhd"]:
                # Non-padding path: flash_attn_func expects (B, S, H, D)
                output = flash_attn_func(
                    query_layer,
                    key_layer,
                    value_layer,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                    window_size=ws,
                    deterministic=self.deterministic,
                    dropout_p=dropout_p,
                )
            else:
                # Variable-length path: flash_attn_varlen_func expects (total, H, D)
                # Note: biren's flash_attn_varlen_func does not support dropout_p
                output = flash_attn_varlen_func(
                    query_layer,
                    key_layer,
                    value_layer,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                    window_size=ws,
                    deterministic=self.deterministic,
                )

        # Handle tuple return (some versions return (output, softmax_lse, ...))
        if isinstance(output, (list, tuple)):
            output = output[0]

        # Unpack if padding was used
        if indices_q is not None and qkv_format in ["sbhd", "bshd"]:
            output = UnpackTensor.apply(indices_q, batch_size * max_seqlen_q, output)

        # Reshape output to expected format
        if q_format == "sbhd":
            # (B, S, H, D) → (B, S, H*D) → (S, B, H*D)
            output = output.reshape(batch_size, max_seqlen_q, -1).transpose(0, 1)
        elif q_format == "bshd":
            # (B, S, H, D) → (B, S, H*D)
            output = output.reshape(batch_size, max_seqlen_q, -1)
        elif q_format == "thd":
            # (total, H, D) → (total, H*D)
            output = output.reshape(output.shape[0], -1)

        return output.contiguous()
