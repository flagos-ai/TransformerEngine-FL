# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Ascend NPU adapter for Transformer Engine's public permutation API."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ....ops import PermutationBase


def _get_torch_npu():
    import torch_npu

    return torch_npu


def _get_tenpu_permutation():
    from transformer_engine_npu.pytorch import permutation

    return permutation


class NPUPermutation(PermutationBase):
    """Adapt TE's public permutation API to TransformerEngineNPU."""

    def _moe_permute_mask_map(
        self,
        inp: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
        max_token_num: int,
    ):
        from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

        if not inp.numel() or isinstance(inp, QuantizedTensor):
            return super()._moe_permute_mask_map(
                inp, routing_map, num_out_tokens, max_token_num
            )

        num_out_tokens = self._normalize_num_out_tokens(
            routing_map, num_out_tokens
        )
        return _get_tenpu_permutation().moe_permute(
            inp,
            routing_map,
            num_out_tokens,
            max_token_num=max_token_num,
            map_type="mask",
        )

    def _moe_permute_mask_map_with_probs(
        self,
        inp: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
    ):
        from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

        if not inp.numel() or isinstance(inp, QuantizedTensor):
            return super()._moe_permute_mask_map_with_probs(
                inp, probs, routing_map, num_out_tokens
            )

        num_out_tokens = self._normalize_num_out_tokens(
            routing_map, num_out_tokens
        )
        # TransformerEngineNPU currently restores both outputs to inp.dtype
        # when inp is BF16 and probs is FP32. TE requires permuted_probs to
        # preserve probs.dtype, so use the same underlying torch_npu kernel
        # directly until the public TENPU interface preserves both dtypes.
        return _get_torch_npu().npu_moe_token_permute_with_routing_map(
            inp,
            routing_map,
            probs=probs,
            num_out_tokens=num_out_tokens,
            drop_and_pad=False,
        )

    def _moe_unpermute_mask_map(
        self,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        merging_probs: Optional[torch.Tensor],
        restore_shape: Optional[torch.Size],
        pad_offsets: Optional[torch.Tensor],
    ):
        from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

        supported = (
            inp.numel()
            and not isinstance(inp, QuantizedTensor)
            and pad_offsets is None
            and row_id_map.ndim == 1
            and row_id_map.dtype == torch.int32
        )
        if not supported:
            return super()._moe_unpermute_mask_map(
                inp,
                row_id_map,
                merging_probs,
                restore_shape,
                pad_offsets,
            )

        if restore_shape is None:
            restore_shape = inp.shape
        if merging_probs is None:
            return _get_tenpu_permutation().moe_unpermute(
                inp,
                row_id_map,
                restore_shape=restore_shape,
                map_type="mask",
            )

        # TE's public API does not pass the original routing map required by
        # TransformerEngineNPU's weighted mask-map unpermute. Megatron's fused
        # path provides already-permuted 1D probabilities, so apply them before
        # calling the underlying torch_npu unpermute operation.
        if merging_probs.ndim != 1 or merging_probs.size(0) != inp.size(0):
            return super()._moe_unpermute_mask_map(
                inp,
                row_id_map,
                merging_probs,
                restore_shape,
                pad_offsets,
            )
        if merging_probs is not None:
            inp = inp * merging_probs.unsqueeze(-1)
        return _get_torch_npu().npu_moe_token_unpermute_with_routing_map(
            inp,
            row_id_map,
            list(restore_shape),
            drop_and_pad=False,
        )

    def moe_sort_chunks_by_index(
        self,
        inp: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_index: torch.Tensor,
    ) -> torch.Tensor:
        """Use TransformerEngineNPU's chunk-sort implementation on Ascend."""
        return _get_tenpu_permutation().moe_sort_chunks_by_index(
            inp, split_sizes, sorted_index
        )

    def moe_sort_chunks_by_index_with_probs(
        self,
        inp: torch.Tensor,
        probs: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sort token chunks and probabilities with TransformerEngineNPU."""
        return _get_tenpu_permutation().moe_sort_chunks_by_index_with_probs(
            inp, probs, split_sizes, sorted_index
        )

    @staticmethod
    def _normalize_num_out_tokens(
        routing_map: torch.Tensor,
        num_out_tokens: int,
    ) -> int:
        if num_out_tokens is None or num_out_tokens < 0:
            num_out_tokens = int(routing_map.sum().item())
        return num_out_tokens
