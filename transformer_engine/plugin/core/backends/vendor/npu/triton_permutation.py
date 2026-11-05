# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""NPU implementations of TE's Triton permutation APIs.

Each function mirrors one public function in
``transformer_engine/pytorch/triton/permutation.py``: same name, same
parameter names and order, same return arity. Dispatch forwards ``*args,
**kwargs`` untouched and never adapts a signature, and the middle layer does
pass some arguments by keyword, so a renamed or reordered parameter breaks the
call rather than being adjusted.

Shapes and dtypes are documented on the TE functions and not restated here.

An implementation must be semantically equivalent to the function it replaces,
**including the intermediates it produces or consumes**. ``make_row_id_map``
builds the ``row_id_map`` the permute and unpermute APIs read. Such a value is
opaque to TE, so a vendor may encode it its own way -- but then every API
touching it must come from the same vendor, or a native consumer misreads it
and returns a wrong result with no error. Given an equivalent encoding, any
single API can be replaced on its own.

TE pins the mask-map layout from outside this file: ``register_fake`` in
``pytorch/permutation.py`` hardcodes the shape and ``moe_unpermute`` recovers
``num_experts`` from it. A wrong output shape or dtype is not validated in
eager; it surfaces as ``AssertionError: ... incorrect fake`` only under
``torch.compile``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _not_implemented(api_name: str) -> NotImplementedError:
    """Error raised while an NPU kernel is missing.

    Dispatch never retries, so a placeholder return value would be consumed
    as a real result.
    """
    return NotImplementedError(
        f"NPU implementation for Triton API '{api_name}' is not available yet. "
        "Unregister it in register_ops.py to fall back to te.native."
    )


def make_row_id_map(
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
) -> torch.Tensor:
    """Build the mask-map token-to-expert map.

    Building the map is a separate step from permuting it.
    """
    raise _not_implemented("make_row_id_map")


def permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
    scale_hidden_dim: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Permute tokens according to a mask-map ``row_id_map``.

    ``permuted_probs`` keeps ``probs.dtype``; it does not adopt ``inp.dtype``.
    """
    raise _not_implemented("permute_with_mask_map")


def unpermute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor],
    permuted_probs: Optional[torch.Tensor],
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Scatter permuted tokens back, optionally weighting by merging probs."""
    raise _not_implemented("unpermute_with_mask_map")


def unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward of the merging-probs unpermute.

    Reads the same ``row_id_map`` as :func:`unpermute_with_mask_map`.
    """
    raise _not_implemented("unpermute_with_mask_map_bwd_with_merging_probs")


def make_chunk_sort_map(
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    num_splits: int,
) -> torch.Tensor:
    """Build the chunk-sort row map."""
    raise _not_implemented("make_chunk_sort_map")


def sort_chunks_by_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reorder token chunks using a chunk-sort ``row_id_map``.

    Both directions come through this one API via ``is_forward``, so
    registering it covers the backward pass too.
    """
    raise _not_implemented("sort_chunks_by_map")
