# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""The list of Triton APIs routed through plugin dispatch.

One entry per API: dispatch selects per operator, so a vendor replaces the
implementations it has one at a time and every API it does not claim stays on
TE's own.

Hand-maintained because there is no module boundary to swap wholesale, unlike
the tex operators (taken over by replacing ``transformer_engine_torch`` in
``sys.modules``). These are plain functions across five modules, mixed with
``@triton.jit`` kernels, imported names, and a public function that is not an
operator.

The authoritative source is the public function definitions in
``pytorch/triton/*.py``, not how consumers import them.
``test_triton_dispatch_manifest.py`` re-derives the list from those sources
and fails when it drifts.
"""

from __future__ import annotations

from types import MappingProxyType


_TARGET_PREFIX = "transformer_engine.pytorch.triton."

TARGET_APIS = MappingProxyType(
    {
        f"{_TARGET_PREFIX}pad": ("pad_columnwise_scale_inv",),
        f"{_TARGET_PREFIX}cross_entropy": (
            "cross_entropy_forward",
            "cross_entropy_backward",
        ),
        f"{_TARGET_PREFIX}permutation": (
            "make_row_id_map",
            "permute_with_mask_map",
            "unpermute_with_mask_map",
            "unpermute_with_mask_map_bwd_with_merging_probs",
            "make_chunk_sort_map",
            "sort_chunks_by_map",
        ),
        f"{_TARGET_PREFIX}grouped_dbias_dscales": (
            "compute_grouped_dbias",
            "compute_grouped_dbias_dscales",
        ),
        f"{_TARGET_PREFIX}mhc": (
            "mhc_fused_sinkhorn",
            "mhc_fused_scale",
            "mhc_fused_aggregate",
            "mhc_fused_expand_combine",
            "mhc_fused_projection",
        ),
    }
)

# Public functions that pass the hook's three checks but are not operators,
# so they must not be dispatched. ``check_deterministic`` only asserts on an
# environment variable. Only a list can tell these apart from real APIs.
EXCLUDED_PUBLIC_FUNCTIONS = MappingProxyType(
    {f"{_TARGET_PREFIX}mhc": frozenset({"check_deterministic"})}
)

TARGET_MODULES = frozenset(TARGET_APIS)

# The ``triton_`` prefix keeps these op names clear of the tex operators in
# the same registry. Vendors register against these names.
TARGET_OPS = MappingProxyType(
    {
        (module_name, api_name): f"triton_{api_name}"
        for module_name, api_names in TARGET_APIS.items()
        for api_name in api_names
    }
)
