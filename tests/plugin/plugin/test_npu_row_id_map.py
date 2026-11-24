# Copyright (c) 2026, BAAI. All rights reserved.
# See LICENSE for license information.

"""NPU row maps must preserve TE's compact descending-expert layout."""

import numpy as np
import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")

from transformer_engine.plugin.core.backends.vendor.npu.triton_permutation import make_row_id_map


def _reference_row_map(routing):
    tokens, experts = routing.shape
    expected = np.zeros((tokens, 2 * experts + 1), dtype=np.int32)
    expected[:, -1] = routing.sum(axis=1)
    offset = 0
    for expert in range(experts):
        selected = np.flatnonzero(routing[:, expert])
        # TE stores each token's routes from the largest expert to the smallest.
        slot = routing[selected, expert + 1 :].sum(axis=1)
        expected[selected, slot] = offset + np.arange(len(selected))
        expected[selected, experts + slot] = expert
        offset += len(selected)
    return expected


@pytest.mark.parametrize(
    "tokens,experts", [(0, 7), (9, 0), (7, 8), (129, 19), (257, 256), (16384, 256)]
)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_row_map_matches_independent_routing_reference(tokens, experts, dtype):
    rng = np.random.default_rng(233 + tokens)
    routing = rng.random((tokens, experts)) < min(0.5, 8 / max(1, experts))
    if tokens:
        routing[0, :] = False
    if experts:
        routing[:, 0] = False
    device_routing = torch.from_numpy(routing).to(device="npu", dtype=dtype)
    actual = make_row_id_map(device_routing, tokens, experts)
    assert actual.dtype == torch.int32
    np.testing.assert_array_equal(actual.cpu().numpy(), _reference_row_map(routing))


def test_row_map_accepts_noncontiguous_routing():
    rng = np.random.default_rng(852)
    routing = rng.random((131, 19)) < 0.3
    device_routing = torch.empty((131, 38), device="npu", dtype=torch.bool)[:, ::2]
    device_routing.copy_(torch.from_numpy(routing).to("npu"))
    actual = make_row_id_map(device_routing, 131, 19)
    np.testing.assert_array_equal(actual.cpu().numpy(), _reference_row_map(routing))


@pytest.mark.parametrize("hint", [None, 131 * 3, 131 * 3 + 32])
def test_output_count_hint_preserves_row_map_and_plugin_dispatch(hint):
    import inspect
    from transformer_engine.pytorch.triton import permutation as public
    from transformer_engine.plugin.core.manager import get_default_manager

    rng = np.random.default_rng(152)
    routing = np.zeros((131, 19), dtype=np.bool_)
    chosen = np.argsort(rng.random((131, 19)), axis=1)[:, :3]
    routing[np.arange(131)[:, None], chosen] = True
    device_routing = torch.from_numpy(routing).to("npu")
    manager = get_default_manager()
    assert manager.get_selected_impl_id("triton_make_row_id_map") == "vendor.npu"
    parameter = inspect.signature(make_row_id_map).parameters["num_out_tokens"]
    assert parameter.default is None
    expected = _reference_row_map(routing)
    without_hint = public.make_row_id_map(device_routing, 131, 19)
    positional_hint = make_row_id_map(device_routing, 131, 19, hint)
    keyword_hint = make_row_id_map(device_routing, 131, 19, num_out_tokens=hint)
    for result in (without_hint, positional_hint, keyword_hint):
        np.testing.assert_array_equal(result.cpu().numpy(), expected)
