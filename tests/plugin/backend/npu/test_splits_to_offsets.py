# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""NPU backend tests for TE-FL splits_to_offsets."""

import pytest
import torch


try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except (ImportError, AttributeError):
    _HAS_NPU = False


pytestmark = pytest.mark.skipif(not _HAS_NPU, reason="NPU not available")


@pytest.mark.parametrize(
    "first_dims,logical_last_dim,expected",
    [
        ([2, 0, 3], 1, [0, 2, 2, 5]),
        ([2, 0, 3], 7, [0, 14, 14, 35]),
        ([1, 2, 3, 4], 128, [0, 128, 384, 768, 1280]),
    ],
)
def test_splits_to_offsets_values(first_dims, logical_last_dim, expected):
    from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend

    splits = torch.tensor(first_dims, dtype=torch.int64, device="npu")
    output = NPUBackend().splits_to_offsets(splits, logical_last_dim)

    assert output.device.type == "npu"
    assert output.dtype == torch.int64
    assert output.is_contiguous()
    torch.testing.assert_close(output.cpu(), torch.tensor(expected, dtype=torch.int64))


def test_splits_to_offsets_through_te_proxy(monkeypatch):
    import transformer_engine_torch as tex
    from transformer_engine.plugin.core.policy import reset_global_policy

    monkeypatch.setenv("TE_FL_PER_OP", "splits_to_offsets=impl:vendor.npu")
    reset_global_policy()

    splits = torch.tensor([3, 1, 0, 2], dtype=torch.int64, device="npu")
    output = tex.splits_to_offsets(splits, 16)
    torch.testing.assert_close(
        output.cpu(),
        torch.tensor([0, 48, 64, 64, 96], dtype=torch.int64),
    )


@pytest.mark.parametrize(
    "first_dims,logical_last_dim,error",
    [
        (torch.tensor([1, 2], dtype=torch.int32), 1, TypeError),
        (torch.tensor([[1, 2]], dtype=torch.int64), 1, ValueError),
        (torch.tensor([], dtype=torch.int64), 1, ValueError),
        (torch.tensor([1, 2], dtype=torch.int64), 0, ValueError),
    ],
)
def test_splits_to_offsets_rejects_invalid_inputs(first_dims, logical_last_dim, error):
    from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend

    with pytest.raises(error):
        NPUBackend().splits_to_offsets(first_dims.to("npu"), logical_last_dim)
