import pytest
import torch

from transformer_engine.plugin.core.backends.reference.impl.context_parallel import (
    thd_get_partitioned_indices_torch,
)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, [0, 1, 6, 7, 8, 11]),
        (1, [2, 3, 4, 5, 9, 10]),
    ],
)
def test_thd_get_partitioned_indices(rank, expected):
    cu_seqlens = torch.tensor([0, 8, 12], dtype=torch.int32)

    result = thd_get_partitioned_indices_torch(cu_seqlens, 12, 2, rank)

    assert result.dtype == torch.int32
    assert result.tolist() == expected


def test_thd_get_partitioned_indices_rejects_indivisible_sequences():
    cu_seqlens = torch.tensor([0, 6, 12], dtype=torch.int32)

    with pytest.raises(ValueError, match="each packed sequence length"):
        thd_get_partitioned_indices_torch(cu_seqlens, 12, 2, 0)
