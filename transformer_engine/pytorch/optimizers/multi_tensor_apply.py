# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-tensor apply entry."""
import torch
from torch.distributed._tensor import DTensor


class MultiTensorApply:  # pylint: disable=too-few-public-methods
    """Multi-tensor apply entry."""

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        for i, ts in enumerate(tensor_lists):
            for j, t in enumerate(ts):
                if isinstance(t, DTensor):
                    tensor_lists[i][j] = t._local_tensor

        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


multi_tensor_applier = MultiTensorApply(2048 * 32)


# TODO(lixianduo): polish
# computes l2 norm for a list of contiguous tensors
# works as a drop-in replacement for amp_C.multi_tensor_l2norm
def native_multi_tensor_l2_norm(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
    """
    Computes l2 norm for a list of contiguous tensors
    works as a drop-in replacement for amp_C.multi_tensor_l2norm
    """
    l2 = [[(torch.norm(tensor)) for tensor in tensor_list] for tensor_list in tensor_lists]
    l2_reduced = torch.norm(torch.tensor(l2))
    l2_cuda = torch.tensor([float(l2_reduced)], dtype=torch.float, device="cuda")
    return l2_cuda, None


# works as a drop-in replacement for amp_C.multi_tensor_scale
def native_multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
    """Works as a drop-in replacement for amp_C.multi_tensor_scale."""
    for src, dst in zip(tensor_lists[0], tensor_lists[1]):
        dst.copy_(src * scale)
