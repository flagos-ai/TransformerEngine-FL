import torch
from torch.distributed._tensor import DTensor


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

