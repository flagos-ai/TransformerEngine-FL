import torch
import torch.distributed as dist
import torch_npu
from torch.autograd.function import once_differentiable


def _communicator(group, device):
    backend = group._get_backend(device)
    backend_name = backend.name()
    if backend_name == "hccl":
        return backend.get_hccl_comm_name(dist.get_rank())
    if backend_name == "flagcx":
        import flagcx

        get_name = getattr(flagcx, "get_hccl_comm_name", None)
        if get_name is not None:
            return get_name(backend, device.index)
    return None


def supports_moe_group(group, device):
    """Probe the existing communicator; all EP ranks must participate."""
    return _communicator(group, device) is not None


def _group_sizes(counts, experts, device):
    return torch.tensor(counts, dtype=torch.int64, device=device).reshape(-1, experts).sum(0)


def _weight_gradient(grad, x, sizes):
    return torch_npu.npu_grouped_matmul(
        [grad.t()], [x], group_list=sizes, group_list_type=1, split_item=3, group_type=2
    )[0]


class _AllToAllVGroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group, send_counts, recv_counts):
        name = _communicator(group, x.device)
        if name is None:
            raise NotImplementedError("MoE fusion requires an HCCL communicator")
        output, _, permuted = torch_npu.npu_alltoallv_gmm(
            x,
            weight,
            name,
            group.size(),
            send_counts,
            recv_counts,
            trans_gmm_weight=True,
            permute_out_flag=ctx.needs_input_grad[1],
        )
        ctx.save_for_backward(permuted, weight)
        ctx.group, ctx.name = group, name
        ctx.send_counts, ctx.recv_counts = send_counts, recv_counts
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad = grad.contiguous()
        dx, _ = torch_npu.npu_gmm_alltoallv(
            grad,
            weight,
            ctx.name,
            ctx.group.size(),
            ctx.recv_counts,
            ctx.send_counts,
        )
        dw = None
        if ctx.needs_input_grad[1]:
            sizes = _group_sizes(ctx.recv_counts, weight.shape[0], grad.device)
            dw = _weight_gradient(grad, x, sizes)
        return dx if ctx.needs_input_grad[0] else None, dw, None, None, None


class _GroupedMatmulAllToAllV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group, send_counts, recv_counts):
        name = _communicator(group, x.device)
        if name is None:
            raise NotImplementedError("MoE fusion requires an HCCL communicator")
        output, _ = torch_npu.npu_gmm_alltoallv(
            x,
            weight,
            name,
            group.size(),
            recv_counts,
            send_counts,
            trans_gmm_weight=True,
        )
        ctx.save_for_backward(x, weight)
        ctx.group, ctx.name = group, name
        ctx.send_counts, ctx.recv_counts = send_counts, recv_counts
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dx, _, permuted = torch_npu.npu_alltoallv_gmm(
            grad.contiguous(),
            weight,
            ctx.name,
            ctx.group.size(),
            ctx.send_counts,
            ctx.recv_counts,
            permute_out_flag=ctx.needs_input_grad[1],
        )
        dw = None
        if ctx.needs_input_grad[1]:
            sizes = _group_sizes(ctx.recv_counts, weight.shape[0], grad.device)
            dw = _weight_gradient(permuted, x, sizes)
        return dx if ctx.needs_input_grad[0] else None, dw, None, None, None


def _packed_weight(weights, dtype):
    # Stack only the legacy tuple input; preserve the parent Parameter's gradient path.
    weight = weights if isinstance(weights, torch.Tensor) else torch.stack(tuple(weights))
    return weight.to(dtype).contiguous()


def alltoallv_grouped_matmul(x, weights, group, send_counts, recv_counts):
    """Exchange expert tokens and apply FC1; differentiate inputs and weights."""
    return _AllToAllVGroupedMatmul.apply(
        x.contiguous(),
        _packed_weight(weights, x.dtype),
        group,
        tuple(send_counts),
        tuple(recv_counts),
    )


def grouped_matmul_alltoallv(x, weights, group, send_counts, recv_counts):
    """Apply FC2 and return tokens to their source ranks, including backward."""
    return _GroupedMatmulAllToAllV.apply(
        x.contiguous(),
        _packed_weight(weights, x.dtype),
        group,
        tuple(send_counts),
        tuple(recv_counts),
    )


def is_moe_available():
    return all(
        callable(getattr(torch_npu, name, None))
        for name in (
            "npu_alltoallv_gmm",
            "npu_gmm_alltoallv",
            "npu_grouped_matmul",
        )
    )
