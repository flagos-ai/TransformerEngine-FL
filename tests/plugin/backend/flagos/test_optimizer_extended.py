import os

import pytest
import torch

os.environ["TE_FL_PREFER"] = "flagos"
pytest.importorskip("flag_gems")

from transformer_engine import te_device_type
from transformer_engine.plugin.core import get_manager, get_tefl_module


@pytest.mark.skipif(
    te_device_type() != "cuda" or not torch.cuda.is_available(),
    reason="torch.cuda.CUDAGraph is CUDA-only",
)
@pytest.mark.parametrize("master_weights", [False, True])
def test_capturable_adam_cuda_graph_replay(master_weights):
    device = torch.device(te_device_type())
    model_param = torch.randn(64, device=device, dtype=torch.bfloat16)
    master_param = model_param.float().clone()
    grad = torch.randn(64, device=device, dtype=torch.bfloat16)
    exp_avg = torch.randn(64, device=device)
    exp_avg_sq = torch.rand(64, device=device)
    noop = torch.zeros(1, device=device, dtype=torch.int)
    lr = torch.tensor(0.003, device=device)
    step = torch.tensor(4, device=device, dtype=torch.int)
    inv_scale = torch.tensor(0.25, device=device)
    tensor_lists = [[grad], [model_param], [exp_avg], [exp_avg_sq]]
    op_name = "multi_tensor_adam_capturable"
    if master_weights:
        tensor_lists.append([master_param])
        op_name = "multi_tensor_adam_capturable_master"
    op = getattr(get_tefl_module(), op_name)

    warmup = [[tensor.clone() for tensor in tensor_list] for tensor_list in tensor_lists]
    op(2048, noop, warmup, lr, 0.9, 0.95, 1e-8, step, 1, 1, 0.1, inv_scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(2048, noop, tensor_lists, lr, 0.9, 0.95, 1e-8, step, 1, 1, 0.1, inv_scale)

    before = [tensor.clone() for tensor_list in tensor_lists for tensor in tensor_list]
    noop.fill_(1)
    graph.replay()
    after_noop = [tensor for tensor_list in tensor_lists for tensor in tensor_list]
    assert all(torch.equal(actual, expected) for actual, expected in zip(after_noop, before))

    noop.zero_()
    lr.fill_(0.001)
    step.fill_(5)
    inv_scale.fill_(0.5)
    graph.replay()
    assert get_manager().get_selected_impl_id(op_name) == "default.flagos"
    updated_param = master_param if master_weights else model_param
    expected_before = before[-1] if master_weights else before[1]
    assert not torch.equal(updated_param, expected_before)
    if master_weights:
        assert torch.equal(model_param, master_param.bfloat16())


@pytest.mark.parametrize(
    "fp8_dtype,torch_dtype", [(7, torch.float8_e4m3fn), (8, torch.float8_e5m2)]
)
def test_fp8_adam_updates_master_and_metadata(fp8_dtype, torch_dtype):
    device = torch.device(te_device_type())
    grad = torch.randn(64, device=device)
    master = torch.randn(64, device=device)
    param = master.to(torch_dtype).view(torch.uint8).clone()
    exp_avg = torch.randn(64, device=device)
    exp_avg_sq = torch.rand(64, device=device)
    scale = torch.tensor(2.0, device=device)
    amax = torch.zeros(1, device=device)
    scale_inv = torch.zeros(1, device=device)
    get_tefl_module().multi_tensor_adam_fp8(
        2048,
        torch.zeros(1, device=device, dtype=torch.int),
        [[grad], [param], [exp_avg], [exp_avg_sq], [master], [scale], [amax], [scale_inv]],
        0.003,
        0.9,
        0.95,
        1e-8,
        4,
        1,
        1,
        0.1,
        fp8_dtype,
    )
    assert get_manager().get_selected_impl_id("multi_tensor_adam_fp8") == "default.flagos"
    assert torch.equal(param, (master * scale).to(torch_dtype).view(torch.uint8))
    torch.testing.assert_close(amax, master.abs().max().reshape_as(amax))
    torch.testing.assert_close(scale_inv, scale.reciprocal().reshape_as(scale_inv))
