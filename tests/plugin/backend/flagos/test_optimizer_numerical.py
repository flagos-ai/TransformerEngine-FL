import os

import pytest
import torch

os.environ["TE_FL_PREFER"] = "flagos"
pytest.importorskip("flag_gems")

from transformer_engine import te_device_type
from transformer_engine.plugin.core import get_manager, get_tefl_module


def _expected(
    param, grad, exp_avg, exp_avg_sq, *, mode, bias_correction, weight_decay, grad_scale=1.0
):
    lr, beta1, beta2, epsilon, step = 0.003, 0.9, 0.95, 1e-8, 4
    adam_grad = grad.float() * grad_scale
    if mode == 0:
        adam_grad = adam_grad + weight_decay * param
    expected_avg = beta1 * exp_avg + (1 - beta1) * adam_grad
    expected_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * adam_grad.square()
    correction1 = 1 - beta1**step if bias_correction else 1.0
    correction2 = 1 - beta2**step if bias_correction else 1.0
    update = (expected_avg / correction1) / ((expected_avg_sq / correction2).sqrt() + epsilon)
    if mode == 1:
        update = update + weight_decay * param
    return param - lr * update, expected_avg, expected_avg_sq


def _run_adam(tensor_lists, noop_flag, mode, bias_correction, weight_decay):
    get_tefl_module().multi_tensor_adam(
        2048,
        noop_flag,
        tensor_lists,
        0.003,
        0.9,
        0.95,
        1e-8,
        4,
        mode,
        bias_correction,
        weight_decay,
    )
    selected = get_manager().get_selected_impl_id("multi_tensor_adam")
    print(f"selected multi_tensor_adam implementation: {selected}")
    assert selected == "default.flagos"


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("master_weights", [False, True])
def test_multi_tensor_adam_matches_formula(mode, bias_correction, weight_decay, master_weights):
    torch.manual_seed(7)
    device = torch.device(te_device_type())
    model_param = torch.randn(128, device=device, dtype=torch.bfloat16)
    master_param = model_param.float().clone()
    grad = torch.randn(128, device=device, dtype=torch.bfloat16)
    exp_avg = torch.randn(128, device=device)
    exp_avg_sq = torch.rand(128, device=device)
    grad_before = grad.clone()
    expected = _expected(
        master_param.clone(),
        grad,
        exp_avg.clone(),
        exp_avg_sq.clone(),
        mode=mode,
        bias_correction=bias_correction,
        weight_decay=weight_decay,
    )
    tensor_lists = [
        [grad],
        [model_param if master_weights else master_param],
        [exp_avg],
        [exp_avg_sq],
    ]
    if master_weights:
        tensor_lists.append([master_param])
    _run_adam(
        tensor_lists,
        torch.zeros(1, device=device, dtype=torch.int),
        mode,
        bias_correction,
        weight_decay,
    )
    torch.testing.assert_close(master_param, expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(exp_avg, expected[1], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(exp_avg_sq, expected[2], rtol=2e-5, atol=2e-6)
    assert torch.equal(grad, grad_before)
    if master_weights:
        assert torch.equal(model_param, master_param.bfloat16())


def test_capturable_adam_matches_formula_with_inv_scale_and_master_writeback():
    device = torch.device(te_device_type())
    for master_weights in (False, True):
        torch.manual_seed(19)
        model_param = torch.randn(128, device=device, dtype=torch.bfloat16)
        master_param = model_param.float().clone()
        grad = torch.randn(128, device=device, dtype=torch.bfloat16)
        exp_avg = torch.randn(128, device=device)
        exp_avg_sq = torch.rand(128, device=device)
        expected = _expected(
            master_param.clone(),
            grad,
            exp_avg.clone(),
            exp_avg_sq.clone(),
            mode=1,
            bias_correction=1,
            weight_decay=0.1,
            grad_scale=0.25,
        )
        tensor_lists = [[grad], [model_param], [exp_avg], [exp_avg_sq]]
        op_name = "multi_tensor_adam_capturable"
        if master_weights:
            tensor_lists.append([master_param])
            op_name = "multi_tensor_adam_capturable_master"
        getattr(get_tefl_module(), op_name)(
            2048,
            torch.zeros(1, device=device, dtype=torch.int),
            tensor_lists,
            torch.tensor(0.003, device=device),
            0.9,
            0.95,
            1e-8,
            torch.tensor(4, device=device, dtype=torch.int),
            1,
            1,
            0.1,
            torch.tensor(0.25, device=device),
        )
        updated_param = master_param if master_weights else model_param
        if master_weights:
            torch.testing.assert_close(updated_param, expected[0], rtol=2e-5, atol=2e-6)
        else:
            assert torch.equal(updated_param, expected[0].to(updated_param.dtype))
        torch.testing.assert_close(exp_avg, expected[1], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(exp_avg_sq, expected[2], rtol=2e-5, atol=2e-6)
        assert get_manager().get_selected_impl_id(op_name) == "default.flagos"
        if master_weights:
            assert torch.equal(model_param, master_param.bfloat16())


def test_multi_tensor_adam_noop_preserves_state():
    device = torch.device(te_device_type())
    tensors = [torch.randn(16, device=device) for _ in range(4)]
    before = [tensor.clone() for tensor in tensors]
    _run_adam(
        [[tensor] for tensor in tensors],
        torch.ones(1, device=device, dtype=torch.int),
        mode=1,
        bias_correction=1,
        weight_decay=0.1,
    )
    assert all(torch.equal(actual, expected) for actual, expected in zip(tensors, before))
