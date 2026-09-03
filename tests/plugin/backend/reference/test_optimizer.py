import pytest
import torch

from transformer_engine.plugin.core.backends.reference.impl.optimizer import (
    multi_tensor_adam_capturable_master_torch,
    multi_tensor_adam_capturable_torch,
    multi_tensor_adam_torch,
)


def _expected(
    param, grad, exp_avg, exp_avg_sq, *, mode, bias_correction, weight_decay, grad_scale=1.0
):
    lr, beta1, beta2, epsilon, step = 0.003, 0.9, 0.95, 1e-8, 4
    scaled_grad = grad.float() * grad_scale
    if mode == 0:
        scaled_grad = scaled_grad + weight_decay * param
    expected_avg = beta1 * exp_avg + (1 - beta1) * scaled_grad
    expected_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * scaled_grad.square()
    correction1 = 1 - beta1**step if bias_correction else 1.0
    correction2 = 1 - beta2**step if bias_correction else 1.0
    update = (expected_avg / correction1) / ((expected_avg_sq / correction2).sqrt() + epsilon)
    if mode == 1:
        update = update + weight_decay * param
    return param - lr * update, expected_avg, expected_avg_sq


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
@pytest.mark.parametrize("master_weights", [False, True])
def test_multi_tensor_adam_matches_formula(mode, bias_correction, master_weights):
    torch.manual_seed(7)
    model_param = torch.randn(128, dtype=torch.bfloat16)
    master_param = model_param.float().clone()
    grad = torch.randn(128, dtype=torch.bfloat16)
    exp_avg = torch.randn(128)
    exp_avg_sq = torch.rand(128)
    grad_before = grad.clone()
    expected = _expected(
        master_param,
        grad,
        exp_avg,
        exp_avg_sq,
        mode=mode,
        bias_correction=bias_correction,
        weight_decay=0.1,
    )
    tensor_lists = [
        [grad],
        [model_param if master_weights else master_param],
        [exp_avg],
        [exp_avg_sq],
    ]
    if master_weights:
        tensor_lists.append([master_param])
    multi_tensor_adam_torch(
        2048,
        torch.zeros(1, dtype=torch.int),
        tensor_lists,
        0.003,
        0.9,
        0.95,
        1e-8,
        4,
        mode,
        bias_correction,
        0.1,
    )
    torch.testing.assert_close(master_param, expected[0])
    torch.testing.assert_close(exp_avg, expected[1])
    torch.testing.assert_close(exp_avg_sq, expected[2])
    assert torch.equal(grad, grad_before)
    if master_weights:
        assert torch.equal(model_param, master_param.bfloat16())


@pytest.mark.parametrize("master_weights", [False, True])
def test_capturable_fallback_applies_inv_scale(master_weights):
    torch.manual_seed(11)
    model_param = torch.randn(64, dtype=torch.bfloat16)
    master_param = model_param.float().clone()
    grad = torch.randn(64, dtype=torch.bfloat16)
    exp_avg = torch.randn(64)
    exp_avg_sq = torch.rand(64)
    expected = _expected(
        master_param,
        grad,
        exp_avg,
        exp_avg_sq,
        mode=1,
        bias_correction=1,
        weight_decay=0.1,
        grad_scale=0.25,
    )
    tensor_lists = [
        [grad],
        [model_param if master_weights else master_param],
        [exp_avg],
        [exp_avg_sq],
    ]
    function = multi_tensor_adam_capturable_torch
    if master_weights:
        tensor_lists.append([master_param])
        function = multi_tensor_adam_capturable_master_torch
    function(
        2048,
        torch.zeros(1, dtype=torch.int),
        tensor_lists,
        torch.tensor(0.003),
        0.9,
        0.95,
        1e-8,
        torch.tensor(4),
        1,
        1,
        0.1,
        torch.tensor(0.25),
    )
    torch.testing.assert_close(master_param, expected[0])
    torch.testing.assert_close(exp_avg, expected[1])
    torch.testing.assert_close(exp_avg_sq, expected[2])
    if master_weights:
        assert torch.equal(model_param, master_param.bfloat16())


def test_multi_tensor_adam_noop_preserves_state():
    tensors = [torch.randn(16) for _ in range(4)]
    before = [tensor.clone() for tensor in tensors]
    multi_tensor_adam_torch(
        2048,
        torch.ones(1, dtype=torch.int),
        [[tensor] for tensor in tensors],
        0.003,
        0.9,
        0.95,
        1e-8,
        4,
        1,
        1,
        0.1,
    )
    assert all(torch.equal(actual, expected) for actual, expected in zip(tensors, before))
