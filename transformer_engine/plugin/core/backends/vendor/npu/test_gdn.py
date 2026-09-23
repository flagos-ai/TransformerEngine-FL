"""NPU adapter contract and real-kernel parity. CPU mocks are not device evidence."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from transformer_engine.plugin.core.backends.vendor.npu import gated_delta_net as vendor
from transformer_engine.plugin.core.manager import OpManager


def tensors(seq=3, dtype=torch.bfloat16):
    q, k, v = [torch.randn(1, seq, 2, 128, dtype=dtype) * 0.05 for _ in range(3)]
    return [q, k, v, -torch.rand(1, seq, 2), torch.rand(1, seq, 2).to(dtype)]


class MetadataNPU:
    """CPU payload with simulated device metadata; no kernels are launched."""
    device = SimpleNamespace(type='npu')

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def simulated(xs):
    return [MetadataNPU(x) for x in xs]


def test_registration_without_fla_import(monkeypatch):
    from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend
    from transformer_engine.plugin.core.backends.vendor.npu.register_ops import register_builtins

    monkeypatch.setattr(NPUBackend, 'is_available', lambda self: True)
    probe = Mock(side_effect=AssertionError('Registration must not probe FLA'))
    monkeypatch.setattr(vendor, 'is_gated_delta_net_available', probe)
    registry = Mock()
    register_builtins(registry)
    impls = registry.register_many.call_args.args[0]
    gdn = [i for i in impls if i.op_name == 'gated_delta_net_forward']
    assert len(gdn) == 1 and gdn[0].is_available()
    probe.assert_not_called()


def test_availability_runs_validator(monkeypatch):
    from transformer_engine.plugin.core.backends.vendor.npu import gdn_operators

    probe = Mock(side_effect=RuntimeError('missing operator'))
    monkeypatch.setattr(gdn_operators, 'validate_runtime', probe)
    assert not vendor.is_gated_delta_net_available()
    probe.assert_called_once()


@pytest.mark.parametrize('chunk', [0, -1, 3, 1.5, True])
def test_bad_chunk_is_error(chunk):
    with pytest.raises(ValueError, match='positive power'):
        vendor.gated_delta_net_forward(*tensors(), chunk_size=chunk)


def test_bad_shape_is_error():
    xs = tensors()
    xs[2] = xs[2][:, :1]
    with pytest.raises(ValueError, match='matching BSH'):
        vendor.gated_delta_net_forward(*xs)


@pytest.mark.parametrize('case', ['cpu', 'fp32', 'empty', 'dimension', 'beta', 'g', 'chunk', 'state', 'final'])
def test_unsupported_declines_before_runtime(monkeypatch, case):
    xs = tensors()
    kw = {}
    if case == 'fp32':
        xs[:3] = [x.float() for x in xs[:3]]
    elif case == 'empty':
        xs = [x[:, :0] for x in xs]
    elif case == 'dimension':
        xs[2] = xs[2][..., :64]
    elif case == 'beta':
        xs[4] = xs[4].float()
    elif case == 'g':
        xs[3] = xs[3].to(torch.bfloat16)
    elif case == 'chunk':
        kw['chunk_size'] = 32
    elif case == 'state':
        kw['initial_state'] = MetadataNPU(torch.zeros(1, 2, 128, 128))
    elif case == 'final':
        kw['output_final_state'] = True
    probe = Mock(side_effect=AssertionError('Unsupported inputs must not load FLA'))
    monkeypatch.setattr(vendor, 'is_gated_delta_net_available', probe)
    assert vendor.gated_delta_net_forward(*(xs if case == 'cpu' else simulated(xs)), **kw) is NotImplemented
    probe.assert_not_called()


@pytest.mark.parametrize('strict', ['0', '1'])
@pytest.mark.parametrize('entry', ['call', 'resolve'])
def test_cached_callable_rechecks_and_recovers(monkeypatch, strict, entry):
    from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend
    monkeypatch.setattr(NPUBackend, 'is_available', lambda self: True)
    monkeypatch.setenv('TE_FL_STRICT', strict)
    monkeypatch.setenv('TE_FL_PER_OP', 'gated_delta_net_forward=impl:vendor.npu.gdn_fwd')
    ready = [False]
    monkeypatch.setattr(vendor, 'is_gated_delta_net_available', lambda: ready[0])
    kernel = Mock(side_effect=lambda **kw: (kw['v'].transpose(1, 2), None))
    monkeypatch.setitem(sys.modules, vendor.__package__ + '.gdn_impl', SimpleNamespace(flash_gated_delta_rule=kernel))
    manager = OpManager()
    invoke = (lambda *a, **kw: manager.call('gated_delta_net_forward', *a, **kw)) if entry == 'call' else manager.resolve('gated_delta_net_forward')
    xs = tensors()
    assert invoke(*simulated(xs)) is NotImplemented
    ready[0] = True
    out, _ = invoke(*simulated(xs))
    torch.testing.assert_close(out, xs[2])
    assert invoke(*simulated(xs), output_final_state=True) is NotImplemented
    invoke(*simulated(xs))
    ready[0] = False
    assert invoke(*simulated(xs)) is NotImplemented
    ready[0] = True
    invoke(*simulated(xs))
    assert kernel.call_count == 3
    assert manager.get_selected_impl_id('gated_delta_net_forward') == 'vendor.npu.gdn_fwd'


@pytest.mark.parametrize('strict', ['0', '1'])
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('error', [ValueError, RuntimeError, torch.OutOfMemoryError])
def test_kernel_errors_never_become_decline(monkeypatch, strict, cached, error):
    from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend
    monkeypatch.setattr(NPUBackend, 'is_available', lambda self: True)
    monkeypatch.setenv('TE_FL_STRICT', strict)
    monkeypatch.setattr(vendor, 'is_gated_delta_net_available', lambda: True)
    kernel = Mock(return_value=(torch.zeros(1, 3, 2, 128), None))
    monkeypatch.setitem(sys.modules, vendor.__package__ + '.gdn_impl', SimpleNamespace(flash_gated_delta_rule=kernel))
    manager = OpManager()
    if cached:
        manager.call('gated_delta_net_forward', *simulated(tensors()))
    kernel.side_effect = error('injected kernel failure')
    # The unmodified manager may wrap an exception; it must not return fallback.
    with pytest.raises((error, RuntimeError), match='injected kernel failure'):
        manager.call('gated_delta_net_forward', *simulated(tensors()))


def reference(q, k, v, g, beta, normalize=False):
    """Independent recurrence used only as the numerical test oracle."""
    dtype = q.dtype
    if normalize:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q, k, v, g, beta = [x.float() for x in (q, k, v, g, beta)]
    state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], device=q.device)
    ys = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        residual = v[:, t] - (k[:, t, :, :, None] * state).sum(-2)
        state = state + k[:, t, :, :, None] * (beta[:, t, :, None] * residual)[..., None, :]
        ys.append((q[:, t, :, :, None] * state).sum(-2) / q.shape[-1] ** 0.5)
    return torch.stack(ys, dim=1).to(dtype)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('seq,normalize,noncontiguous', [(64, False, False), (65, False, False), (128, True, False), (64, False, True)])
def test_real_npu_forward_backward(dtype, seq, normalize, noncontiguous, monkeypatch):
    import json
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), 'Real NPU tests require a healthy device'
    monkeypatch.setenv('TE_FL_STRICT', '0')
    torch.manual_seed(42)
    xs = tensors(seq, dtype)
    xs[:2] = [torch.nn.functional.normalize(x.float(), dim=-1).to(dtype) for x in xs[:2]]
    if noncontiguous:
        xs = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in xs]
    cpu = [x.detach().clone().requires_grad_() for x in xs]
    device = os.environ.get('GDN_TEST_DEVICE', 'npu:0')
    npu = [x.to(device).detach().requires_grad_() for x in xs]
    manager = OpManager()
    actual, final = manager.call('gated_delta_net_forward', *npu, use_qk_l2norm=normalize)
    assert final is None
    assert manager.get_selected_impl_id('gated_delta_net_forward') == 'vendor.npu.gdn_fwd'
    expected = reference(*cpu, normalize=normalize)
    dy = torch.randn_like(expected) * 0.05
    expected.backward(dy)
    actual.backward(dy.to(device))
    torch.npu.synchronize()
    for name, a, b in [('output', actual, expected)] + [(n, a.grad, b.grad) for n, a, b in zip(('q', 'k', 'v', 'g', 'beta'), npu, cpu)]:
        assert a is not None and b is not None
        a, b = a.detach().float().cpu(), b.detach().float().cpu()
        diff = (a - b).abs()
        l2 = (a - b).norm() / b.norm().clamp_min(1e-12)
        atol, rtol, limit = (1e-3, 2e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-2, 3e-3)
        print('GDN_ERROR_METRICS=' + json.dumps(dict(dtype=str(dtype), seq=seq, normalize=normalize, noncontiguous=noncontiguous, tensor=name, relative_l2=l2.item(), max_abs=diff.max().item(), mean_abs=diff.mean().item(), p99_abs=diff.quantile(.99).item(), violations=int((diff > atol + rtol * b.abs()).sum()), atol=atol, rtol=rtol, l2_limit=limit)))
        assert torch.isfinite(a).all() and l2 < limit
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
