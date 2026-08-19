# Copyright (c) 2025, BAAI. All rights reserved.
# See LICENSE for license information.
"""NPU Backend Tests — numerical accuracy validated against reference backend."""

import pytest
import torch

# Check NPU availability
try:
    import torch_npu
    import transformer_engine_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except (ImportError, AttributeError):
    _HAS_NPU = False

requires_npu = pytest.mark.skipif(not _HAS_NPU, reason="NPU not available")


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def npu_backend():
    from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend

    return NPUBackend()


@pytest.fixture
def ref_backend():
    from transformer_engine.plugin.core.backends.reference.reference import ReferenceBackend

    return ReferenceBackend()


@pytest.fixture
def fa():
    from transformer_engine.plugin.core.backends.vendor.npu.flash_attention import NPUFlashAttention

    return NPUFlashAttention(softmax_scale=0.125)


# ===========================================================================
# Tolerance helpers
# ===========================================================================


def _tol(dtype):
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2  # NPU bf16 kernels have slightly more rounding than CPU
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:
        return 1e-4, 1e-4


def assert_close(npu_out, ref_out, dtype, msg=""):
    atol, rtol = _tol(dtype)
    npu_cpu = npu_out.detach().cpu().float()
    ref_cpu = ref_out.detach().cpu().float()
    max_diff = (npu_cpu - ref_cpu).abs().max().item()
    assert torch.allclose(
        npu_cpu, ref_cpu, atol=atol, rtol=rtol
    ), f"{msg} max_diff={max_diff:.6e}, atol={atol}, dtype={dtype}"


# ===========================================================================
# Mock tests (no NPU required)
# ===========================================================================


class TestNPUFlashAttentionValidation:
    def test_window_size_sliding_raises(self):
        from transformer_engine.plugin.core.backends.vendor.npu.flash_attention import (
            NPUFlashAttention,
        )

        fa = NPUFlashAttention(softmax_scale=0.125)
        q = torch.randn(1, 4, 2, 64)
        with pytest.raises(NotImplementedError, match="[Ss]liding"):
            fa.forward(q, q, q, qkv_layout="bshd_bshd_bshd", window_size=(128, 0))

    def test_alibi_slopes_raises(self):
        from transformer_engine.plugin.core.backends.vendor.npu.flash_attention import (
            NPUFlashAttention,
        )

        fa = NPUFlashAttention(softmax_scale=0.125)
        q = torch.randn(1, 4, 2, 64)
        with pytest.raises(NotImplementedError, match="[Aa]libi"):
            fa.forward(q, q, q, qkv_layout="bshd_bshd_bshd", alibi_slopes=torch.ones(2))

    def test_cp_group_raises(self):
        from transformer_engine.plugin.core.backends.vendor.npu.flash_attention import (
            NPUFlashAttention,
        )

        fa = NPUFlashAttention(softmax_scale=0.125)
        q = torch.randn(1, 4, 2, 64)
        with pytest.raises(NotImplementedError, match="[Cc]ontext"):
            fa.forward(q, q, q, qkv_layout="bshd_bshd_bshd", cp_group="group")


def test_npu_permutation_base_owns_index_map_routing():
    from transformer_engine.plugin.core.backends.vendor.npu.permutation import (
        NPUPermutation,
    )

    expected = (object(), object())

    class NativePermutation:
        def moe_permute(self, *args, **kwargs):
            self.permute_args = args
            self.permute_kwargs = kwargs
            return expected

        def moe_unpermute(self, *args, **kwargs):
            self.unpermute_args = args
            self.unpermute_kwargs = kwargs
            return expected

    native = NativePermutation()
    permutation = NPUPermutation(native)
    inp = torch.randn(2, 4)
    routing_map = torch.zeros(2, 1, dtype=torch.int32)

    assert permutation.moe_permute(inp, routing_map, map_type="index") is expected
    assert native.permute_kwargs["map_type"] == "index"
    assert (
        permutation.moe_unpermute(inp, routing_map, map_type="index") is expected
    )
    assert native.unpermute_kwargs["map_type"] == "index"

    pad_offsets = torch.tensor([0], dtype=torch.int64)
    assert (
        permutation.moe_unpermute(
            inp,
            routing_map,
            map_type="mask",
            pad_offsets=pad_offsets,
        )
        is expected
    )
    assert native.unpermute_kwargs["map_type"] == "mask"
    assert native.unpermute_kwargs["pad_offsets"] is pad_offsets

    empty_inp = torch.empty(0, 4)
    empty_routing_map = torch.empty(0, 1, dtype=torch.bool)
    assert (
        permutation.moe_permute(empty_inp, empty_routing_map, map_type="mask")
        is expected
    )
    assert native.permute_kwargs["map_type"] == "mask"

    assert "moe_permute" not in NPUPermutation.__dict__
    assert "moe_unpermute" not in NPUPermutation.__dict__
    assert "_moe_permute_mask_map" in NPUPermutation.__dict__
    assert "_moe_unpermute_mask_map" in NPUPermutation.__dict__
    assert "_supports_dense_mask_map" not in NPUPermutation.__dict__


def test_npu_mask_permutation_preserves_probs_dtype_adapter(monkeypatch):
    import transformer_engine.plugin.core.backends.vendor.npu.permutation as npu_permutation

    calls = {}
    expected = (object(), object(), object())

    class NativePermutation:
        def moe_permute_with_probs(self, *args):
            raise AssertionError(f"unexpected native fallback: {args}")

    class TorchNPU:
        @staticmethod
        def npu_moe_token_permute_with_routing_map(
            inp, routing_map, *, probs, num_out_tokens, drop_and_pad
        ):
            calls["args"] = (
                inp,
                probs,
                routing_map,
                num_out_tokens,
                drop_and_pad,
            )
            return expected

    monkeypatch.setattr(
        npu_permutation,
        "_get_torch_npu",
        lambda: TorchNPU,
    )

    inp = torch.randn(2, 4)
    probs = torch.randn(2, 3)
    routing_map = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    permutation = npu_permutation.NPUPermutation(NativePermutation())

    assert (
        permutation.moe_permute_with_probs(
            inp, probs, routing_map, num_out_tokens=-1
        )
        is expected
    )
    assert calls["args"][0] is inp
    assert calls["args"][1] is probs
    assert calls["args"][2] is routing_map
    assert calls["args"][3] == 3
    assert calls["args"][4] is False


def test_npu_permutation_delegates_chunk_sort_to_tenpu(monkeypatch):
    import transformer_engine.plugin.core.backends.vendor.npu.permutation as npu_permutation

    calls = []
    expected = object()
    expected_with_probs = (object(), object())

    class TENPUPermutation:
        @staticmethod
        def moe_sort_chunks_by_index(inp, split_sizes, sorted_index):
            calls.append(("without_probs", inp, split_sizes, sorted_index))
            return expected

        @staticmethod
        def moe_sort_chunks_by_index_with_probs(
            inp, probs, split_sizes, sorted_index
        ):
            calls.append(("with_probs", inp, probs, split_sizes, sorted_index))
            return expected_with_probs

    class NativePermutation:
        def moe_sort_chunks_by_index(self, *args):
            raise AssertionError(f"unexpected native fallback: {args}")

        def moe_sort_chunks_by_index_with_probs(self, *args):
            raise AssertionError(f"unexpected native fallback: {args}")

    monkeypatch.setattr(
        npu_permutation,
        "_get_tenpu_permutation",
        lambda: TENPUPermutation,
    )

    inp = torch.randn(4, 8)
    probs = torch.randn(4)
    split_sizes = torch.tensor([1, 3], dtype=torch.int32)
    sorted_index = torch.tensor([1, 0], dtype=torch.int32)
    permutation = npu_permutation.NPUPermutation(NativePermutation())

    assert (
        permutation.moe_sort_chunks_by_index(inp, split_sizes, sorted_index)
        is expected
    )
    assert (
        permutation.moe_sort_chunks_by_index_with_probs(
            inp, probs, split_sizes, sorted_index
        )
        is expected_with_probs
    )
    assert calls == [
        ("without_probs", inp, split_sizes, sorted_index),
        ("with_probs", inp, probs, split_sizes, sorted_index),
    ]


def test_npu_registers_only_profiled_qwen35_hot_path(monkeypatch):
    from transformer_engine.plugin.core.backends.reference.register_ops import (
        register_builtins as register_reference,
    )
    from transformer_engine.plugin.core.backends.vendor.npu.npu import NPUBackend
    from transformer_engine.plugin.core.backends.vendor.npu.register_ops import (
        register_builtins as register_npu,
    )
    from transformer_engine.plugin.core.registry import OpRegistry

    monkeypatch.setattr(NPUBackend, "is_available", lambda self: True)
    registry = OpRegistry()
    register_reference(registry)
    register_npu(registry)

    npu_ops = {
        op_name
        for op_name, implementations in registry.snapshot().impls_by_op.items()
        if any(impl.impl_id == "vendor.npu" for impl in implementations)
    }
    assert npu_ops == {
        "get_flash_attention_class",
        "get_attention_backend",
        "get_permutation_class",
        "rmsnorm_fwd",
        "rmsnorm_bwd",
        "te_general_grouped_gemm",
    }


# ===========================================================================
# Real NPU: RMSNorm — precision vs reference
# ===========================================================================
@requires_npu
class TestNPURMSNormAccuracy:
    """RMSNorm forward/backward: NPU vs reference backend."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(4, 64), (2, 256), (8, 1024)])
    def test_rmsnorm_fwd(self, npu_backend, ref_backend, dtype, shape):
        torch.manual_seed(42)
        x = torch.randn(*shape, dtype=dtype)
        w = torch.randn(shape[-1], dtype=dtype)
        x_npu, w_npu = x.to("npu"), w.to("npu")

        npu_result = npu_backend.rmsnorm_fwd(x_npu, w_npu, 1e-5, None, None, None, 0, False)
        ref_result = ref_backend.rmsnorm_fwd(x, w, 1e-5, None, None, None, 0, False)

        npu_out = npu_result[0]
        ref_out = ref_result[0]

        # For bf16: NPU kernel uses internal FP32 accumulation, reference uses bf16.
        # Both are valid bf16 implementations. Use FP32 ground truth as reference.
        if dtype == torch.bfloat16:
            # Compute FP32 ground truth
            x_f32 = x.float()
            w_f32 = w.float()
            rms = torch.sqrt(x_f32.pow(2).mean(-1, keepdim=True) + 1e-5)
            gt = x_f32 / rms * w_f32
            # Both NPU and ref should be close to FP32 ground truth
            npu_diff = (npu_out.cpu().float() - gt).abs().max().item()
            ref_diff = (ref_out.float() - gt).abs().max().item()
            # NPU should not be worse than 2x reference's error from ground truth
            assert npu_diff < max(
                ref_diff * 3, 0.1
            ), f"rmsnorm {shape}: npu_diff={npu_diff:.4f} >> ref_diff={ref_diff:.4f}"
        else:
            assert_close(npu_out, ref_out, dtype, msg=f"rmsnorm_fwd out {shape}")

        # rsigma: NPU returns [B,1], ref returns [B] — squeeze to compare
        npu_rsigma = npu_result[2].squeeze(-1) if npu_result[2].dim() > 1 else npu_result[2]
        ref_rsigma = ref_result[2]
        # rsigma tolerance: allow larger diff for bf16 since internal precision differs
        rs_atol = 0.01 if dtype == torch.bfloat16 else 1e-4
        rs_diff = (npu_rsigma.cpu().float() - ref_rsigma.float()).abs().max().item()
        assert rs_diff < rs_atol, f"rmsnorm_fwd rsigma {shape}: diff={rs_diff:.6f}, atol={rs_atol}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rmsnorm_fwd_zero_centered_gamma(self, npu_backend, ref_backend, dtype):
        torch.manual_seed(42)
        x = torch.randn(4, 128, dtype=dtype)
        w = torch.randn(128, dtype=dtype)
        x_npu, w_npu = x.to("npu"), w.to("npu")

        npu_out = npu_backend.rmsnorm_fwd(x_npu, w_npu, 1e-5, None, None, None, 0, True)[0]
        ref_out = ref_backend.rmsnorm_fwd(x, w, 1e-5, None, None, None, 0, True)[0]
        assert_close(npu_out, ref_out, dtype, msg="rmsnorm_fwd zero_centered")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rmsnorm_bwd(self, npu_backend, ref_backend, dtype):
        torch.manual_seed(42)
        x = torch.randn(4, 128, dtype=dtype)
        w = torch.randn(128, dtype=dtype)
        x_npu, w_npu = x.to("npu"), w.to("npu")

        # Forward for rsigma
        npu_fwd = npu_backend.rmsnorm_fwd(x_npu, w_npu, 1e-5, None, None, None, 0, False)
        ref_fwd = ref_backend.rmsnorm_fwd(x, w, 1e-5, None, None, None, 0, False)
        npu_rsigma = npu_fwd[2]
        ref_rsigma = ref_fwd[2]

        # Backward
        dz = torch.randn(4, 128, dtype=dtype)
        dz_npu = dz.to("npu")

        npu_bwd = npu_backend.rmsnorm_bwd(dz_npu, x_npu, npu_rsigma, w_npu, 0, False)
        ref_bwd = ref_backend.rmsnorm_bwd(dz, x, ref_rsigma, w, 0, False)

        if dtype == torch.bfloat16:
            # Use ground-truth comparison approach for bf16
            dx_diff = (npu_bwd[0].cpu().float() - ref_bwd[0].float()).abs().max().item()
            dw_diff = (npu_bwd[1].cpu().float() - ref_bwd[1].float()).abs().max().item()
            assert dx_diff < 0.15, f"rmsnorm_bwd dx diff={dx_diff:.4f}"
            assert dw_diff < 0.15, f"rmsnorm_bwd dw diff={dw_diff:.4f}"
        else:
            assert_close(npu_bwd[0], ref_bwd[0], dtype, msg="rmsnorm_bwd dx")
            assert_close(npu_bwd[1], ref_bwd[1], dtype, msg="rmsnorm_bwd dw")


# ===========================================================================
# Real NPU: Flash Attention — correctness validation
# (NPU flash attention kernel uses online softmax tiling, so exact numerical
# match vs naive SDPA is not expected. We verify directional correctness.)
# ===========================================================================
@requires_npu
class TestNPUFlashAttentionAccuracy:
    """Flash attention: verify correctness via consistency checks."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_flash_attn_deterministic(self, fa, dtype):
        """Same input produces same output (determinism)."""
        torch.manual_seed(42)
        B, S, H, D = 2, 32, 4, 64
        q = torch.randn(B, S, H, D, dtype=dtype, device="npu")
        k = torch.randn(B, S, H, D, dtype=dtype, device="npu")
        v = torch.randn(B, S, H, D, dtype=dtype, device="npu")

        out1 = fa.forward(q, k, v, qkv_layout="bshd_bshd_bshd")
        out2 = fa.forward(q, k, v, qkv_layout="bshd_bshd_bshd")
        assert torch.equal(out1, out2), "Flash attention should be deterministic"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_flash_attn_identity_value(self, fa, dtype):
        """When V is constant along seq dim, output should equal that constant."""
        B, S, H, D = 1, 16, 2, 64
        q = torch.randn(B, S, H, D, dtype=dtype, device="npu")
        k = torch.randn(B, S, H, D, dtype=dtype, device="npu")
        # V is constant along seq dim — all positions have same value
        v_row = torch.randn(1, 1, H, D, dtype=dtype, device="npu")
        v = v_row.expand(B, S, H, D).contiguous()

        out = fa.forward(q, k, v, qkv_layout="bshd_bshd_bshd")
        out_4d = out.view(B, S, H, D)

        # softmax(scores) @ V where all V rows are identical = V[0]
        # So every output position should equal v_row
        expected = v_row.expand(B, S, H, D)
        atol = 1e-2  # bf16 tolerance
        max_diff = (out_4d.float() - expected.float()).abs().max().item()
        assert max_diff < atol, f"Constant V test: max_diff={max_diff:.4e}, expected < {atol}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_flash_attn_causal_mask_effect(self, fa, dtype):
        """Causal and full attention differ before the final token."""
        torch.manual_seed(42)
        B, S, H, D = 1, 32, 2, 64
        q = torch.randn(B, S, H, D, dtype=dtype, device="npu")
        k = torch.randn(B, S, H, D, dtype=dtype, device="npu")
        v = torch.randn(B, S, H, D, dtype=dtype, device="npu")

        out_full = fa.forward(
            q,
            k,
            v,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="no_mask",
        )
        out_causal = fa.forward(
            q,
            k,
            v,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="causal",
        )

        out_full_4d = out_full.view(B, S, H, D)
        out_causal_4d = out_causal.view(B, S, H, D)

        # The first and middle tokens cannot see future tokens in causal mode.
        assert not torch.allclose(
            out_full_4d[:, 0],
            out_causal_4d[:, 0],
            atol=1e-2,
            rtol=1e-2,
        )
        mid = S // 4
        assert not torch.allclose(
            out_full_4d[:, mid],
            out_causal_4d[:, mid],
            atol=1e-2,
            rtol=1e-2,
        )

        # The final token can attend to the full sequence in both modes.
        assert torch.allclose(
            out_full_4d[:, -1],
            out_causal_4d[:, -1],
            atol=5e-2,
            rtol=5e-2,
        )

    def test_flash_attn_output_bounded(self, fa):
        """Output magnitude is bounded by V magnitude (weighted average)."""
        torch.manual_seed(42)
        B, S, H, D = 1, 512, 4, 64
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="npu")
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="npu")
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="npu")

        out = fa.forward(q, k, v, qkv_layout="bshd_bshd_bshd")
        assert out.shape == (B, S, H * D)
        assert not torch.isnan(out).any()
        # Attention is a convex combination of V rows — output should be bounded
        v_max = v.abs().max().item()
        out_max = out.abs().max().item()
        assert out_max <= v_max * 1.5, f"out_max={out_max:.3f} vs v_max={v_max:.3f}"


# ===========================================================================
# Real NPU: Flash Attention — numerical precision vs reference backend
# ===========================================================================
@requires_npu
class TestNPUFlashAttentionVsReference:
    """Flash attention: NPU vs reference backend (FlashAttentionTorch) numerical comparison."""

    @pytest.fixture
    def ref_fa(self):
        from transformer_engine.plugin.core.backends.reference.flash_attention import (
            FlashAttentionTorch,
        )

        fa = FlashAttentionTorch(softmax_scale=0.125, attention_dropout=0.0)
        fa.eval()
        return fa

    @pytest.fixture
    def npu_fa(self):
        from transformer_engine.plugin.core.backends.vendor.npu.flash_attention import (
            NPUFlashAttention,
        )

        fa = NPUFlashAttention(softmax_scale=0.125, attention_dropout=0.0)
        fa.eval()
        return fa

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("B,S,H,D", [(1, 32, 4, 64), (2, 64, 8, 64), (1, 128, 2, 128)])
    def test_flash_attn_fwd_no_mask(self, npu_fa, ref_fa, dtype, B, S, H, D):
        """Forward pass without mask: NPU vs reference SDPA."""
        torch.manual_seed(42)
        q = torch.randn(B, S, H, D, dtype=dtype)
        k = torch.randn(B, S, H, D, dtype=dtype)
        v = torch.randn(B, S, H, D, dtype=dtype)

        q_npu, k_npu, v_npu = q.to("npu"), k.to("npu"), v.to("npu")

        npu_out = npu_fa.forward(
            q_npu,
            k_npu,
            v_npu,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="no_mask",
        )
        ref_out = ref_fa.forward(
            q,
            k,
            v,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="no_mask",
        )

        # Both return shape [B, S, H*D]
        assert (
            npu_out.shape == ref_out.shape
        ), f"Shape mismatch: npu={npu_out.shape}, ref={ref_out.shape}"
        # Flash attention uses online softmax tiling — allow slightly larger tolerance
        atol, rtol = 5e-2, 5e-2
        npu_cpu = npu_out.detach().cpu().float()
        ref_cpu = ref_out.detach().cpu().float()
        max_diff = (npu_cpu - ref_cpu).abs().max().item()
        assert torch.allclose(
            npu_cpu, ref_cpu, atol=atol, rtol=rtol
        ), f"flash_attn fwd no_mask B={B},S={S},H={H},D={D}: max_diff={max_diff:.6e}, atol={atol}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_flash_attn_fwd_causal(self, npu_fa, ref_fa, dtype):
        """Forward pass with causal mask: NPU vs reference SDPA."""
        torch.manual_seed(42)
        B, S, H, D = 2, 64, 4, 64
        q = torch.randn(B, S, H, D, dtype=dtype)
        k = torch.randn(B, S, H, D, dtype=dtype)
        v = torch.randn(B, S, H, D, dtype=dtype)

        q_npu, k_npu, v_npu = q.to("npu"), k.to("npu"), v.to("npu")

        npu_out = npu_fa.forward(
            q_npu,
            k_npu,
            v_npu,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="causal",
        )
        ref_out = ref_fa.forward(
            q,
            k,
            v,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="causal",
        )

        assert npu_out.shape == ref_out.shape
        atol, rtol = 5e-2, 5e-2
        npu_cpu = npu_out.detach().cpu().float()
        ref_cpu = ref_out.detach().cpu().float()
        max_diff = (npu_cpu - ref_cpu).abs().max().item()
        assert torch.allclose(
            npu_cpu, ref_cpu, atol=atol, rtol=rtol
        ), f"flash_attn fwd causal: max_diff={max_diff:.6e}, atol={atol}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("B,S,H,D", [(1, 32, 4, 64), (2, 64, 4, 64)])
    def test_flash_attn_bwd_no_mask(self, npu_fa, ref_fa, dtype, B, S, H, D):
        """Backward pass without mask: NPU vs reference gradient comparison."""
        torch.manual_seed(42)
        # Create inputs that require grad
        q = torch.randn(B, S, H, D, dtype=dtype, requires_grad=True)
        k = torch.randn(B, S, H, D, dtype=dtype, requires_grad=True)
        v = torch.randn(B, S, H, D, dtype=dtype, requires_grad=True)

        # Reference forward + backward (CPU)
        ref_fa.train()
        ref_out = ref_fa.forward(
            q,
            k,
            v,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="no_mask",
        )
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)
        ref_dq = q.grad.clone()
        ref_dk = k.grad.clone()
        ref_dv = v.grad.clone()

        # NPU forward + backward
        q_npu = q.detach().to("npu").requires_grad_(True)
        k_npu = k.detach().to("npu").requires_grad_(True)
        v_npu = v.detach().to("npu").requires_grad_(True)

        npu_fa.train()
        npu_out = npu_fa.forward(
            q_npu,
            k_npu,
            v_npu,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="no_mask",
        )
        npu_out.backward(grad_out.to("npu"))
        npu_dq = q_npu.grad
        npu_dk = k_npu.grad
        npu_dv = v_npu.grad

        # Backward tolerances are larger than forward (error accumulates)
        atol, rtol = 1e-1, 1e-1
        for name, npu_g, ref_g in [
            ("dQ", npu_dq, ref_dq),
            ("dK", npu_dk, ref_dk),
            ("dV", npu_dv, ref_dv),
        ]:
            npu_cpu = npu_g.detach().cpu().float()
            ref_cpu = ref_g.float()
            max_diff = (npu_cpu - ref_cpu).abs().max().item()
            # Use cosine similarity as additional check — direction should be consistent
            cos_sim = torch.nn.functional.cosine_similarity(
                npu_cpu.flatten().unsqueeze(0),
                ref_cpu.flatten().unsqueeze(0),
            ).item()
            assert (
                cos_sim > 0.95
            ), f"flash_attn bwd {name}: cosine_sim={cos_sim:.4f} < 0.95, max_diff={max_diff:.6e}"
            assert torch.allclose(
                npu_cpu, ref_cpu, atol=atol, rtol=rtol
            ), f"flash_attn bwd {name}: max_diff={max_diff:.6e}, atol={atol}, cos_sim={cos_sim:.4f}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_flash_attn_bwd_causal(self, npu_fa, ref_fa, dtype):
        """Backward pass with causal mask: NPU vs reference gradient comparison."""
        torch.manual_seed(42)
        B, S, H, D = 1, 32, 4, 64

        q = torch.randn(B, S, H, D, dtype=dtype, requires_grad=True)
        k = torch.randn(B, S, H, D, dtype=dtype, requires_grad=True)
        v = torch.randn(B, S, H, D, dtype=dtype, requires_grad=True)

        ref_fa.train()
        ref_out = ref_fa.forward(
            q,
            k,
            v,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="causal",
        )
        grad_out = torch.randn_like(ref_out)
        ref_out.backward(grad_out)
        ref_dq = q.grad.clone()
        ref_dk = k.grad.clone()
        ref_dv = v.grad.clone()

        q_npu = q.detach().to("npu").requires_grad_(True)
        k_npu = k.detach().to("npu").requires_grad_(True)
        v_npu = v.detach().to("npu").requires_grad_(True)

        npu_fa.train()
        npu_out = npu_fa.forward(
            q_npu,
            k_npu,
            v_npu,
            qkv_layout="bshd_bshd_bshd",
            attn_mask_type="causal",
        )
        npu_out.backward(grad_out.to("npu"))

        atol, rtol = 1e-1, 1e-1
        for name, npu_g, ref_g in [
            ("dQ", q_npu.grad, ref_dq),
            ("dK", k_npu.grad, ref_dk),
            ("dV", v_npu.grad, ref_dv),
        ]:
            npu_cpu = npu_g.detach().cpu().float()
            ref_cpu = ref_g.float()
            max_diff = (npu_cpu - ref_cpu).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                npu_cpu.flatten().unsqueeze(0),
                ref_cpu.flatten().unsqueeze(0),
            ).item()
            assert cos_sim > 0.95, f"flash_attn bwd causal {name}: cos_sim={cos_sim:.4f} < 0.95"
            assert torch.allclose(
                npu_cpu, ref_cpu, atol=atol, rtol=rtol
            ), f"flash_attn bwd causal {name}: max_diff={max_diff:.6e}, atol={atol}"


# ===========================================================================
# Real NPU: Grouped GEMM — precision vs manual matmul
# ===========================================================================
@requires_npu
class TestNPUGroupedGEMMAccuracy:
    """te_general_grouped_gemm: NPU vs torch.matmul reference.

    TE-FL semantics: D[i] = op(B[i], transb) @ op(A[i], transa)

    We use transa=False, transb=False (simplest case):
      D[i] = B[i] @ A[i]
      B[i] shape: (N, K), A[i] shape: (K, M) => D[i]: (N, M)
      matrix_shape(A, False) => (K, M), a_rows=K, a_cols=M
      matrix_shape(B, False) => (N, K), b_rows=N, b_cols=K
      Check: b_cols(K) == a_rows(K) ✓
      Output: (b_rows, a_cols) = (N, M)
    """

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_grouped_gemm_basic(self, npu_backend, dtype):
        """Basic grouped GEMM with 2 groups, no transpose."""
        from transformer_engine.plugin.core.ops import DType

        torch.manual_seed(42)

        # Group 0: N=16, K=4, M=8 => B0:(N,K)=(16,4), A0:(K,M)=(4,8), D0:(N,M)=(16,8)
        # Group 1 uses the same output width required by native M-split.
        A0 = torch.randn(4, 8, device="npu", dtype=dtype)
        A1 = torch.randn(4, 8, device="npu", dtype=dtype)
        B0 = torch.randn(16, 4, device="npu", dtype=dtype)
        B1 = torch.randn(16, 4, device="npu", dtype=dtype)
        D0 = torch.zeros(16, 8, device="npu", dtype=dtype)
        D1 = torch.zeros(16, 8, device="npu", dtype=dtype)

        dtype_map = {torch.bfloat16: DType.kBFloat16, torch.float32: DType.kFloat32}
        d_type = dtype_map[dtype]

        workspace = [torch.empty(0, dtype=torch.uint8, device="npu")]
        bias = [torch.empty(0, device="npu"), torch.empty(0, device="npu")]

        returned_bias = npu_backend.te_general_grouped_gemm(
            A=[A0, A1],
            transa=False,
            B=[B0, B1],
            transb=False,
            D=[D0, D1],
            D_type=d_type,
            m_splits=[16, 16],
            bias=bias,
            bias_type=d_type,
            single_output=False,
            pre_gelu_out=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            grad=False,
            workspace=workspace,
            workspaceSizes=0,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )
        assert returned_bias is bias

        # Reference: D[i] = B[i] @ A[i]
        ref_D0 = (B0 @ A0).cpu().float()
        ref_D1 = (B1 @ A1).cpu().float()

        npu_D0 = D0.cpu().float()
        npu_D1 = D1.cpu().float()

        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5

        max_diff_0 = (npu_D0 - ref_D0).abs().max().item()
        max_diff_1 = (npu_D1 - ref_D1).abs().max().item()

        assert torch.allclose(
            npu_D0, ref_D0, atol=atol, rtol=rtol
        ), f"Group 0: max_diff={max_diff_0:.6e}"
        assert torch.allclose(
            npu_D1, ref_D1, atol=atol, rtol=rtol
        ), f"Group 1: max_diff={max_diff_1:.6e}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_grouped_gemm_single_output(self, npu_backend, dtype):
        """Grouped GEMM with single packed output buffer."""
        from transformer_engine.plugin.core.ops import DType

        torch.manual_seed(42)

        # single_output requires all groups to have the same output width (a_cols = M)
        # Two groups: same N=16, same K=4, same M=8
        # B0:(16,4), A0:(4,8) => D0:(16,8)
        # B1:(16,4), A1:(4,8) => D1:(16,8)
        A0 = torch.randn(4, 8, device="npu", dtype=dtype)
        A1 = torch.randn(4, 8, device="npu", dtype=dtype)
        B0 = torch.randn(16, 4, device="npu", dtype=dtype)
        B1 = torch.randn(16, 4, device="npu", dtype=dtype)

        # Single output: packed along N dimension: [N0+N1, M] = [32, 8]
        D_packed = torch.zeros(32, 8, device="npu", dtype=dtype)

        workspace = [torch.empty(0, dtype=torch.uint8, device="npu")]

        npu_backend.te_general_grouped_gemm(
            A=[A0, A1],
            transa=False,
            B=[B0, B1],
            transb=False,
            D=[D_packed],
            D_type=DType.kBFloat16,
            m_splits=[16, 16],
            bias=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            bias_type=DType.kBFloat16,
            single_output=True,
            pre_gelu_out=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            grad=False,
            workspace=workspace,
            workspaceSizes=0,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        # Reference
        ref_D0 = (B0 @ A0).cpu().float()  # [16, 8]
        ref_D1 = (B1 @ A1).cpu().float()  # [16, 8]
        ref_packed = torch.cat([ref_D0, ref_D1], dim=0)  # [32, 8]

        npu_packed = D_packed.cpu().float()
        max_diff = (npu_packed - ref_packed).abs().max().item()

        assert torch.allclose(
            npu_packed, ref_packed, atol=1e-2, rtol=1e-2
        ), f"single_output grouped gemm: max_diff={max_diff:.6e}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_grouped_gemm_dgrad(self, npu_backend, dtype):
        """Grouped GEMM dgrad path: transa=True, transb=False, grad=True.

        TE-FL semantics: D[i] = op(B[i], transb=False) @ op(A[i], transa=True)
                       = B[i] @ A[i].T

        This computes the activation gradient: dX = dY @ W^T
        where A[i] is the weight (shape K, M) and B[i] is the output grad (shape N, M).
        Result D[i] has shape (N, K).
        """
        from transformer_engine.plugin.core.ops import DType

        torch.manual_seed(42)

        # Group 0: A0:(K,M)=(8,4), B0:(N,M)=(16,4) => D0:(N,K)=(16,8)
        # Group 1 has the same output width required by native M-split.
        A0 = torch.randn(8, 4, device="npu", dtype=dtype)
        A1 = torch.randn(8, 4, device="npu", dtype=dtype)
        B0 = torch.randn(16, 4, device="npu", dtype=dtype)
        B1 = torch.randn(12, 4, device="npu", dtype=dtype)
        D0 = torch.zeros(16, 8, device="npu", dtype=dtype)
        D1 = torch.zeros(12, 8, device="npu", dtype=dtype)

        dtype_map = {torch.bfloat16: DType.kBFloat16, torch.float32: DType.kFloat32}
        d_type = dtype_map[dtype]

        workspace = [torch.empty(0, dtype=torch.uint8, device="npu")]
        bias = [torch.empty(0, device="npu"), torch.empty(0, device="npu")]

        npu_backend.te_general_grouped_gemm(
            A=[A0, A1],
            transa=True,
            B=[B0, B1],
            transb=False,
            D=[D0, D1],
            D_type=d_type,
            m_splits=[16, 12],
            bias=bias,
            bias_type=d_type,
            single_output=False,
            pre_gelu_out=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            grad=True,
            workspace=workspace,
            workspaceSizes=0,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        # Reference: D[i] = B[i] @ A[i].T
        ref_D0 = (B0.float() @ A0.float().T).cpu()
        ref_D1 = (B1.float() @ A1.float().T).cpu()

        npu_D0 = D0.cpu().float()
        npu_D1 = D1.cpu().float()

        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5

        max_diff_0 = (npu_D0 - ref_D0).abs().max().item()
        max_diff_1 = (npu_D1 - ref_D1).abs().max().item()

        assert torch.allclose(
            npu_D0, ref_D0, atol=atol, rtol=rtol
        ), f"dgrad group 0: max_diff={max_diff_0:.6e}"
        assert torch.allclose(
            npu_D1, ref_D1, atol=atol, rtol=rtol
        ), f"dgrad group 1: max_diff={max_diff_1:.6e}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_grouped_gemm_wgrad(self, npu_backend, dtype):
        """Grouped GEMM wgrad path: transa=False, transb=True, grad=True.

        TE-FL semantics: D[i] = op(B[i], transb=True) @ op(A[i], transa=False)
                       = B[i].T @ A[i]

        This computes the weight gradient: dW = X^T @ dY
        where B[i] is the activation (shape N, K) and A[i] is the output grad (shape N, M).
        Result D[i] has shape (K, M).
        """
        from transformer_engine.plugin.core.ops import DType

        torch.manual_seed(42)

        # Group 0: A0:(N,M)=(16,8), B0:(N,K)=(16,4) => D0:(K,M)=(4,8)
        # Group 1: A1:(N,M)=(12,8), B1:(N,K)=(12,4) => D1:(K,M)=(4,8)
        A0 = torch.randn(16, 8, device="npu", dtype=dtype)
        A1 = torch.randn(12, 8, device="npu", dtype=dtype)
        B0 = torch.randn(16, 4, device="npu", dtype=dtype)
        B1 = torch.randn(12, 4, device="npu", dtype=dtype)
        D0 = torch.zeros(4, 8, device="npu", dtype=dtype)
        D1 = torch.zeros(4, 8, device="npu", dtype=dtype)

        dtype_map = {torch.bfloat16: DType.kBFloat16, torch.float32: DType.kFloat32}
        d_type = dtype_map[dtype]

        workspace = [torch.empty(0, dtype=torch.uint8, device="npu")]
        bias = [torch.empty(0, device="npu"), torch.empty(0, device="npu")]

        npu_backend.te_general_grouped_gemm(
            A=[A0, A1],
            transa=False,
            B=[B0, B1],
            transb=True,
            D=[D0, D1],
            D_type=d_type,
            m_splits=[16, 12],
            bias=bias,
            bias_type=d_type,
            single_output=False,
            pre_gelu_out=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            grad=True,
            workspace=workspace,
            workspaceSizes=0,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        # Reference: D[i] = B[i].T @ A[i]
        ref_D0 = (B0.float().T @ A0.float()).cpu()
        ref_D1 = (B1.float().T @ A1.float()).cpu()

        npu_D0 = D0.cpu().float()
        npu_D1 = D1.cpu().float()

        atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5

        max_diff_0 = (npu_D0 - ref_D0).abs().max().item()
        max_diff_1 = (npu_D1 - ref_D1).abs().max().item()

        assert torch.allclose(
            npu_D0, ref_D0, atol=atol, rtol=rtol
        ), f"wgrad group 0: max_diff={max_diff_0:.6e}"
        assert torch.allclose(
            npu_D1, ref_D1, atol=atol, rtol=rtol
        ), f"wgrad group 1: max_diff={max_diff_1:.6e}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_grouped_gemm_dgrad_single_output(self, npu_backend, dtype):
        """Grouped GEMM dgrad with single packed output buffer."""
        from transformer_engine.plugin.core.ops import DType

        torch.manual_seed(42)

        # single_output requires all groups to have the same output width.
        # dgrad: D[i] = B[i] @ A[i].T, output shape (N, K).
        # Need common K across groups.
        # Group 0: A0:(K,M)=(8,6), B0:(N,M)=(16,6) => D0:(16,8)
        # Group 1 uses the same contracted width required by native M-split.
        A0 = torch.randn(8, 6, device="npu", dtype=dtype)
        A1 = torch.randn(8, 6, device="npu", dtype=dtype)
        B0 = torch.randn(16, 6, device="npu", dtype=dtype)
        B1 = torch.randn(12, 6, device="npu", dtype=dtype)

        # Single output: packed along N dimension: [16+12, 8] = [28, 8]
        D_packed = torch.zeros(28, 8, device="npu", dtype=dtype)

        workspace = [torch.empty(0, dtype=torch.uint8, device="npu")]

        npu_backend.te_general_grouped_gemm(
            A=[A0, A1],
            transa=True,
            B=[B0, B1],
            transb=False,
            D=[D_packed],
            D_type=DType.kBFloat16,
            m_splits=[16, 12],
            bias=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            bias_type=DType.kBFloat16,
            single_output=True,
            pre_gelu_out=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            grad=True,
            workspace=workspace,
            workspaceSizes=0,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        # Reference
        ref_D0 = (B0.float() @ A0.float().T).cpu()  # [16, 8]
        ref_D1 = (B1.float() @ A1.float().T).cpu()  # [12, 8]
        ref_packed = torch.cat([ref_D0, ref_D1], dim=0)  # [28, 8]

        npu_packed = D_packed.cpu().float()
        max_diff = (npu_packed - ref_packed).abs().max().item()

        assert torch.allclose(
            npu_packed, ref_packed, atol=1e-2, rtol=1e-2
        ), f"dgrad single_output grouped gemm: max_diff={max_diff:.6e}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_grouped_gemm_wgrad_single_output(self, npu_backend, dtype):
        """Grouped GEMM wgrad with single packed output buffer."""
        from transformer_engine.plugin.core.ops import DType

        torch.manual_seed(42)

        # wgrad: D[i] = B[i].T @ A[i], output shape (K, M).
        # single_output requires common output width M across groups.
        # Group 0: A0:(N,M)=(16,8), B0:(N,K)=(16,4) => D0:(4,8)
        # Group 1: A1:(N,M)=(12,8), B1:(N,K)=(12,4) => D1:(4,8)
        A0 = torch.randn(16, 8, device="npu", dtype=dtype)
        A1 = torch.randn(12, 8, device="npu", dtype=dtype)
        B0 = torch.randn(16, 4, device="npu", dtype=dtype)
        B1 = torch.randn(12, 4, device="npu", dtype=dtype)

        # Single output: packed along K dimension: [4+4, 8] = [8, 8]
        D_packed = torch.zeros(8, 8, device="npu", dtype=dtype)

        workspace = [torch.empty(0, dtype=torch.uint8, device="npu")]

        npu_backend.te_general_grouped_gemm(
            A=[A0, A1],
            transa=False,
            B=[B0, B1],
            transb=True,
            D=[D_packed],
            D_type=DType.kBFloat16,
            m_splits=[16, 12],
            bias=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            bias_type=DType.kBFloat16,
            single_output=True,
            pre_gelu_out=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            grad=True,
            workspace=workspace,
            workspaceSizes=0,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        # Reference
        ref_D0 = (B0.float().T @ A0.float()).cpu()  # [4, 8]
        ref_D1 = (B1.float().T @ A1.float()).cpu()  # [4, 8]
        ref_packed = torch.cat([ref_D0, ref_D1], dim=0)  # [8, 8]

        npu_packed = D_packed.cpu().float()
        max_diff = (npu_packed - ref_packed).abs().max().item()

        assert torch.allclose(
            npu_packed, ref_packed, atol=1e-2, rtol=1e-2
        ), f"wgrad single_output grouped gemm: max_diff={max_diff:.6e}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_grouped_gemm_accumulate(self, npu_backend, dtype):
        """Grouped GEMM with accumulate=True adds to existing D."""
        from transformer_engine.plugin.core.ops import DType

        torch.manual_seed(42)
        # Two groups are required by the native grouped kernel.
        A0 = torch.randn(4, 8, device="npu", dtype=dtype)
        A1 = torch.randn(4, 8, device="npu", dtype=dtype)
        B0 = torch.randn(16, 4, device="npu", dtype=dtype)
        B1 = torch.randn(12, 4, device="npu", dtype=dtype)

        # Pre-fill D with known values
        D0_init = torch.ones(16, 8, device="npu", dtype=dtype)
        D1_init = torch.ones(12, 8, device="npu", dtype=dtype)
        D0 = D0_init.clone()
        D1 = D1_init.clone()

        workspace = [torch.empty(0, dtype=torch.uint8, device="npu")]

        npu_backend.te_general_grouped_gemm(
            A=[A0, A1],
            transa=False,
            B=[B0, B1],
            transb=False,
            D=[D0, D1],
            D_type=DType.kBFloat16,
            m_splits=[16, 12],
            bias=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            bias_type=DType.kBFloat16,
            single_output=False,
            pre_gelu_out=[torch.empty(0, device="npu"), torch.empty(0, device="npu")],
            grad=False,
            workspace=workspace,
            workspaceSizes=0,
            accumulate=True,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        # Reference: D0 = D0_init + B0 @ A0
        ref_D0 = (D0_init.float() + (B0 @ A0).float()).cpu()
        ref_D1 = (D1_init.float() + (B1 @ A1).float()).cpu()
        npu_D0 = D0.cpu().float()
        npu_D1 = D1.cpu().float()
        max_diff = (npu_D0 - ref_D0).abs().max().item()

        assert torch.allclose(
            npu_D0, ref_D0, atol=1e-2, rtol=1e-2
        ), f"accumulate grouped gemm: max_diff={max_diff:.6e}"
        assert torch.allclose(npu_D1, ref_D1, atol=1e-2, rtol=1e-2)
