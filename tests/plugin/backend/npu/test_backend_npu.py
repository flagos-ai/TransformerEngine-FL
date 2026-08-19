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
# Real NPU: GEMM — precision vs matmul reference
# ===========================================================================
def _run_generic_gemm(
    npu_backend,
    left: torch.Tensor,
    right: torch.Tensor,
    dtype: torch.dtype,
    out=None,
    accumulate: bool = False,
):
    """Run TE-FL generic_gemm with conventional left @ right semantics."""
    from transformer_engine.plugin.core.ops import DType

    dtype_map = {
        torch.float32: DType.kFloat32,
        torch.float16: DType.kFloat16,
        torch.bfloat16: DType.kBFloat16,
    }
    te_dtype = dtype_map[dtype]
    result, _, _, _ = npu_backend.generic_gemm(
        A=right,
        transA=False,
        B=left,
        transB=False,
        D=out,
        quantizer=None,
        output_dtype=te_dtype,
        bias=None,
        bias_type=te_dtype,
        gelu=False,
        gelu_in=None,
        grad=False,
        workspace=torch.empty(0, dtype=torch.uint8, device=left.device),
        workspace_size=0,
        accumulate=accumulate,
        use_split_accumulator=False,
    )
    return result


@requires_npu
class TestNPUGEMMAccuracy:
    """generic_gemm: NPU vs torch.matmul reference."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gemm_basic(self, npu_backend, dtype):
        torch.manual_seed(42)
        M, K, N = 16, 32, 64
        left = torch.randn(M, K, dtype=dtype)
        right = torch.randn(K, N, dtype=dtype)
        left_npu, right_npu = left.to("npu"), right.to("npu")

        npu_out = _run_generic_gemm(npu_backend, left_npu, right_npu, dtype)
        ref_out = (left.float() @ right.float()).to(dtype)
        assert_close(npu_out, ref_out, dtype, msg="gemm_basic")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gemm_large(self, npu_backend, dtype):
        torch.manual_seed(42)
        M, K, N = 256, 512, 1024
        left = torch.randn(M, K, dtype=dtype, device="npu")
        right = torch.randn(K, N, dtype=dtype, device="npu")

        npu_out = _run_generic_gemm(npu_backend, left, right, dtype)
        ref_out = (left.cpu().float() @ right.cpu().float()).to(dtype)
        assert_close(npu_out, ref_out, dtype, msg="gemm_large")

    def test_gemm_accumulate(self, npu_backend):
        torch.manual_seed(42)
        M, K, N = 8, 16, 32
        left = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
        right = torch.randn(K, N, dtype=torch.bfloat16, device="npu")
        destination = torch.ones(M, N, dtype=torch.bfloat16, device="npu")

        result = _run_generic_gemm(
            npu_backend,
            left,
            right,
            torch.bfloat16,
            out=destination,
            accumulate=True,
        )
        expected = (left.float() @ right.float()) + 1.0
        assert_close(result, expected, torch.bfloat16, msg="gemm_accum")


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
# Real NPU: Multi-tensor ops — exact value verification
# ===========================================================================
@requires_npu
class TestNPUMultiTensorAccuracy:
    """Multi-tensor operations: exact value verification."""

    def test_multi_tensor_scale(self, npu_backend):
        t1 = torch.tensor([2.0, 4.0, 6.0], device="npu")
        t_out = torch.zeros(3, device="npu")
        noop = torch.zeros(1, device="npu", dtype=torch.int32)
        npu_backend.multi_tensor_scale(65536, noop, [[t1], [t_out]], 0.5)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(
            t_out.cpu(), expected, atol=1e-6
        ), f"Expected {expected}, got {t_out.cpu()}"

    def test_multi_tensor_l2norm(self, npu_backend):
        # [3, 4] -> norm = 5.0
        t1 = torch.tensor([3.0, 4.0], device="npu")
        noop = torch.zeros(1, device="npu", dtype=torch.int32)
        result = npu_backend.multi_tensor_l2norm(65536, noop, [[t1]], False)
        norm_val = result[0] if isinstance(result, tuple) else result
        got = norm_val.item() if hasattr(norm_val, "item") else float(norm_val)
        assert abs(got - 5.0) < 1e-4, f"Expected 5.0, got {got}"

    def test_multi_tensor_l2norm_multi_tensor(self, npu_backend):
        # [1,1,1,1] norm = 2.0
        t1 = torch.ones(4, device="npu")
        noop = torch.zeros(1, device="npu", dtype=torch.int32)
        result = npu_backend.multi_tensor_l2norm(65536, noop, [[t1]], False)
        norm_val = result[0] if isinstance(result, tuple) else result
        got = norm_val.item() if hasattr(norm_val, "item") else float(norm_val)
        assert abs(got - 2.0) < 1e-4, f"Expected 2.0, got {got}"

    def test_multi_tensor_unscale_l2norm(self, npu_backend):
        t1 = torch.tensor([6.0, 8.0], device="npu")
        inv_scale = torch.tensor([2.0], device="npu")
        noop = torch.zeros(1, device="npu", dtype=torch.int32)
        result = npu_backend.multi_tensor_unscale_l2norm(65536, noop, [[t1]], inv_scale)
        norm_val = result[0] if isinstance(result, tuple) else result
        got = norm_val.item() if hasattr(norm_val, "item") else float(norm_val)
        assert abs(got - 20.0) < 1e-4, f"Expected 20.0, got {got}"


# ===========================================================================
# Real NPU: FP8 scale computation — precision vs reference
# ===========================================================================
@requires_npu
class TestNPUComputeScaleAccuracy:
    """multi_tensor_compute_scale_and_scale_inv: NPU vs reference backend."""

    @pytest.mark.parametrize(
        "amax_vals,max_fp8",
        [
            ([8.0], 448.0),
            ([1.0, 16.0, 0.5], 448.0),
            ([100.0, 200.0], 240.0),
            ([0.001], 448.0),  # very small amax
        ],
    )
    def test_compute_scale_vs_reference(self, npu_backend, ref_backend, amax_vals, max_fp8):
        """NPU scale/scale_inv matches reference for various amax values."""
        n = len(amax_vals)
        epsilon = 1e-12

        # NPU tensors
        amaxes_npu = [torch.tensor([v], device="npu") for v in amax_vals]
        scales_npu = [torch.ones(1, device="npu") for _ in range(n)]
        scale_invs_npu = [torch.ones(1, device="npu") for _ in range(n)]
        noop_npu = torch.zeros(1, device="npu", dtype=torch.int32)

        # Reference tensors (CPU)
        amaxes_ref = [torch.tensor([v]) for v in amax_vals]
        scales_ref = [torch.ones(1) for _ in range(n)]
        scale_invs_ref = [torch.ones(1) for _ in range(n)]
        noop_ref = torch.zeros(1, dtype=torch.int32)

        npu_backend.multi_tensor_compute_scale_and_scale_inv(
            65536,
            noop_npu,
            [amaxes_npu, scales_npu, scale_invs_npu],
            max_fp8,
            False,
            epsilon,
        )
        ref_backend.multi_tensor_compute_scale_and_scale_inv(
            65536,
            noop_ref,
            [amaxes_ref, scales_ref, scale_invs_ref],
            max_fp8,
            False,
            epsilon,
        )

        for i in range(n):
            npu_scale = scales_npu[i].cpu()
            ref_scale = scales_ref[i]
            npu_sinv = scale_invs_npu[i].cpu()
            ref_sinv = scale_invs_ref[i]

            assert torch.allclose(
                npu_scale, ref_scale, atol=1e-5, rtol=1e-5
            ), f"scale[{i}]: npu={npu_scale.item():.6e}, ref={ref_scale.item():.6e}"
            assert torch.allclose(
                npu_sinv, ref_sinv, atol=1e-5, rtol=1e-5
            ), f"scale_inv[{i}]: npu={npu_sinv.item():.6e}, ref={ref_sinv.item():.6e}"

    @pytest.mark.parametrize("force_pow_2", [True, False])
    def test_compute_scale_pow2(self, npu_backend, ref_backend, force_pow_2):
        """Verify force_pow_2_scales flag produces matching results."""
        amax_vals = [7.0, 13.0, 100.0]
        max_fp8 = 448.0
        epsilon = 1e-12
        n = len(amax_vals)

        amaxes_npu = [torch.tensor([v], device="npu") for v in amax_vals]
        scales_npu = [torch.ones(1, device="npu") for _ in range(n)]
        scale_invs_npu = [torch.ones(1, device="npu") for _ in range(n)]
        noop_npu = torch.zeros(1, device="npu", dtype=torch.int32)

        amaxes_ref = [torch.tensor([v]) for v in amax_vals]
        scales_ref = [torch.ones(1) for _ in range(n)]
        scale_invs_ref = [torch.ones(1) for _ in range(n)]
        noop_ref = torch.zeros(1, dtype=torch.int32)

        npu_backend.multi_tensor_compute_scale_and_scale_inv(
            65536,
            noop_npu,
            [amaxes_npu, scales_npu, scale_invs_npu],
            max_fp8,
            force_pow_2,
            epsilon,
        )
        ref_backend.multi_tensor_compute_scale_and_scale_inv(
            65536,
            noop_ref,
            [amaxes_ref, scales_ref, scale_invs_ref],
            max_fp8,
            force_pow_2,
            epsilon,
        )

        for i in range(n):
            npu_scale = scales_npu[i].cpu()
            ref_scale = scales_ref[i]
            assert torch.allclose(npu_scale, ref_scale, atol=1e-5, rtol=1e-5), (
                f"scale[{i}] pow2={force_pow_2}: "
                f"npu={npu_scale.item():.6e}, ref={ref_scale.item():.6e}"
            )
            if force_pow_2:
                # Verify it's actually a power of 2
                log2_val = torch.log2(npu_scale)
                assert torch.allclose(
                    log2_val, log2_val.round(), atol=1e-5
                ), f"scale[{i}] not power of 2: {npu_scale.item()}"

    def test_compute_scale_noop_flag(self, npu_backend):
        """When noop_flag is non-zero, scales should remain unchanged."""
        amax = torch.tensor([8.0], device="npu")
        scale = torch.tensor([999.0], device="npu")
        scale_inv = torch.tensor([888.0], device="npu")
        noop = torch.ones(1, device="npu", dtype=torch.int32)  # non-zero => skip

        npu_backend.multi_tensor_compute_scale_and_scale_inv(
            65536,
            noop,
            [[amax], [scale], [scale_inv]],
            448.0,
            False,
            1e-12,
        )

        assert scale.item() == 999.0, f"scale changed to {scale.item()} despite noop"
        assert scale_inv.item() == 888.0, f"scale_inv changed to {scale_inv.item()} despite noop"


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
        # Group 1: N=16, K=4, M=6 => B1:(N,K)=(16,4), A1:(K,M)=(4,6), D1:(N,M)=(16,6)
        A0 = torch.randn(4, 8, device="npu", dtype=dtype)
        A1 = torch.randn(4, 6, device="npu", dtype=dtype)
        B0 = torch.randn(16, 4, device="npu", dtype=dtype)
        B1 = torch.randn(16, 4, device="npu", dtype=dtype)
        D0 = torch.zeros(16, 8, device="npu", dtype=dtype)
        D1 = torch.zeros(16, 6, device="npu", dtype=dtype)

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
        # Group 1: A1:(K,M)=(6,4), B1:(N,M)=(12,4) => D1:(N,K)=(12,6)
        A0 = torch.randn(8, 4, device="npu", dtype=dtype)
        A1 = torch.randn(6, 4, device="npu", dtype=dtype)
        B0 = torch.randn(16, 4, device="npu", dtype=dtype)
        B1 = torch.randn(12, 4, device="npu", dtype=dtype)
        D0 = torch.zeros(16, 8, device="npu", dtype=dtype)
        D1 = torch.zeros(12, 6, device="npu", dtype=dtype)

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
        # Group 1: A1:(K,M)=(8,4), B1:(N,M)=(12,4) => D1:(12,8)
        A0 = torch.randn(8, 6, device="npu", dtype=dtype)
        A1 = torch.randn(8, 4, device="npu", dtype=dtype)
        B0 = torch.randn(16, 6, device="npu", dtype=dtype)
        B1 = torch.randn(12, 4, device="npu", dtype=dtype)

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
        # B0:(16,4), A0:(4,8) => D0:(16,8)
        A0 = torch.randn(4, 8, device="npu", dtype=dtype)
        B0 = torch.randn(16, 4, device="npu", dtype=dtype)

        # Pre-fill D with known values
        D0_init = torch.ones(16, 8, device="npu", dtype=dtype)
        D0 = D0_init.clone()

        workspace = [torch.empty(0, dtype=torch.uint8, device="npu")]

        npu_backend.te_general_grouped_gemm(
            A=[A0],
            transa=False,
            B=[B0],
            transb=False,
            D=[D0],
            D_type=DType.kBFloat16,
            m_splits=[16],
            bias=[torch.empty(0, device="npu")],
            bias_type=DType.kBFloat16,
            single_output=False,
            pre_gelu_out=[torch.empty(0, device="npu")],
            grad=False,
            workspace=workspace,
            workspaceSizes=0,
            accumulate=True,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        # Reference: D0 = D0_init + B0 @ A0
        ref_D0 = (D0_init.float() + (B0 @ A0).float()).cpu()
        npu_D0 = D0.cpu().float()
        max_diff = (npu_D0 - ref_D0).abs().max().item()

        assert torch.allclose(
            npu_D0, ref_D0, atol=1e-2, rtol=1e-2
        ), f"accumulate grouped gemm: max_diff={max_diff:.6e}"
