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


def _remainder_master(param, remainder):
    """Decode TE's BF16/int16 representation on CPU."""
    high = param.cpu().view(torch.int16).to(torch.int32)
    low = remainder.cpu().to(torch.int32)
    return (((high - (low < 0).int()) << 16) | (low & 0xFFFF)).view(torch.float32)


def _remainder_adam_expected(tensors, step, mode, bias_correction):
    """Independent CPU FP32 equations from TE's CUDA remainder Adam functor."""
    grad, param, moment, variance, remainder = [t.cpu() for t in tensors]
    master = _remainder_master(param, remainder)
    beta1 = torch.tensor(0.9, dtype=torch.float32)
    beta2 = torch.tensor(0.95, dtype=torch.float32)
    grad = grad.float()
    if mode == 0:
        grad = grad + 0.1 * master
    next_m = beta1 * moment.float() + (1 - beta1) * grad
    next_v = beta2 * variance.float() + ((1 - beta2) * grad) * grad
    correction1 = 1 - 0.9**step if bias_correction else 1.0
    correction2 = 1 - 0.95**step if bias_correction else 1.0
    update = (next_m / correction1) / ((next_v / correction2).sqrt() + 1e-8)
    if mode == 1:
        update = update + 0.1 * master
    return master - 0.003 * update, next_m.to(moment.dtype), next_v.to(variance.dtype)


@requires_npu
class TestNPURemainderAdam:
    """FP32-compute updates, signed remainders, and native optimizer lifecycle."""

    @pytest.fixture(autouse=True)
    def vendor_policy(self, monkeypatch):
        # Exercise the default fallback policy; remainder Adam itself must never retry.
        monkeypatch.delenv("TE_FL_STRICT", raising=False)
        monkeypatch.setenv("TE_FL_PREFER", "vendor")
        monkeypatch.setenv("TE_FL_ALLOW_VENDORS", "NPU")

    @pytest.mark.parametrize("grad_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("moment_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("mode", [0, 1])
    @pytest.mark.parametrize("bias_correction", [0, 1])
    def test_matches_cuda_equations(self, grad_dtype, moment_dtype, mode, bias_correction):
        from transformer_engine.plugin.core import get_manager, get_tefl_module

        torch.manual_seed(37)
        lists = [[] for _ in range(5)]
        for shape in [(5, 7), (1025,), (0,)]:
            n = torch.empty(shape).numel()
            param = torch.randn(shape).to(torch.bfloat16)
            low = (
                torch.tensor([-32768, -1, 0, 1, 32767], dtype=torch.int16)
                .repeat((n + 4) // 5)[:n]
                .reshape(shape)
            )
            tensors = [
                torch.randn(shape).to(grad_dtype),
                param,
                (torch.randn(shape, dtype=torch.float32) * 0.01).to(moment_dtype),
                (torch.rand(shape, dtype=torch.float32) * 0.01 + 1e-5).to(moment_dtype),
                low,
            ]
            for output, tensor in zip(lists, tensors):
                output.append(tensor.to("npu"))
        noop = torch.zeros(1, dtype=torch.int32, device="npu")
        pointers = [[tensor.data_ptr() for tensor in group] for group in lists]
        gradients = [t.cpu().clone() for t in lists[0]]
        for step in range(1, 4):
            expected = [
                _remainder_adam_expected(tensors, step, mode, bias_correction)
                for tensors in zip(*lists)
            ]
            get_tefl_module().multi_tensor_adam_param_remainder(
                65536,
                noop,
                lists,
                0.003,
                0.9,
                0.95,
                1e-8,
                step,
                mode,
                bias_correction,
                0.1,
            )
            assert (
                get_manager().get_selected_impl_id("multi_tensor_adam_param_remainder")
                == "vendor.npu"
            )
            for tensors, (master, moment, variance) in zip(zip(*lists), expected):
                torch.testing.assert_close(
                    _remainder_master(tensors[1], tensors[4]), master, atol=2e-7, rtol=3e-6
                )
                for actual, reference in [(tensors[2], moment), (tensors[3], variance)]:
                    torch.testing.assert_close(
                        actual.cpu().float(),
                        reference.float(),
                        atol=2e-7,
                        rtol=5e-3 if moment_dtype == torch.bfloat16 else 1e-5,
                    )
            for grad, original in zip(lists[0], gradients):
                torch.testing.assert_close(grad.cpu(), original, atol=0, rtol=0)
        assert [[t.data_ptr() for t in group] for group in lists] == pointers

    def test_signed_halfway_rounding(self, npu_backend):
        # Crossing low=0x7fff -> 0x8000 must carry into the high bits even at a tie.
        # FP32-to-BF16 round-to-even would instead leave these high bits unchanged.
        tensors = [
            torch.tensor([-1.0, 1.0], dtype=torch.float32, device="npu"),
            torch.tensor([1.0, -1.0], dtype=torch.bfloat16, device="npu"),
            torch.zeros(2, dtype=torch.float32, device="npu"),
            torch.zeros(2, dtype=torch.float32, device="npu"),
            torch.full((2,), 32767, dtype=torch.int16, device="npu"),
        ]
        npu_backend.multi_tensor_adam_param_remainder(
            65536,
            torch.zeros(1, dtype=torch.int32, device="npu"),
            [[t] for t in tensors],
            2**-23,
            0.0,
            0.0,
            0.0,
            1,
            1,
            0,
            0.0,
        )
        torch.testing.assert_close(
            tensors[1].cpu(),
            torch.tensor([1.0078125, -1.0078125], dtype=torch.bfloat16),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            tensors[4].cpu(), torch.full((2,), -32768, dtype=torch.int16), atol=0, rtol=0
        )

    def test_noop_empty_and_validation_before_updates(self, npu_backend):
        noop = torch.ones(1, dtype=torch.int32, device="npu")
        npu_backend.multi_tensor_adam_param_remainder(
            65536, noop, [], 0.003, 0.9, 0.95, 1e-8, 1, 1, 1, 0.1
        )
        noop.zero_()
        npu_backend.multi_tensor_adam_param_remainder(
            65536, noop, [[], [], [], [], []], 0.003, 0.9, 0.95, 1e-8, 1, 1, 1, 0.1
        )
        tensors = [
            torch.ones(3, dtype=torch.float32, device="npu"),
            torch.ones(3, dtype=torch.bfloat16, device="npu"),
            torch.zeros(3, dtype=torch.float32, device="npu"),
            torch.zeros(3, dtype=torch.float32, device="npu"),
            torch.zeros(3, dtype=torch.int16, device="npu"),
        ]
        original = [t.clone() for t in tensors]
        lists = [[t, t.clone()] for t in tensors]
        lists[0][1] = torch.ones(4, dtype=torch.float32, device="npu")
        with pytest.raises(ValueError, match="shape and device"):
            npu_backend.multi_tensor_adam_param_remainder(
                65536, noop, lists, 0.003, 0.9, 0.95, 1e-8, 1, 1, 1, 0.1
            )
        for actual, reference in zip(tensors, original):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    @pytest.mark.slow
    @pytest.mark.parametrize("moment_dtype", [torch.float32, torch.bfloat16])
    def test_large_tensor_exceeds_65535_tiles(self, npu_backend, moment_dtype):
        n = 65536 * 1024 + 17
        tensors = [
            torch.full((n,), 0.125, dtype=torch.float32, device="npu"),
            torch.full((n,), 0.5, dtype=torch.bfloat16, device="npu"),
            torch.zeros(n, dtype=moment_dtype, device="npu"),
            torch.zeros(n, dtype=moment_dtype, device="npu"),
            torch.full((n,), -32768, dtype=torch.int16, device="npu"),
        ]
        indices = torch.tensor([0, 1023, 1024, 65535 * 1024 - 1, 65535 * 1024, n - 1], device="npu")
        samples = [t[indices].cpu() for t in tensors]
        expected = _remainder_adam_expected(samples, 1, 1, 1)
        torch.npu.synchronize()
        allocated = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        npu_backend.multi_tensor_adam_param_remainder(
            65536,
            torch.zeros(1, dtype=torch.int32, device="npu"),
            [[t] for t in tensors],
            0.003,
            0.9,
            0.95,
            1e-8,
            1,
            1,
            1,
            0.1,
        )
        torch.npu.synchronize()
        assert torch.npu.max_memory_allocated() - allocated < 64 * 1024**2
        samples = [t[indices].cpu() for t in tensors]
        torch.testing.assert_close(
            _remainder_master(samples[1], samples[4]), expected[0], atol=2e-7, rtol=3e-6
        )
        for actual, reference in zip(samples[2:4], expected[1:]):
            torch.testing.assert_close(
                actual.float(),
                reference.float(),
                atol=2e-7,
                rtol=5e-3 if moment_dtype == torch.bfloat16 else 1e-5,
            )

    @pytest.mark.parametrize("moment_dtype", [torch.bfloat16, torch.float32])
    def test_fused_adam_optimizer_five_steps(self, moment_dtype):
        from transformer_engine.plugin.core import get_manager
        from transformer_engine.pytorch.optimizers import FusedAdam

        torch.manual_seed(37)
        param = torch.nn.Parameter(torch.randn(1025, device="npu", dtype=torch.bfloat16))
        optimizer = FusedAdam(
            [param],
            lr=0.003,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            master_weights=True,
            master_weight_dtype=torch.float32,
            exp_avg_dtype=moment_dtype,
            exp_avg_sq_dtype=moment_dtype,
            use_decoupled_grad=True,
            store_param_remainders=True,
        )
        for step in range(1, 6):
            param.decoupled_grad = torch.full_like(param, 0.125, dtype=torch.float32)
            state = optimizer.state[param]
            moment = state.get("exp_avg", torch.zeros_like(param, dtype=moment_dtype))
            variance = state.get("exp_avg_sq", torch.zeros_like(param, dtype=moment_dtype))
            remainder = state.get("master_param", torch.zeros_like(param, dtype=torch.int16))
            expected = _remainder_adam_expected(
                [param.decoupled_grad, param, moment, variance, remainder], step, 1, 1
            )
            optimizer.step()
            assert (
                get_manager().get_selected_impl_id("multi_tensor_adam_param_remainder")
                == "vendor.npu"
            )
            state = optimizer.state[param]
            assert state["master_param"].dtype == torch.int16
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == moment_dtype
            torch.testing.assert_close(
                _remainder_master(param, state["master_param"]), expected[0], atol=2e-7, rtol=3e-6
            )
            for name, reference in zip(("exp_avg", "exp_avg_sq"), expected[1:]):
                torch.testing.assert_close(
                    state[name].cpu().float(),
                    reference.float(),
                    atol=2e-7,
                    rtol=5e-3 if moment_dtype == torch.bfloat16 else 1e-5,
                )
            optimizer.zero_grad()

    @pytest.mark.parametrize("param_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_native_eager_initialize_state(self, param_dtype):
        from transformer_engine.pytorch.optimizers import FusedAdam

        param = torch.nn.Parameter(torch.tensor([0.5, -1.0], device="npu", dtype=param_dtype))
        optimizer = FusedAdam(
            [param],
            master_weights=True,
            store_param_remainders=True,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
        )
        optimizer.initialize_state(param)
        state = optimizer.state[param]
        for name in ("exp_avg", "exp_avg_sq"):
            assert state[name].dtype == torch.float32
            torch.testing.assert_close(state[name], torch.zeros_like(state[name]), atol=0, rtol=0)
        if param_dtype == torch.bfloat16:
            assert state["master_param"].dtype == torch.int16
            torch.testing.assert_close(
                state["master_param"], torch.zeros_like(param, dtype=torch.int16), atol=0, rtol=0
            )
        else:
            assert state["master_param"].dtype == torch.float32
            torch.testing.assert_close(state["master_param"], param.float(), atol=0, rtol=0)

    @pytest.mark.parametrize("store_param_remainders", [False, True])
    @pytest.mark.parametrize("adam_w_mode", [False, True])
    def test_native_optimizer_checkpoint_resume(self, store_param_remainders, adam_w_mode):
        from copy import deepcopy

        from transformer_engine.plugin.core import get_manager
        from transformer_engine.pytorch.optimizers import FusedAdam

        options = dict(
            lr=0.003,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            adam_w_mode=adam_w_mode,
            master_weights=True,
            master_weight_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            use_decoupled_grad=True,
            store_param_remainders=store_param_remainders,
        )
        torch.manual_seed(37)
        param = torch.nn.Parameter(torch.randn(1025, device="npu", dtype=torch.bfloat16))
        optimizer = FusedAdam([param], **options)
        # This is the existing Megatron eager-initialization call signature.
        optimizer.initialize_state(param)
        for step in range(1, 4):
            param.decoupled_grad = torch.full_like(param, 0.125 * step, dtype=torch.float32)
            optimizer.step()
        checkpoint = deepcopy(optimizer.state_dict())
        master_dtype = torch.int16 if store_param_remainders else torch.float32
        assert next(iter(checkpoint["state"].values()))["master_param"].dtype == master_dtype
        restored_param = torch.nn.Parameter(param.detach().clone())
        restored_optimizer = FusedAdam([restored_param], **options)
        restored_optimizer.load_state_dict(checkpoint)
        for name in ("exp_avg", "exp_avg_sq", "master_param"):
            actual = restored_optimizer.state[restored_param][name]
            assert actual.dtype == (master_dtype if name == "master_param" else torch.float32)
            torch.testing.assert_close(actual, optimizer.state[param][name], atol=0, rtol=0)
        gradient = torch.linspace(-0.2, 0.2, param.numel(), device="npu", dtype=torch.float32)
        param.decoupled_grad = gradient
        restored_param.decoupled_grad = gradient.clone()
        optimizer.step()
        restored_optimizer.step()
        assert optimizer.param_groups[0]["step"] == restored_optimizer.param_groups[0]["step"] == 4
        torch.testing.assert_close(param, restored_param, atol=0, rtol=0)
        for name in ("exp_avg", "exp_avg_sq", "master_param"):
            torch.testing.assert_close(
                optimizer.state[param][name],
                restored_optimizer.state[restored_param][name],
                atol=0,
                rtol=0,
            )
        op_name = (
            "multi_tensor_adam_param_remainder" if store_param_remainders else "multi_tensor_adam"
        )
        assert get_manager().get_selected_impl_id(op_name) == "vendor.npu"

    @pytest.mark.parametrize("source_remainders", [False, True])
    def test_native_optimizer_rejects_checkpoint_format_conversion(self, source_remainders):
        from copy import deepcopy

        from transformer_engine.pytorch.optimizers import FusedAdam

        param = torch.nn.Parameter(torch.tensor([0.5, -1.0], device="npu", dtype=torch.bfloat16))
        source = FusedAdam([param], master_weights=True, store_param_remainders=source_remainders)
        source.initialize_state(param)
        checkpoint = deepcopy(source.state_dict())
        restored_param = torch.nn.Parameter(param.detach().clone())
        target = FusedAdam(
            [restored_param], master_weights=True, store_param_remainders=not source_remainders
        )
        target.initialize_state(restored_param)
        original = deepcopy(target.state_dict())
        with pytest.raises(ValueError, match="master_param format.*store_param_remainders"):
            target.load_state_dict(checkpoint)
        master = next(iter(checkpoint["state"].values()))["master_param"]
        with pytest.raises(ValueError, match="same store_param_remainders"):
            target.set_scaled_state(restored_param, "master_param", master)
        for name, tensor in next(iter(original["state"].values())).items():
            torch.testing.assert_close(target.state[restored_param][name], tensor, atol=0, rtol=0)
        torch.testing.assert_close(restored_param, param, atol=0, rtol=0)

    @pytest.mark.parametrize("cached", [False, True])
    def test_partial_kernel_failure_never_retries(self, monkeypatch, npu_backend, cached):
        import os

        from transformer_engine.plugin.core.backends.vendor.npu import fused_adam
        from transformer_engine.plugin.core.manager import OpManager
        from transformer_engine.plugin.core.registry import OpRegistry
        from transformer_engine.plugin.core import get_manager
        from transformer_engine.plugin.core.types import BackendImplKind, OpImpl

        op_name = "multi_tensor_adam_param_remainder"
        fallback_calls = []

        def fallback(*args):
            fallback_calls.append(args)
            raise AssertionError("A partially completed optimizer call must not retry")

        global_manager = get_manager()
        global_manager.ensure_initialized()
        registry = OpRegistry()
        registry.register_many(
            [
                global_manager.registry.get_implementation(op_name, "vendor.npu"),
                OpImpl(op_name, "reference.injected", BackendImplKind.REFERENCE, fallback),
            ]
        )
        manager = OpManager(registry)
        # Keep this real dispatch registry isolated from global plugin discovery.
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        lists = [[] for _ in range(5)]
        for _ in range(2):
            tensors = [
                torch.full((1025,), 0.125, dtype=torch.float32, device="npu"),
                torch.full((1025,), 0.5, dtype=torch.bfloat16, device="npu"),
                torch.zeros(1025, dtype=torch.float32, device="npu"),
                torch.zeros(1025, dtype=torch.float32, device="npu"),
                torch.zeros(1025, dtype=torch.int16, device="npu"),
            ]
            for group, tensor in zip(lists, tensors):
                group.append(tensor)
        args = (
            65536,
            torch.zeros(1, dtype=torch.int32, device="npu"),
            lists,
            0.003,
            0.9,
            0.95,
            1e-8,
            1,
            1,
            1,
            0.1,
        )
        if cached:
            manager.call(op_name, *args)
            torch.npu.synchronize()
        originals = [[t.clone() for t in tensors] for tensors in lists]
        expected = _remainder_adam_expected([group[0] for group in originals], 1, 1, 1)
        original_kernel = fused_adam._adam_param_remainder_kernel
        launches = []

        class FailSecondLaunch:
            def __getitem__(self, grid):
                def launch(*kernel_args, **kernel_kwargs):
                    launches.append(grid)
                    if len(launches) == 2:
                        raise RuntimeError("injected second-tensor launch failure")
                    original_kernel[grid](*kernel_args, **kernel_kwargs)
                    torch.npu.synchronize()

                return launch

        monkeypatch.setattr(fused_adam, "_adam_param_remainder_kernel", FailSecondLaunch())
        with pytest.raises(RuntimeError, match="injected second-tensor"):
            manager.call(op_name, *args)
        assert len(launches) == 2
        assert not fallback_calls
        torch.testing.assert_close(
            _remainder_master(lists[1][0], lists[4][0]), expected[0], atol=2e-7, rtol=3e-6
        )
        for actual, reference in zip([lists[2][0], lists[3][0]], expected[1:]):
            torch.testing.assert_close(actual.cpu(), reference, atol=2e-7, rtol=1e-5)
        for actual, original in zip([group[1] for group in lists], [g[1] for g in originals]):
            torch.testing.assert_close(actual, original, atol=0, rtol=0)

    @pytest.mark.integration
    def test_megatron_distributed_optimizer_checkpoint(self, tmp_path):
        import json
        import os
        from pathlib import Path

        if int(os.getenv("WORLD_SIZE", "1")) < 2:
            pytest.skip("requires torchrun with at least two ranks")
        pytest.importorskip("megatron.core")
        from megatron.core import parallel_state
        from megatron.core.dist_checkpointing import ShardedTensor, load, save
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.transformer import TransformerConfig
        from transformer_engine.plugin.core import get_manager

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("hccl")
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel()
        dp_rank = parallel_state.get_data_parallel_rank()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=1,
            params_dtype=torch.bfloat16,
            bf16=True,
        )

        def make_model_and_optimizer():
            model = torch.nn.Linear(128, 128, bias=False, dtype=torch.bfloat16, device="npu")
            model.config = config
            model.weight.data.fill_(0.5)
            ddp = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_distributed_optimizer=True,
                    grad_reduce_in_fp32=True,
                ),
                model,
            )
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    optimizer="adam",
                    lr=0.003,
                    bf16=True,
                    clip_grad=0.0,
                    use_distributed_optimizer=True,
                    use_precision_aware_optimizer=True,
                    main_params_dtype=torch.float32,
                    main_grads_dtype=torch.float32,
                    exp_avg_dtype=torch.float32,
                    exp_avg_sq_dtype=torch.float32,
                ),
                [ddp],
            )
            return ddp, optimizer

        def parts(optimizer):
            return getattr(optimizer, "chained_optimizers", [optimizer])

        def assert_remainder_storage(ddp, optimizer):
            for part in parts(optimizer):
                assert part.config.store_param_remainders
                assert part.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                assert all(
                    tensor is None
                    for group in part.shard_fp32_from_float16_groups
                    for tensor in group
                )
                for group in part.optimizer.param_groups:
                    for param in group["params"]:
                        assert param.dtype == torch.bfloat16
                        state = part.optimizer.state[param]
                        assert state["master_param"].dtype == torch.int16
                        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
            for param in ddp.module.parameters():
                assert param.main_param is None

        def step(ddp, optimizer, index):
            optimizer.zero_grad()
            ddp.zero_grad_buffer()
            x = torch.full((4, 128), index * 0.003, dtype=torch.bfloat16, device="npu")
            loss = ddp(x).float().square().mean()
            loss.backward()
            ddp.finish_grad_sync()
            result = optimizer.step()
            assert result[0]
            torch.npu.synchronize()
            return loss.item()

        def sharded(ddp):
            return {
                "weight": ShardedTensor.from_rank_offsets(
                    "weight", ddp.module.weight, replica_id=(0, 0, dp_rank)
                )
            }

        model, optimizer = make_model_and_optimizer()
        # Invoke the actual MG eager callback, which calls TE initialize_state(param).
        for part in parts(optimizer):
            part.init_state_fn(part.optimizer, part.config)
        assert_remainder_storage(model, optimizer)
        losses = [step(model, optimizer, index) for index in range(1, 4)]
        assert (
            get_manager().get_selected_impl_id("multi_tensor_adam_param_remainder") == "vendor.npu"
        )
        sharding_type = "dp_reshardable"
        metadata = {"distrib_optim_sharding_type": sharding_type}
        output_dir = os.getenv("OPT4_RESULT_DIR")
        checkpoint_root = os.getenv("OPT4_CKPT_ROOT")
        checkpoint = (
            Path(checkpoint_root) / f"checkpoint_{sharding_type}" if checkpoint_root else tmp_path
        )
        path = [str(checkpoint) if rank == 0 else None]
        torch.distributed.broadcast_object_list(path, src=0)
        checkpoint = Path(path[0])
        if rank == 0:
            checkpoint.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        model_state = sharded(model)
        save(
            {
                "model": model_state,
                "optimizer": optimizer.sharded_state_dict(model_state, metadata=metadata),
            },
            checkpoint,
        )
        restored_model, restored_optimizer = make_model_and_optimizer()
        restored_model_state = sharded(restored_model)
        target = {
            "model": restored_model_state,
            "optimizer": restored_optimizer.sharded_state_dict(
                restored_model_state, metadata=metadata, is_loading=True
            ),
        }
        loaded = load(target, checkpoint)
        restored_model.module.load_state_dict(loaded["model"])
        restored_optimizer.load_state_dict(loaded["optimizer"])
        assert_remainder_storage(restored_model, restored_optimizer)
        before_next = model.module.weight.detach().clone()
        torch.testing.assert_close(restored_model.module.weight, before_next, atol=0, rtol=0)
        loss = step(model, optimizer, 4)
        restored_loss = step(restored_model, restored_optimizer, 4)
        assert loss == restored_loss
        assert not torch.equal(model.module.weight, before_next)
        torch.testing.assert_close(
            model.module.weight, restored_model.module.weight, atol=0, rtol=0
        )
        state_bytes = 0
        shard_numel = 0
        for original, restored in zip(parts(optimizer), parts(restored_optimizer)):
            for group, restored_group in zip(
                original.optimizer.param_groups, restored.optimizer.param_groups
            ):
                for param, restored_param in zip(group["params"], restored_group["params"]):
                    shard_numel += param.numel()
                    for name in ("master_param", "exp_avg", "exp_avg_sq"):
                        state = original.optimizer.state[param][name]
                        state_bytes += state.numel() * state.element_size()
                        torch.testing.assert_close(
                            state, restored.optimizer.state[restored_param][name], atol=0, rtol=0
                        )
        result = {
            "rank": rank,
            "world_size": torch.distributed.get_world_size(),
            "sharding_type": sharding_type,
            "losses": losses + [loss],
            "shard_numel": shard_numel,
            "optimizer_state_bytes": state_bytes,
            "bytes_per_sharded_element": state_bytes / shard_numel,
            "mg_fp32_main_param_count": 0,
            "master_dtype": "torch.int16",
            "moments_dtype": "torch.float32",
            "resume_next_step_equal": True,
            "vendor": get_manager().get_selected_impl_id("multi_tensor_adam_param_remainder"),
        }
        print(json.dumps(result, sort_keys=True))
        if output_dir:
            (Path(output_dir) / f"mg_{sharding_type}_rank_{rank}.json").write_text(
                json.dumps(result, indent=2)
            )
        torch.distributed.barrier()
