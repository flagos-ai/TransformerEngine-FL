# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Numerical layout regressions for the real FlagGems attention adapter.

Run in a fresh process with a compatible FlagGems installation and accelerator:
    python -m pytest -q tests/pytorch/attention/test_flagos_attention.py
"""

from importlib import import_module

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an accelerator")


@pytest.fixture
def backend():
    pytest.importorskip("flag_gems")
    return import_module(
        "transformer_engine.plugin.core.backends.flagos.attention.dot_product_attention.backends"
    )


def _inputs(fmt, kv_heads=2, head_dim=64, cross_attention=False):
    torch.manual_seed(197)
    q_lengths = [17, 31] if fmt == "thd" else [32, 32]
    kv_lengths = [23, 37] if fmt == "thd" else [48, 48]
    if not cross_attention:
        kv_lengths = q_lengths

    def make(lengths, heads):
        if fmt == "thd":
            shape = (sum(lengths), heads, head_dim * 2)
        elif fmt == "sbhd":
            shape = (lengths[0], len(lengths), heads, head_dim * 2)
        else:
            shape = (len(lengths), lengths[0], heads, head_dim * 2)
        # Exercise non-contiguous Q/K/V views as well as the output layout.
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16)[..., ::2].requires_grad_()

    q, k, v = make(q_lengths, 4), make(kv_lengths, kv_heads), make(kv_lengths, kv_heads)

    def bounds(lengths):
        return torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)

    args = dict(
        is_training=True,
        max_seqlen_q=max(q_lengths),
        max_seqlen_kv=max(kv_lengths),
        cu_seqlens_q=bounds(q_lengths),
        cu_seqlens_kv=bounds(kv_lengths),
        page_table_k=None,
        page_table_v=None,
        q=q,
        k=k,
        v=v,
        attn_scale=head_dim**-0.5,
        dropout_p=0.0,
        qkv_layout="_".join([fmt] * 3),
        attn_mask_type="no_mask",
        window_size=(-1, -1),
        rng_gen=None,
        deterministic=False,
        layer_number=1,
    )
    return args, q_lengths, kv_lengths


def _reference(q, k, v, fmt, q_lengths, kv_lengths, causal, scale):
    """FP32 matmul/softmax reference, independent of both fused backends.

    Packed inputs use one block-diagonal attention matrix, rather than copying
    the adapter's per-sequence dispatch algorithm.
    """

    def to_bhsd(x):
        if fmt == "thd":
            return x.transpose(0, 1).unsqueeze(0)
        return x.permute(1, 2, 0, 3) if fmt == "sbhd" else x.permute(0, 2, 1, 3)

    q, k, v = (to_bhsd(x) for x in (q, k, v))
    groups = q.shape[1] // k.shape[1]
    k, v = (x.repeat_interleave(groups, dim=1) for x in (k, v))
    scores = (q @ k.transpose(-1, -2)) * scale
    allowed = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device)
    if fmt == "thd":
        q_ids = torch.arange(len(q_lengths), device=q.device).repeat_interleave(
            torch.tensor(q_lengths, device=q.device)
        )
        k_ids = torch.arange(len(kv_lengths), device=q.device).repeat_interleave(
            torch.tensor(kv_lengths, device=q.device)
        )
        allowed &= q_ids[:, None] == k_ids[None, :]
    if causal:
        allowed &= torch.ones_like(allowed).tril()
    result = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1) @ v
    if fmt == "thd":
        return result.squeeze(0).transpose(0, 1)
    return result.permute(2, 0, 1, 3) if fmt == "sbhd" else result.permute(0, 2, 1, 3)


def _check_attention(backend, fmt, mask_type, kv_heads=2, head_dim=64, cross_attention=False):
    args, q_lengths, kv_lengths = _inputs(fmt, kv_heads, head_dim, cross_attention)
    causal = mask_type in ("causal", "padding_causal")
    args.update(attn_mask_type=mask_type, window_size=(-1, 0) if causal else (-1, -1))
    inputs = [args[name] for name in ("q", "k", "v")]
    refs = [x.detach().float().requires_grad_() for x in inputs]
    actual = backend.AttnFuncFL.apply(*args.values())
    expected = _reference(*refs, fmt, q_lengths, kv_lengths, causal, args["attn_scale"])
    assert actual.shape == inputs[0].shape
    assert actual.dtype == inputs[0].dtype
    assert actual.is_contiguous()  # FlashAttentionFL subsequently uses view().
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.float())
    for result, reference in [(actual, expected)] + [
        (x.grad, ref.grad) for x, ref in zip(inputs, refs)
    ]:
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result.float(), reference, atol=0.08, rtol=0.05)


@pytest.mark.parametrize("fmt", ["sbhd", "bshd", "thd"])
@pytest.mark.parametrize("mask_type", ["no_mask", "causal"])
@pytest.mark.parametrize("kv_heads", [4, 2])
def test_attention_forward_backward(backend, fmt, mask_type, kv_heads):
    _check_attention(backend, fmt, mask_type, kv_heads)


def test_large_head_mqa(backend):
    _check_attention(backend, "sbhd", "causal", kv_heads=1, head_dim=256)


@pytest.mark.parametrize("mask_type", ["padding", "padding_causal"])
def test_packed_padding_masks(backend, mask_type):
    _check_attention(backend, "thd", mask_type)


@pytest.mark.parametrize("fmt", ["sbhd", "bshd", "thd"])
def test_noncausal_cross_attention(backend, fmt):
    _check_attention(backend, fmt, "no_mask", cross_attention=True)


@pytest.mark.parametrize(
    "case, error, message",
    [
        ("dropout", NotImplementedError, "zero dropout"),
        ("paged", NotImplementedError, "non-paged"),
        ("window", NotImplementedError, "sliding windows"),
        ("dense_padding", NotImplementedError, "packed THD"),
        ("non_square_causal", NotImplementedError, "Non-square causal"),
        ("mask", NotImplementedError, "mask"),
        ("missing_bounds", ValueError, "integer tensors"),
        ("float_bounds", ValueError, "integer tensors"),
        ("bad_end", ValueError, "span unpadded"),
        ("bad_start", ValueError, "span unpadded"),
        ("batch_mismatch", ValueError, "span unpadded"),
        ("empty_sequence", NotImplementedError, "Empty packed"),
        ("wrong_rank", ValueError, "dimensions"),
        ("kv_mismatch", ValueError, "matching shapes"),
        ("mixed_layout", NotImplementedError, "layout"),
    ],
)
def test_reject_invalid_inputs(backend, monkeypatch, case, error, message):
    fmt = "sbhd" if case == "dense_padding" else "thd"
    args, _, _ = _inputs(fmt, cross_attention=case == "non_square_causal")
    if case == "dropout":
        args["dropout_p"] = 0.1
    elif case == "paged":
        args["page_table_k"] = torch.zeros(1, device="cuda", dtype=torch.int32)
    elif case == "window":
        args["window_size"] = (8, 0)
    elif case == "dense_padding":
        args["attn_mask_type"] = "padding"
    elif case == "non_square_causal":
        args.update(attn_mask_type="causal", window_size=(-1, 0))
    elif case == "mask":
        args["attn_mask_type"] = "arbitrary"
    elif case == "missing_bounds":
        args["cu_seqlens_q"] = None
    elif case == "float_bounds":
        args["cu_seqlens_q"] = args["cu_seqlens_q"].float()
    elif case == "bad_end":
        args["cu_seqlens_q"][-1] -= 1
    elif case == "bad_start":
        args["cu_seqlens_q"][0] = 1
    elif case == "batch_mismatch":
        args["cu_seqlens_q"] = args["cu_seqlens_q"][:2]
    elif case == "empty_sequence":
        args["cu_seqlens_q"][1] = 0
    elif case == "wrong_rank":
        args["q"] = args["q"].unsqueeze(0)
    elif case == "kv_mismatch":
        args["v"] = args["v"][:-1]
    elif case == "mixed_layout":
        args["qkv_layout"] = "thd_bshd_bshd"

    def unexpected_kernel(*_args, **_kwargs):
        pytest.fail("Invalid inputs must be rejected before launching an attention kernel")

    monkeypatch.setattr(
        backend.flag_gems, "scaled_dot_product_attention_forward", unexpected_kernel
    )
    with pytest.raises(error, match=message):
        backend.AttnFuncFL.apply(*args.values())


@pytest.mark.parametrize("fmt", ["sbhd", "bshd", "thd"])
def test_wrapper_output_layout(backend, fmt):
    args, _, _ = _inputs(fmt)
    attention = backend.FlashAttentionFL(softmax_scale=args["attn_scale"])
    result = attention._forward_impl(
        args["q"],
        args["k"],
        args["v"],
        qkv_layout=args["qkv_layout"],
        cu_seqlens_q=args["cu_seqlens_q"],
        cu_seqlens_kv=args["cu_seqlens_kv"],
        max_seqlen_q=args["max_seqlen_q"],
        max_seqlen_kv=args["max_seqlen_kv"],
        attn_mask_type="no_mask",
    )
    expected = backend.AttnFuncFL.apply(*args.values()).flatten(-2)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("option", ["pad_between_seqs", "alibi_slopes", "fp8"])
def test_wrapper_rejects_unsupported_options(backend, option):
    args, _, _ = _inputs("thd")
    attention = backend.FlashAttentionFL(softmax_scale=args["attn_scale"])
    with pytest.raises(NotImplementedError):
        attention._forward_impl(args["q"], args["k"], args["v"], **{option: True})
