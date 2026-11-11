# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Bit-exact native comparisons for every dispatched Triton API, plus the
torch.compile tracing guard that needs the same isolation.

Not named ``test_*`` on purpose: these cases need a clean CUDA context.
``test_triton_dispatch_numerics.py`` runs this file in a fresh process, which
keeps the comparisons meaningful when the full suite runs in one process --
sibling suites that mock ``flag_gems`` abort a CUDA graph capture, and an
aborted capture poisons the context for every later allocation.
"""

import importlib
import inspect

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _entry_and_native(module_name, api_name):
    entry = getattr(importlib.import_module(module_name), api_name)
    return entry, inspect.unwrap(entry)


def _assert_equal(actual, expected):
    if isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_equal(actual_item, expected_item)
    elif actual is None:
        assert expected is None
    else:
        assert torch.equal(actual, expected)


def test_pad_matches_native_exactly():
    entry, native = _entry_and_native(
        "transformer_engine.pytorch.triton.pad", "pad_columnwise_scale_inv"
    )
    inp = torch.randn(129, 5, device="cuda")

    _assert_equal(entry(inp.clone()), native(inp.clone()))


def test_cross_entropy_forward_and_backward_match_native_exactly():
    module_name = "transformer_engine.pytorch.triton.cross_entropy"
    forward, native_forward = _entry_and_native(module_name, "cross_entropy_forward")
    backward, native_backward = _entry_and_native(module_name, "cross_entropy_backward")
    logits = torch.randn(2, 3, 17, device="cuda")
    target = torch.tensor([[0, 4, 16], [3, -100, 8]], device="cuda")
    args = (target, 0.1, False, None, -100, False)

    actual_forward = forward(logits.clone(), *args)
    expected_forward = native_forward(logits.clone(), *args)
    _assert_equal(actual_forward[:5], expected_forward[:5])
    assert actual_forward[5:] == expected_forward[5:]

    grad = torch.randn(2, 3, device="cuda")

    def run_backward(fn, forward_result):
        _, saved_input, stats, saved_target, n_non_ignore, rank, world_size = forward_result
        return fn(
            saved_input.clone(),
            stats,
            saved_target,
            n_non_ignore,
            grad.clone(),
            0.1,
            False,
            rank,
            world_size,
            -100,
        )

    _assert_equal(
        run_backward(backward, actual_forward),
        run_backward(native_backward, expected_forward),
    )


def test_grouped_dbias_functions_match_native_exactly():
    module_name = "transformer_engine.pytorch.triton.grouped_dbias_dscales"
    dy = torch.randn(7, 19, device="cuda", dtype=torch.float16)
    offsets = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int64)

    entry, native = _entry_and_native(module_name, "compute_grouped_dbias")
    initial_dbias = torch.randn(3, 19, device="cuda", dtype=torch.float32)
    _assert_equal(
        entry(dy, offsets, 3, initial_dbias.clone()),
        native(dy, offsets, 3, initial_dbias.clone()),
    )

    entry, native = _entry_and_native(module_name, "compute_grouped_dbias_dscales")
    scales = torch.randn(7, device="cuda", dtype=torch.float32)
    bias = torch.randn(3, 19, device="cuda", dtype=torch.float16)
    initial_dscales = torch.randn(7, device="cuda", dtype=torch.float32)
    _assert_equal(
        entry(
            dy,
            scales,
            bias,
            offsets,
            initial_dbias.clone(),
            initial_dscales.clone(),
        ),
        native(
            dy,
            scales,
            bias,
            offsets,
            initial_dbias.clone(),
            initial_dscales.clone(),
        ),
    )


def test_permutation_apis_match_native_exactly():
    module_name = "transformer_engine.pytorch.triton.permutation"
    routing_map = torch.tensor(
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
        device="cuda",
        dtype=torch.bool,
    )
    num_tokens, num_experts = routing_map.shape
    hidden_size = 13
    num_out_tokens = int(routing_map.sum())

    make_map, native_make_map = _entry_and_native(module_name, "make_row_id_map")
    row_id_map = make_map(routing_map, num_tokens, num_experts)
    native_row_id_map = native_make_map(routing_map, num_tokens, num_experts)
    for token in range(num_tokens):
        routed = int(row_id_map[token, -1])
        assert routed == int(native_row_id_map[token, -1])
        assert torch.equal(row_id_map[token, :routed], native_row_id_map[token, :routed])
        assert torch.equal(
            row_id_map[token, num_experts : num_experts + routed],
            native_row_id_map[token, num_experts : num_experts + routed],
        )

    inp = torch.randn(num_tokens, hidden_size, device="cuda")
    probs = torch.randn(num_tokens, num_experts, device="cuda")
    permute, native_permute = _entry_and_native(module_name, "permute_with_mask_map")
    permute_args = (
        row_id_map,
        probs,
        None,
        None,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
        0,
    )
    actual_permuted = permute(inp, *permute_args)
    expected_permuted = native_permute(inp, *permute_args)
    _assert_equal(actual_permuted, expected_permuted)

    unpermute, native_unpermute = _entry_and_native(module_name, "unpermute_with_mask_map")
    merging_probs = torch.randn(num_tokens, num_experts, device="cuda")
    unpermute_args = (
        row_id_map,
        merging_probs,
        actual_permuted[2],
        None,
        num_tokens,
        num_experts,
        hidden_size,
    )
    _assert_equal(
        unpermute(actual_permuted[0], *unpermute_args),
        native_unpermute(expected_permuted[0], *unpermute_args),
    )

    backward, native_backward = _entry_and_native(
        module_name, "unpermute_with_mask_map_bwd_with_merging_probs"
    )
    output_grad = torch.randn(num_tokens, hidden_size, device="cuda")
    backward_args = (
        row_id_map,
        actual_permuted[0],
        merging_probs,
        None,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
    )
    _assert_equal(
        backward(output_grad, *backward_args),
        native_backward(output_grad, *backward_args),
    )

    make_sort_map, native_make_sort_map = _entry_and_native(module_name, "make_chunk_sort_map")
    split_sizes = torch.tensor([2, 1, 3], device="cuda", dtype=torch.int32)
    sorted_indices = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    sort_map = make_sort_map(split_sizes, sorted_indices, 6, 3)
    _assert_equal(sort_map, native_make_sort_map(split_sizes, sorted_indices, 6, 3))

    sort_chunks, native_sort_chunks = _entry_and_native(module_name, "sort_chunks_by_map")
    chunks = torch.randn(6, hidden_size, device="cuda")
    chunk_probs = torch.randn(6, device="cuda")
    _assert_equal(
        sort_chunks(chunks, sort_map, chunk_probs, 6, hidden_size, True),
        native_sort_chunks(chunks, sort_map, chunk_probs, 6, hidden_size, True),
    )


def test_real_permutation_forward_backward_matches_pytorch():
    from transformer_engine.pytorch.permutation import (
        moe_permute_with_probs,
        moe_unpermute,
    )

    num_tokens, num_experts, hidden_size = 4, 3, 13
    routing_map = torch.tensor(
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
        device="cuda",
        dtype=torch.bool,
    )
    inp = torch.randn(num_tokens, hidden_size, device="cuda", requires_grad=True)
    probs = torch.randn(num_tokens, num_experts, device="cuda", requires_grad=True)
    num_out_tokens = int(routing_map.sum())

    permuted, _, row_id_map = moe_permute_with_probs(inp, probs, routing_map, num_out_tokens)
    output = moe_unpermute(
        permuted,
        row_id_map,
        merging_probs=probs,
        restore_shape=inp.shape,
        map_type="mask",
    )
    output.sum().backward()

    selected = routing_map.T.contiguous().nonzero()[:, 1]
    expected_permuted = inp.detach().index_select(0, selected)
    expected_probs = probs.detach().T.contiguous().masked_select(routing_map.T.contiguous())
    expected_output = torch.zeros_like(inp).scatter_add(
        0,
        selected[:, None].expand(-1, hidden_size),
        expected_permuted * expected_probs[:, None],
    )
    expected_inp_grad = routing_map.to(inp.dtype) * probs.detach()
    expected_inp_grad = expected_inp_grad.sum(dim=1, keepdim=True).expand_as(inp)
    expected_probs_grad = routing_map.to(probs.dtype) * (inp.detach().sum(dim=1, keepdim=True))

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(inp.grad, expected_inp_grad)
    torch.testing.assert_close(probs.grad, expected_probs_grad)


def test_mhc_apis_match_native_exactly():
    module_name = "transformer_engine.pytorch.triton.mhc"
    n = 4
    s, b, hidden = 2, 2, 8

    projection, native_projection = _entry_and_native(module_name, "mhc_fused_projection")
    x_flat = torch.randn(s * b, hidden * n, device="cuda")
    phi = torch.randn(2 * n + n * n, hidden * n, device="cuda")
    actual_h, actual_ms = projection(x_flat, phi, False)
    expected_h, expected_ms = native_projection(x_flat, phi, False)
    _assert_equal(actual_h[:, : phi.shape[0]], expected_h[:, : phi.shape[0]])
    _assert_equal(actual_ms, expected_ms)

    scale, native_scale = _entry_and_native(module_name, "mhc_fused_scale")
    padded_h = torch.randn(s * b, 32, device="cuda")
    alpha = torch.randn(3, device="cuda")
    beta = torch.randn(1, 2 * n + n * n, device="cuda")
    ms = torch.rand(s * b, device="cuda") + 1
    _assert_equal(scale(padded_h, alpha, beta, ms, n), native_scale(padded_h, alpha, beta, ms, n))

    sinkhorn, native_sinkhorn = _entry_and_native(module_name, "mhc_fused_sinkhorn")
    sinkhorn_input = torch.randn(s, b, n, n, device="cuda")
    _assert_equal(
        sinkhorn(sinkhorn_input, n, True, 4),
        native_sinkhorn(sinkhorn_input, n, True, 4),
    )

    aggregate, native_aggregate = _entry_and_native(module_name, "mhc_fused_aggregate")
    x = torch.randn(s, b, hidden, n, device="cuda")
    h_pre = torch.randn(s, b, n, device="cuda")
    _assert_equal(aggregate(x, h_pre, n, False), native_aggregate(x, h_pre, n, False))

    expand, native_expand = _entry_and_native(module_name, "mhc_fused_expand_combine")
    f = torch.randn(s, b, hidden, device="cuda")
    bias = torch.randn(hidden, device="cuda")
    h_post = torch.randn(s, b, n, device="cuda")
    h_res = torch.randn(s, b, n, n, device="cuda")
    _assert_equal(
        expand(f, bias, h_post, x, h_res, n, False),
        native_expand(f, bias, h_post, x, h_res, n, False),
    )


def _captured_call_functions(fn, arg):
    """Compile ``fn`` and count the nodes Dynamo captured into its graphs."""
    import torch._dynamo as dynamo

    captured = []

    def backend(graph_module, _example_inputs):
        captured.append(sum(1 for node in graph_module.graph.nodes if node.op == "call_function"))
        return graph_module.forward

    dynamo.reset()
    torch.compile(fn, backend=backend)(arg)
    return sum(captured)


def test_dispatch_entry_does_not_shrink_the_compiled_graph():
    """A dispatch entry must stay traceable by Dynamo.

    Dynamo inlines the entry's bytecode into its caller, so a statement it
    cannot translate does not merely break the graph at that point -- it
    discards the whole caller frame, and the surrounding operators silently
    fall back to eager. A dict keyed by a Python object (the manager-keyed
    bookkeeping on the slow path) is exactly such a statement, which is why
    the repeat path must avoid it.

    Counting nodes rather than asserting a single graph on purpose: locks on
    the resolve path legitimately split the graph, and both sides stay
    captured. Losing the surrounding operators entirely is the regression.
    """
    entry, native = _entry_and_native(
        "transformer_engine.pytorch.triton.pad", "pad_columnwise_scale_inv"
    )
    inp = torch.randn(1024, 32, device="cuda")

    def caller(fn):
        def run(tensor):
            value = tensor
            for _ in range(10):
                value = value * 1.001 + 0.001
            result = fn(value)
            for _ in range(10):
                result = result * 1.001 + 0.001
            return result

        return run

    # The slow path runs once per op, in eager, before any compiled region.
    entry(inp)

    native_captured = _captured_call_functions(caller(native), inp)
    entry_captured = _captured_call_functions(caller(entry), inp)

    assert native_captured > 0
    assert entry_captured >= native_captured, (
        f"dispatch entry lost graph capture: {entry_captured} nodes vs "
        f"{native_captured} calling the native function directly. Dynamo "
        "discarded the caller frame, so the surrounding operators dropped "
        "back to eager."
    )
