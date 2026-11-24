# FlagOS attention layouts

The FlagGems SDPA adapter converts dense `SBHD` and `BSHD` inputs to contiguous
`BHSD` tensors and restores the original layout for the output and Q/K/V
gradients. Packed `THD` inputs are split using the Q and KV `cu_seqlens` arrays;
each sequence is dispatched separately so attention never crosses a sequence
boundary. The output remains contiguous for `FlashAttentionFL`'s final reshape.

The supported path uses FP16/BF16, zero attention dropout, no context
parallelism, no paged KV cache, and no padding between packed sequences.
Full attention and square causal attention are supported, including GQA/MQA.
For packed THD, `padding` and `padding_causal` describe the sequence boundaries
in `cu_seqlens`; dense padded inputs, sliding windows, ALiBi, FP8, and
non-square causal alignment are rejected rather than silently misinterpreted.

This is a correctness adapter over the dense SDPA interface. Reading packed
sequence bounds synchronizes metadata to the host; layout conversions may copy
tensors, and multiple packed sequences require multiple kernel calls. It is
not a native variable-length attention kernel, and no throughput improvement
is implied.

To run the numerical regressions with a compatible FlagGems installation and
CUDA/HIP accelerator:

```bash
python -m pytest -q tests/pytorch/attention/test_flagos_attention.py
```

The tests compare BF16 outputs and Q/K/V gradients to an explicit FP32
matmul/softmax reference. For packed inputs the reference uses a block-diagonal
mask, independently checking the per-sequence dispatch implementation.
