# BW1100 BF16 expert GEMMs

This opt-in implementation uses the existing `vendor.hygon` grouped-GEMM
registration. Enable it before training with:

```bash
export TE_FL_PREFER=vendor
export NVTE_HYGON_EXPERT_GEMM=1
```

The default is off. Unsetting the second variable or setting it to `0` restores
native dispatch. The implementation has no dependency on the CE changes in
PR #149, and changes no FlagScale, Megatron, GroupedLinear autograd, routing,
pipeline, optimizer, or shared GEMM implementation.

## Admitted calls

- HIP BW11* / gfx938 devices; same-device contiguous ordinary BF16 tensors.
- Exactly 32 local experts; FC1 weights `[1024, 2048]` or FC2 `[2048, 512]`.
- TN forward, NN input gradients, and NT weight gradients, with native token
  splits, including empty, unaligned, and imbalanced experts.
- No bias, GELU, quantized tensors, or `accumulate=True`.
  Compile, graph capture, and higher-order-autograd calls use native dispatch.
- `use_split_accumulator` selects FP8 accumulation behavior and has no effect
  here: BF16 GroupedLinear passes both values, including `True` in backward.
- BF16 output dtype. dW is written in BF16 and the existing GroupedLinear
  code accumulates it into FP32 `main_grad` when gradient accumulation fusion
  is disabled. The fast path does not implement FP32/fused dW accumulation.

All arguments and output shapes are checked before writing any output. Each
expert uses its actual token count with `torch.mm(..., out=...)`; no routing
changes or padding are introduced. Unsupported calls use the original native
backend. Execution errors propagate rather than retrying a partially completed
operation. Calls with a nonzero SM-count target use native dispatch; the opt-in
BLAS loop does not implement SM partitioning. Native workspace arguments remain
unchanged on fallback.

## Evidence and validation boundary

The external BW1100 experiment `gemm019` validated the three expressions with
native TE GroupedLinear, four-rank expert DDP and distributed Adam, including
empty and imbalanced experts. The runtime adapter kept BF16 dW before FP32
accumulation. `v037` then completed 20 full-model steps on 32 cards.

With identical batch/position handling in both model runs, `v027` averaged
142.1045 seconds/step and `v037` 119.0326 seconds/step: throughput increased
19.38%, and MFU340 increased from 14.0968% to 16.8292%. These are mock-data,
random-initialization results for the frozen runtime adapter. They are not
new measurements of this backend integration, or measurements with the
FlagScale batch compatibility adapter removed.

The new tests cover routing guards, all three matrix identities, empty expert
outputs, BF16 dW followed by FP32 accumulation, and native fallback. CPU tests
do not establish GPU numerical error or performance. Before promotion, rerun
four-rank GroupedLinear/DDP/Adam validation and full-model timing on this exact
commit. The existing opt-in default keeps the draft integration inactive until
explicitly selected.
