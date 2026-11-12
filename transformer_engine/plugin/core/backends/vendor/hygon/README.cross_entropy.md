# Hygon parallel cross entropy

The Hygon CE implementation preserves the caller's logits and stores the
precomputed input gradients in a separate FP32 buffer until backward scaling.
It follows the numerical fix by Przemek Tredak (`ptrendx`) in
[NVIDIA TransformerEngine PR #3193](https://github.com/NVIDIA/TransformerEngine/pull/3193),
commit `1f34594`. This is a vendor-local adaptation of that upstream correction.

## Activation and dispatch

```bash
export TE_FL_PREFER=vendor
export NVTE_HYGON_PARALLEL_CE=1
export NVTE_HYGON_CE_NUM_WARPS=16
```

`NVTE_HYGON_PARALLEL_CE` defaults to enabled when the Hygon backend is available.
The original `CrossEntropyFunction` is marked `@overridable`; its original
forward and backward bodies are unchanged. `CrossEntropyFunctionHygon` uses
`@override` and is registered as one complete class with ID `vendor.hygon`,
through the existing backend registry and policy.

The input predicate admits HIP tensors on the validated BW11* device family
with architecture `gfx938`, 3D BF16/FP32 logits and targets on the same device.
It checks the input tensor's actual device, not just the current device or
whether a Hygon library loaded. Other inputs use the original path if no
other registered implementation matches. In particular, this change does
not enable this kernel on BW1000, other HIP device families or NVIDIA devices.

The class is selected at `.apply()`, so PyTorch retains its matching backward
even if the policy or enable flag changes after forward. The public
`parallel_cross_entropy` signature is unchanged. See the
[decorator API](../../../README.decorators.md) for use at other entry points.

Set `NVTE_HYGON_PARALLEL_CE=0` to select the original path. That original path
retains its original numerical behavior and launch policy; disabling this
plugin is not a precision or Hygon launch fix. Supported Hygon warp settings
are 8 and 16, with 16 the default.

The Hygon wrapper passes this setting to all three CE launches:
`online_softmax_kernel`, `cross_entropy_kernel`, and `element_mul_kernel`.
The shared PyTorch/Triton wrapper retains its original `num_warps=32` launches.
The validated device reports BW1101, `gfx938`, and a warp size of 64. This is
the vendor-scoped launch configuration (T2), combined here with the FP32
gradient-storage correction (T1). The benchmarks compare working CE paths;
they do not establish an isolated performance gain from changing 32 to 16
warps. NVIDIA hardware execution has not been checked in this validation.

## Implementation boundary

- `cross_entropy.py`: the decorated autograd class, input compatibility check,
  FP32 gradient allocation, CE launches, TP statistics and backward scaling.
- `cross_entropy_kernel.py`: a local copy of the corrected upstream CE kernel;
  logits are read-only and gradient writes use `grad_input_ptr`.
- `register_ops.py`: register the decorated Hygon class after the existing
  backend availability checks.
- `plugin/core/decorators.py`: reusable function/class selection.
- `plugin/core/manager.py`: optional candidate lookup, preserving existing
  callers' default behavior.
- `pytorch/cross_entropy.py`: add an import and `@overridable` to the original
  autograd class. No CE-specific dispatch helper remains.

The shared `common/triton/cross_entropy.py` and
`pytorch/triton/cross_entropy.py` are unchanged. The shared online-softmax and
backward multiplication kernels are reused. The fix does not patch global
PyTorch functions or bypass Megatron/FlagScale CE validation guards.

The final input gradient dtype is handled by PyTorch autograd; a BF16 input
still receives a BF16 gradient. Its intermediate gradient and upstream-gradient
multiplication remain FP32.

## Scope and memory

External validation on BW1100 covers BF16/FP32, TP1/TP2, reduced/unreduced
losses, ignored labels, noncontiguous inputs, the actual vocabulary 248320,
and disabling the plugin between forward and backward. Fallback selection
and device filters are also checked with simulated other-vendor properties;
these checks are not execution tests on other vendors' hardware. Validation
scripts are retained separately and are not repository test files.

For TP greater than one, the baseline CE kernel has a separate loss-reduction
issue when `label_smoothing > 0`. This gradient-storage change does not fix it.
The validated distributed training setting uses `label_smoothing=0`; smoothed
loss is validated at TP1. Do not interpret this patch as distributed
label-smoothing support. CUDA graph capture itself, compilation, the full
repository test suite and other device families have not been validated here.

The extra FP32 gradient buffer holds `rows * local_vocab_size * 4` bytes.
For local vocab 124160 and 8192/16384 rows, this is approximately 3.789/7.578 GiB
per buffer. This buffer size is not a measurement of full-model peak-memory
increase. Standalone CE timings are not new full-model MFU measurements.
