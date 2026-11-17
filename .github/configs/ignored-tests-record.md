# Ignored tests: record of reasons, environment and reproduction

Every `ignored_tests` entry added or changed by the shared-flow alignment has a
one-line comment in `.github/configs/<platform>.yml` and its full record here.
Restore an entry when its reproduction command passes on that platform.

## The alignment run (2026-11-17)

Goal: every platform runs the same shared QA scripts for the `debug`,
`distributed` and `onnx` jobs, and platform differences exist only as
`ignored_tests` entries in its config. Before this change Enflame, Hygon,
Iluvatar, KunlunXin and PPU ran private file lists in their launchers, Ascend
ran `qa/*/test_ascend.sh` variants that skipped every shared test, and MetaX
ignored whole shared files that had never been re-verified.

Method:

- Commit `deb7ee4d` (branch `ci-triton-dispatch-tests`), each platform's CI image
  and `container_options`, setup script and `set_env.sh`, with the GITHUB_ENV that
  setup writes loaded before every step, as CI does.
- Every step of `qa/L0_pytorch_debug_unittest/test.sh`,
  `qa/L1_pytorch_distributed_unittest/test.sh` (including the steps that were
  commented out) and `qa/L1_pytorch_onnx_unittest/test.sh`, one pytest process per
  file, **with no ignore list**, on idle devices isolated with the platform's
  visibility variable.
- A step whose process died from a device fault was rerun with the faulting case
  deselected (up to 10 times). A step with no new test result for 15 minutes was
  killed, its last case recorded as a hang and deselected.
- Entries were generated from the JUnit results: an existing entry that hid
  passing cases no longer applies to that job; a failing case not covered gets a
  job-scoped entry, compressed to the most general pattern that matches no
  passing case. The `pytorch_unit` job was not part of the run and keeps its
  entries unchanged.

Not covered by the run: the `pytorch_unit` job on every platform; the KunlunXin
`debug` job (still running when this was written; its launcher is unchanged for
now); MUSA (machine network problem, not run); CUDA `debug` and `onnx` (already on
the shared scripts, unchanged).

## Root causes shared by several platforms

| Cause | Platforms | Effect | Owner |
|---|---|---|---|
| `transformer_engine/pytorch/quantization.py:181` evaluates `float(torch.version.cuda)` once the device reports compute capability >= 9.0; on DCU/NPU `torch.version.cuda` is `None` | Hygon, Ascend | `TypeError` at import, so whole files fail to collect: `test_sanity.py`, `test_numerics.py`, `debug/test_log.py` (both), plus `distributed/test_numerics.py`, `test_numerics_exact.py`, `test_torch_fsdp2.py` (Hygon) | TE core (`quantization.py`); one-line guard `torch.version.cuda is not None` |
| FP8 operators missing from the selected backend: `Operator 'quantize' not found`, `'fused_amax_and_scale_update_after_reduction' not found` | Hygon, Ascend | debug FP8 numerics/sanity cases and every FP8-recipe ONNX export | backend op coverage (FlagOS / vendor plugin) |
| `torch.compile` / inductor has no backend for the device: `KeyError: 'gcu'`, `BackendCompilerFailed` | Enflame, Ascend | debug logging/sanity cases and LayerNormMLP ONNX exports | vendor torch integration |
| Shared tests address CUDA directly: module-level `pytest.skip` when `torch.cuda.device_count() < 2`, `torch._C._cuda_setDevice`, `Torch not compiled with CUDA enabled` | Ascend | distributed files fail to collect; linear/LayerNorm ONNX exports fail | shared tests (device abstraction) |
| FlagGems norm kernels miss the FP16/BF16 tolerance in `test_layernorm_accuracy` | Iluvatar, MetaX (also CUDA/PPU, already ignored) | debug `test_numerics.py` cases | FlagGems norm |

## Shared steps kept disabled

`qa/L1_pytorch_distributed_unittest/test.sh` enables `distributed/test_fusible_ops.py`
and `debug/test_distributed.py` (they pass on most platforms; the failures are
per-platform entries below). These stay commented out because they failed on
every platform, so they are not a platform difference:

| Step | Result per platform |
|---|---|
| `distributed/test_sanity.py` | `test_current_device` fails on all 8 (KunlunXin hangs); the other cases pass |
| `distributed/test_comm_gemm_overlap.py` | collection error on 7 (`ubuf_built_with_mpi` missing, or module-level skip); CUDA: 37 failed, 97 skipped |
| `distributed/test_fusible_ops_with_userbuffers.py` | fails on all 8 |
| `attention/test_attention_with_cp.py` | `test_cp_with_fused_attention` fails (20 on CUDA/MetaX/KunlunXin, 17 on PPU); 10304 skipped elsewhere; Hygon timed out after 3600 s |

## Hangs killed during the run

| Platform | Step | Case |
|---|---|---|
| kunlun | debug D5 | `tests/pytorch/debug/test_api_features.py::test_per_tensor_scaling` |
| kunlun | debug D5 rerun | `tests/pytorch/debug/test_api_features.py::test_statistics_multi_run` |
| kunlun | debug D6 | `tests/pytorch/debug/test_perf.py::test_layer_switches_to_nondebug_mode[False]` |
| kunlun | distributed (step kept disabled) | `tests/pytorch/distributed/test_sanity.py::test_current_device[DotProductAttention-small]`, `[Linear-small]` |
| kunlun | distributed | `tests/pytorch/distributed/test_fusible_ops.py::test_distributed_fuser_ops[4]` |

The KunlunXin debug hangs apply once its debug job moves to the shared script.

Enflame `debug`: `tests/pytorch/test_numerics.py::test_gpt_accuracy` aborts the
process (`Fatal Python error: Aborted`, SIGABRT, core dumped) at its first case in
four consecutive runs. The functions after it in that file (`test_mha_accuracy`
onward) never ran in this experiment; CI is their first run.

## Per platform

### cuda

- Machine: A100 (sm80); image `harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-cuda-ci`
- Versions: torch 2.13.0+cu130, flag_gems 5.4.0, onnxruntime 1.31.1
- Devices: `CUDA_VISIBLE_DEVICES` dist=3,4,5,6

| job | entry | cases | first effective error | repro (CI container, after `source tests/plugin/backend/cuda/set_env.sh`) |
|---|---|---|---|---|
| `pytorch_distributed` | `tests/pytorch/distributed/test_fusible_ops.py::test_distributed_fuser_ops[4]` | 1 | subprocess.CalledProcessError: Command '[PosixPath('/flagos/bin/python3'), '-m', 'torch.distributed.run', '--nproc_per_node=4', PosixPath('/work/src/tests/pytor | `pytest tests/pytorch/distributed/test_fusible_ops.py -k "test_distributed_fuser_ops"` |

### ppu

- Machine: T-Head PPU 810E; image `harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-thead-ci`
- Versions: torch 2.10.0+v0.1.0.ppu2.1.1, flag_gems 5.4.0, onnxruntime 1.31.1; TE_FL_PER_OP=generic_gemm=reference
- Devices: `CUDA_VISIBLE_DEVICES` debug=8 onnx=9 dist=10,11

Narrowed (the entry hid cases that passed in these jobs; it now applies to the remaining jobs only):

- `tests/pytorch/test_numerics.py::test_transformer_layer_hidden_states_format[*126m-2-dtype*]` — no longer applies to `pytorch_debug`
- `tests/pytorch/test_sanity.py::test_sanity_grouped_linear[*-1-dtype*]` — no longer applies to `pytorch_debug`
- `tests/pytorch/test_sanity.py::test_sanity_grouped_linear[*-2-dtype*]` — no longer applies to `pytorch_debug`

### iluvatar

- Machine: Iluvatar BI-V150; image `harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-iluvatar-ci`
- Versions: torch 2.10.0, flag_gems 5.4.0, onnxruntime 1.31.1
- Devices: `CUDA_VISIBLE_DEVICES` debug=8 onnx=9 dist=10-15 (physical ids)

Narrowed (the entry hid cases that passed in these jobs; it now applies to the remaining jobs only):

- `tests/pytorch/debug/test_api_features.py::test_disable_fp8_gemm` — no longer applies to `pytorch_debug`
- `tests/pytorch/debug/test_api_features.py::test_disable_fp8_layer` — no longer applies to `pytorch_debug`
- `tests/pytorch/distributed/test_cast_master_weights_to_fp8.py` — no longer applies to `pytorch_distributed`
- `tests/pytorch/distributed/test_numerics.py` — no longer applies to `pytorch_distributed`
- `tests/pytorch/test_numerics.py` — no longer applies to `pytorch_debug`
- `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_activation[clamped_swiglu]` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_activation[geglu]` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_activation[reglu]` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_activation[relu]` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation[clamped_swiglu]` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation[geglu]` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation[reglu]` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation[relu]` — no longer applies to `pytorch_onnx`

| job | entry | cases | first effective error | repro (CI container, after `source tests/plugin/backend/iluvatar/set_env.sh`) |
|---|---|---|---|---|
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_layernorm_accuracy[*-*-*-*-dtype1]` | 16 | AssertionError: Outputs not close enough in tensor at idx=4. Maximum difference at location [0] with 5.44140625 vs 5.44921875 (diff 0.0078125). | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_layernorm_accuracy"` |

### metax

- Machine: MetaX C550; image `harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-metax-ci`
- Versions: torch 2.8.0+metax3.7.2.0, flag_gems 5.4.0, onnxruntime 1.23.2
- Devices: `CUDA_VISIBLE_DEVICES` debug=0 onnx=1 dist=2-5 (a vllm-plugin-FL CI job shared the cards for part of the run)

Narrowed (the entry hid cases that passed in these jobs; it now applies to the remaining jobs only):

- `tests/pytorch/debug/test_sanity.py` — no longer applies to `pytorch_debug`
- `tests/pytorch/test_numerics.py` — no longer applies to `pytorch_debug`
- `tests/pytorch/test_onnx_export.py::test_export_core_attention` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_recipe` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_sanity.py` — no longer applies to `pytorch_debug`

| job | entry | cases | first effective error | repro (CI container, after `source tests/plugin/backend/metax/set_env.sh`) |
|---|---|---|---|---|
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[fake_quant-*-mha_attention]` | 1 | RuntimeError: All 1 implementation(s) failed for op='quantize'. Last error: /workspace/TransformerEngine/transformer_engine/common/util/cuda_runtime.cpp:196 in  | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[fake_quant-*-transformer_layer]` | 1 | RuntimeError: All 1 implementation(s) failed for op='quantize'. Last error: /workspace/TransformerEngine/transformer_engine/common/util/cuda_runtime.cpp:196 in  | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_dpa_accuracy[*-*-dtype1]` | 2 | AssertionError: Outputs not close enough in tensor at idx=0. Maximum difference at location [0, 650] with 3.091796875 vs -0.9990234375 (diff 4.08984375). | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_dpa_accuracy"` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_dpa_accuracy[*-*-dtype2]` | 2 | AssertionError: Outputs not close enough in tensor at idx=0. Maximum difference at location [0, 650] with 3.09375 vs -1.0 (diff 4.09375). | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_dpa_accuracy"` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_gpt_checkpointing[*-2-dtype2]` | 1 | AssertionError: Mismatch in tensor 4 | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_gpt_checkpointing"` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_layernorm_accuracy[*-*-*-*-1-*]` | 12 | AssertionError: Outputs not close enough in tensor at idx=651. Maximum difference at location [0] with 5.727876663208008 vs 5.727760314941406 (diff 0.0001163482 | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_layernorm_accuracy"` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_layernorm_accuracy[*-*-*-*-dtype1]` | 2 | AssertionError: Outputs not close enough in tensor at idx=11. Maximum difference at location [0] with -3.353515625 vs -3.34375 (diff 0.009765625). | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_layernorm_accuracy"` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_layernorm_accuracy[*-*-*-*-dtype2]` | 2 | AssertionError: Outputs not close enough in tensor at idx=11. Maximum difference at location [0] with -3.171875 vs -3.265625 (diff 0.09375). | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_layernorm_accuracy"` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_layernorm_accuracy[*-*-*-2-*]` | 24 | AssertionError: Outputs not close enough in tensor at idx=57. Maximum difference at location [0] with 66.87229919433594 vs 66.8720703125 (diff 0.000228881835937 | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_layernorm_accuracy"` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_layernorm_accuracy[*-0.001-*-*-*]` | 6 | AssertionError: Outputs not close enough in tensor at idx=343. Maximum difference at location [0] with 62.41499710083008 vs 62.4147834777832 (diff 0.00021362304 | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_layernorm_accuracy"` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_transformer_layer_hidden_states_format[*-*-dtype1]` | 2 | AssertionError: GEMM not possible | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_transformer_layer_hidden_states_format"` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_transformer_layer_hidden_states_format[*-*-dtype2]` | 2 | AssertionError: GEMM not possible | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_transformer_layer_hidden_states_format"` |
| `pytorch_debug` | `tests/pytorch/test_sanity.py::test_sanity_amp_and_nvfuser[*-*-*-dtype0]` | 2 | RuntimeError: All 3 implementation(s) failed for op='get_flash_attention_class'. Last error: FlashAttention only support fp16 and bf16 data type | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_sanity.py -k "test_sanity_amp_and_nvfuser"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_core_attention[precision0-*-*-*]` | 1 | ValueError: Output validation of /tmp/./gen_onnx_models/te.core_attention_arbitrary-mask_fp32.onnx failed with 16013 errors | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_core_attention"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_core_attention[precision3-*-*-*]` | 1 | ValueError: Output validation of /tmp/./gen_onnx_models/te.core_attention_arbitrary-mask_fp16.onnx failed with 16043 errors | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_core_attention"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation[clamped_swiglu]` | 1 | ValueError: Output validation of /tmp/./gen_onnx_models/te.transformer_layer_arbitrary-mask_fused-qkv_fp32_clamped_swiglu.onnx failed with 3123 errors | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_activation"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation[geglu]` | 1 | ValueError: Output validation of /tmp/./gen_onnx_models/te.transformer_layer_arbitrary-mask_fused-qkv_fp32_geglu.onnx failed with 3144 errors | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_activation"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation[reglu]` | 1 | ValueError: Output validation of /tmp/./gen_onnx_models/te.transformer_layer_arbitrary-mask_fused-qkv_fp32_reglu.onnx failed with 3181 errors | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_activation"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation[relu]` | 1 | ValueError: Output validation of /tmp/./gen_onnx_models/te.transformer_layer_arbitrary-mask_fused-qkv_fp32_relu.onnx failed with 3378 errors | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_activation"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_recipe[precision0-*]` | 1 | ValueError: Output validation of /tmp/./gen_onnx_models/te.transformer_layer_arbitrary-mask_fused-qkv_fp32_gelu.onnx failed with 3332 errors | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_recipe"` |

### enflame

- Machine: Enflame S60; image `harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-enflame-ci-gems5.0.2-20261112`
- Versions: torch 2.10.0+cpu + torch_gcu, flag_gems 5.0.2, onnxruntime 1.28.0, pytest 9.0.3 (PEP 668 image)
- Devices: `TOPS_VISIBLE_DEVICES` debug=1 onnx=2 dist=3,4

Narrowed (the entry hid cases that passed in these jobs; it now applies to the remaining jobs only):

- `tests/pytorch/distributed/test_torch_fsdp2.py` — no longer applies to `pytorch_distributed`
- `tests/pytorch/test_onnx_export.py::test_export_layernorm_linear*` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp*` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_linear*` — no longer applies to `pytorch_onnx`

| job | entry | cases | first effective error | repro (CI container, after `source tests/plugin/backend/enflame/set_env.sh`) |
|---|---|---|---|---|
| `pytorch_debug` | `tests/pytorch/debug/test_api_features.py::test_per_tensor_scaling` | 1 | RuntimeError: All 1 implementation(s) failed for op='quantize'. Last error: Failed to load Enflame Transformer Engine extension: API for FP8GlobalStateManager n | `pytest tests/pytorch/debug/test_api_features.py -k "test_per_tensor_scaling" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_log.py::test_log_stats_numerics` | 3 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `pytest tests/pytorch/debug/test_log.py -k "test_log_stats_numerics" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_logging_zero_numel_tensor` | 1 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `pytest tests/pytorch/debug/test_numerics.py -k "test_logging_zero_numel_tensor" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[*-*-layernorm_mlp]` | 3 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[*-*-transformer_layer]` | 3 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[fake_quant-*-mha_attention]` | 1 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[log-*-mha_attention]` | 1 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py::test_gpt_accuracy` | 1 | Fatal Python error: Aborted (SIGABRT, core dumped) at the first test_gpt_accuracy case, in 4 consecutive runs | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py -k "test_gpt_accuracy"` |
| `pytorch_debug` | `tests/pytorch/test_sanity.py::test_sanity_amp_and_nvfuser[*-*-*-dtype1]` | 2 | TypeError: Data types for parameters must match when outside of autocasted region. Found input dtype: torch.float16 and 'weight' dtype: torch.float32 | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_sanity.py -k "test_sanity_amp_and_nvfuser"` |
| `pytorch_debug` | `tests/pytorch/test_sanity.py::test_sanity_normalization_amp[*-*-*-*-dtype1]` | 12 | AssertionError: AMP wrong output type. | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_sanity.py -k "test_sanity_normalization_amp"` |
| `pytorch_distributed` | `tests/pytorch/debug/test_distributed.py::test_debug_distributed` | 1 | AssertionError: torchrun exited with 1 | `pytest tests/pytorch/debug/test_distributed.py -k "test_debug_distributed" --feature_dirs=$TE_PATH/transformer_engine/debug/features` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_torch_fsdp2.py::test_fsdp2_mem_leak_tests` | 1 | AssertionError: Inner pytest failed with exit code 1 | `pytest tests/pytorch/distributed/test_torch_fsdp2.py -k "test_fsdp2_mem_leak_tests"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp` | 3 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_no*` | 2 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp_no"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_return*` | 2 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp_return"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_zero_centered_gamma` | 1 | torch._inductor.exc.InductorError: KeyError: 'gcu' | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp_zero_centered_gamma"` |

### hygon

- Machine: Hygon BW1000 (DCU); image `harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-hygon-ci`
- Versions: torch 2.13.0, flag_gems 5.4.0, onnxruntime 1.31.1
- Devices: `HIP_VISIBLE_DEVICES` debug=6 onnx=7 dist=0-5 (host shared with a flagscale-train job)

Narrowed (the entry hid cases that passed in these jobs; it now applies to the remaining jobs only):

- `tests/pytorch/test_onnx_export.py::test_export_core_attention*` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_gpt_generation*` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_layernorm_linear*` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp*` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_linear*` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_multihead_attention*` — no longer applies to `pytorch_onnx`
- `tests/pytorch/test_onnx_export.py::test_export_transformer_layer*` — no longer applies to `pytorch_onnx`

| job | entry | cases | first effective error | repro (CI container, after `source tests/plugin/backend/hygon/set_env.sh`) |
|---|---|---|---|---|
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_disable_fp8*` | 9 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_numerics.py -k "test_disable_fp8" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_fake_quant_fp8` | 10 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_numerics.py -k "test_fake_quant_fp8" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_microbatching_per_tensor_scaling` | 20 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_numerics.py -k "test_microbatching_per_tensor_scaling" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_per_tensor_scaling` | 20 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_numerics.py -k "test_per_tensor_scaling" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[*-True-*]` | 20 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[fake_quant-*-mha_attention]` | 1 | SystemExit: 1 | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[fake_quant-*-transformer_layer]` | 1 | SystemExit: 1 | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py` | 1 | collection failure | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py` |
| `pytorch_distributed` | `tests/pytorch/debug/test_distributed.py::test_debug_distributed` | 1 | AssertionError: torchrun exited with 1 | `pytest tests/pytorch/debug/test_distributed.py -k "test_debug_distributed" --feature_dirs=$TE_PATH/transformer_engine/debug/features` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_cast_master_weights_to_fp8.py::test_cast_master_weights_to_fp8` | 1 | subprocess.CalledProcessError: Command '[PosixPath('/flagos/bin/python3'), '-m', 'torch.distributed.run', '--nproc_per_node=2', PosixPath('/work/src/tests/pytor | `pytest tests/pytorch/distributed/test_cast_master_weights_to_fp8.py -k "test_cast_master_weights_to_fp8"` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_fusible_ops.py::test_distributed_fuser_ops` | 3 | subprocess.CalledProcessError: Command '[PosixPath('/flagos/bin/python3'), '-m', 'torch.distributed.run', '--nproc_per_node=6', PosixPath('/work/src/tests/pytor | `pytest tests/pytorch/distributed/test_fusible_ops.py -k "test_distributed_fuser_ops"` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_numerics.py` | 1 | collection failure | `pytest tests/pytorch/distributed/test_numerics.py` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_numerics_exact.py` | 1 | collection failure | `pytest tests/pytorch/distributed/test_numerics_exact.py` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_torch_fsdp2.py` | 1 | collection failure | `pytest tests/pytorch/distributed/test_torch_fsdp2.py` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_core_attention[*-*-*-fp8_recipe1]` | 8 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_core_attention"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_core_attention[*-*-*-fp8_recipe2]` | 8 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_core_attention"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_gpt_generation[*-fp8_recipe0]` | 2 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_gpt_generation"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_gpt_generation[*-fp8_recipe1]` | 2 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_gpt_generation"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_linear_no*` | 2 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_linear_no"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_linear_recipe[*-fp8_recipe0]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_linear_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_linear_recipe[*-fp8_recipe1]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_linear_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_linear_return*` | 2 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_linear_return"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_linear_zero_centered_gamma` | 1 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_linear_zero_centered_gamma"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp[*-fp8_recipe0]` | 3 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp[*-fp8_recipe1]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_activation` | 5 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp_activation"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_no*` | 2 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp_no"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_return*` | 2 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp_return"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp_zero_centered_gamma` | 1 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp_zero_centered_gamma"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_linear_recipe[*-fp8_recipe0]` | 3 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_linear_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_linear_recipe[*-fp8_recipe1]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_linear_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_linear_return_bias` | 2 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_linear_return_bias"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_linear_use_bias` | 2 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_linear_use_bias"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_cross_attn` | 1 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_cross_attn"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_no*` | 2 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_no"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_recipe[*-fp8_recipe0]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_recipe[*-fp8_recipe1]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_unfused_qkv_params` | 1 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_unfused_qkv_params"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_activation` | 5 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_activation"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_no_mask` | 1 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_no_mask"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_output_layernorm` | 1 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_output_layernorm"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_recipe[*-fp8_recipe0]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_recipe[*-fp8_recipe1]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_unfused_qkv_params` | 1 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_unfused_qkv_params"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer_zero_centered_gamma` | 1 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer_zero_centered_gamma"` |

### ascend

- Machine: Ascend 910C; image `harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-ascend-ci`
- Versions: torch 2.10.0+cpu + torch_npu, flag_gems 5.4.0, onnxruntime 1.31.1; NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1
- Devices: `ASCEND_RT_VISIBLE_DEVICES` debug=0 onnx=1 dist=2,3 (the container maps only davinci0-3)

| job | entry | cases | first effective error | repro (CI container, after `source tests/plugin/backend/ascend/set_env.sh`) |
|---|---|---|---|---|
| `pytorch_debug` | `tests/pytorch/debug/test_log.py` | 1 | collection failure | `pytest tests/pytorch/debug/test_log.py --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_disable_fp8*` | 9 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_numerics.py -k "test_disable_fp8" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_fake_quant_fp8` | 10 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_numerics.py -k "test_fake_quant_fp8" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_logging_zero_numel_tensor` | 1 | torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: | `pytest tests/pytorch/debug/test_numerics.py -k "test_logging_zero_numel_tensor" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_microbatching_per_tensor_scaling` | 20 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_numerics.py -k "test_microbatching_per_tensor_scaling" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_numerics.py::test_per_tensor_scaling` | 20 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_numerics.py -k "test_per_tensor_scaling" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[*-*-layernorm_mlp]` | 2 | torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[*-*-transformer_layer]` | 2 | torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[*-True-*]` | 15 | AttributeError: Operator 'fused_amax_and_scale_update_after_reduction' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnor | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/debug/test_sanity.py::test_sanity_debug[log-*-mha_attention]` | 1 | torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: | `pytest tests/pytorch/debug/test_sanity.py -k "test_sanity_debug" --feature_dirs=$TE_PATH/transformer_engine/debug/features --configs_dir=$TE_PATH/tests/pytorch/debug/test_configs/` |
| `pytorch_debug` | `tests/pytorch/test_numerics.py` | 1 | collection failure | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_numerics.py` |
| `pytorch_debug` | `tests/pytorch/test_sanity.py` | 1 | collection failure | `NVTE_TEST_NVINSPECT_ENABLED=1 NVTE_TEST_NVINSPECT_CONFIG_FILE=$TE_PATH/tests/pytorch/debug/test_configs/dummy_feature.yaml NVTE_TEST_NVINSPECT_FEATURE_DIRS=$TE_PATH/transformer_engine/debug/features pytest tests/pytorch/test_sanity.py` |
| `pytorch_distributed` | `tests/pytorch/debug/test_distributed.py` | 1 | collection failure | `pytest tests/pytorch/debug/test_distributed.py --feature_dirs=$TE_PATH/transformer_engine/debug/features` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_fusible_ops.py::test_distributed_fuser_ops` | 2 | subprocess.CalledProcessError: Command '[PosixPath('/flagos/bin/python3'), '-m', 'torch.distributed.run', '--nproc_per_node=0', PosixPath('/work/src/tests/pytor | `pytest tests/pytorch/distributed/test_fusible_ops.py -k "test_distributed_fuser_ops"` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_numerics.py` | 1 | collection failure | `pytest tests/pytorch/distributed/test_numerics.py` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_numerics_exact.py` | 1 | collection failure | `pytest tests/pytorch/distributed/test_numerics_exact.py` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_torch_fsdp2.py` | 1 | collection failure | `pytest tests/pytorch/distributed/test_torch_fsdp2.py` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_core_attention[*-*-*-fp8_recipe1]` | 8 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_core_attention"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_core_attention[*-*-*-fp8_recipe2]` | 8 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_core_attention"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_gpt_generation` | 6 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_gpt_generation"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_linear*` | 14 | AssertionError: Torch not compiled with CUDA enabled | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_linear"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_layernorm_mlp*` | 19 | AssertionError: Torch not compiled with CUDA enabled | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_layernorm_mlp"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_linear*` | 13 | AssertionError: Torch not compiled with CUDA enabled | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_linear"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_cross_attn` | 1 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_cross_attn"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_no*` | 2 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_no"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_recipe[*-fp8_recipe0]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_recipe[*-fp8_recipe1]` | 3 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_recipe"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_multihead_attention_unfused_qkv_params` | 1 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_multihead_attention_unfused_qkv_params"` |
| `pytorch_onnx` | `tests/pytorch/test_onnx_export.py::test_export_transformer_layer*` | 18 | AttributeError: Operator 'quantize' not found. Available operators: ['layernorm_fwd', 'layernorm_bwd', 'rmsnorm_fwd', 'rmsnorm_bwd', 'generic_gemm', 'te_general | `NVTE_UnfusedDPA_Emulate_FP8=1 pytest tests/pytorch/test_onnx_export.py -k "test_export_transformer_layer"` |

### kunlun

- Machine: KunlunXin P800; image `harbor.baai.ac.cn/flagos-dev/transformerengine-fl:b98d43b-kunlun-dev (conda env python310_torch29_cuda)`
- Versions: torch 2.9.0+cu129, flag_gems 5.4.0dev, onnxruntime 1.23.2
- Devices: `XPU_VISIBLE_DEVICES` onnx=2 dist=3,4,5,6

Narrowed (the entry hid cases that passed in these jobs; it now applies to the remaining jobs only):

- `tests/pytorch/distributed/test_cast_master_weights_to_fp8.py::test_cast_master_weights_to_fp8[2]` — passed in this run but **kept**: its entry records an intermittent failure whose restore condition is repeated passing runs, and one pass does not meet it

| job | entry | cases | first effective error | repro (CI container, after `source tests/plugin/backend/kunlun/set_env.sh`) |
|---|---|---|---|---|
| `pytorch_distributed` | `tests/pytorch/debug/test_distributed.py::test_debug_distributed` | 1 | AssertionError: torchrun exited with 1 | `pytest tests/pytorch/debug/test_distributed.py -k "test_debug_distributed" --feature_dirs=$TE_PATH/transformer_engine/debug/features` |
| `pytorch_distributed` | `tests/pytorch/distributed/test_fusible_ops.py::test_distributed_fuser_ops` | 1 | HANG: the first case ([4]) produced no result until torchrun was terminated (SIGTERM); no case completed | `pytest tests/pytorch/distributed/test_fusible_ops.py -k "test_distributed_fuser_ops"` |

