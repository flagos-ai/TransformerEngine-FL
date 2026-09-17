---
name: e2e-stage-manager
description: Manage end-to-end training test stages for FlagScale/TransformerEngine experiments. Use this skill when the user wants to create a unified stage configuration, migrate between stages (stage9/10/11), generate new stage configs, run batch training tests, or clean up old stage directories. Trigger on phrases like "create unified stage", "migrate stage configs", "run e2e tests", "clean up stages", "generate stage configs", or "consolidate training runs".
---

# E2E Stage Manager

Manage end-to-end training test stages for FlagScale/TransformerEngine experiments. This skill helps you create unified stage configurations, migrate between stages, run batch tests, and clean up old runs.

## Context

The user's repo contains training test stages (stage9_runs, stage10_runs, stage11_runs) that test different TransformerEngine implementations:
- **Models**: Qwen3-32b, DeepSeek-V3
- **TE implementations**: flagos, reference, vendor
- **Attention backends**: flash, fused, unfused (12 configs total per stage)

Stage configurations are stored in `FlagScale/run_configs/stage{N}_*/` as Hydra YAML configs.

## Stage Evolution

**Stage 9 (Legacy):**
- Contains `legacy_tokenizer: true` in tokenizer config
- Missing `eval_interval` in model config
- `te_fl_prefer` parameter positioned after optimizer section

**Stage 10+ (Current Standard):**
- Added `eval_interval: 1000` to model config
- Removed `legacy_tokenizer: true` from tokenizer config
- Moved `te_fl_prefer` parameter position (after `transformer_impl`)
- Re-runs use incremented stage numbers (stage11, stage12, ...) with identical format

The latest deployed configs are stage11. Use Stage 10 format for all new stages.

## Core Operations

### 1. Create Unified Stage

Generate a new unified stage configuration based on stage10 standard.

**Steps:**
1. Create new stage directory structure: `FlagScale/run_configs/stage{N}_*/`
2. For each configuration (model × implementation × backend):
   - Copy stage10 YAML structure
   - Update `exp_name` to new stage number
   - Update `exp_dir` path
   - Ensure `eval_interval: 1000` is present
   - Ensure `legacy_tokenizer` is removed
3. Create run directory: `stage{N}_runs/`

**Config naming pattern:**
```
stage{N}_te_fl_prefer-{impl}__attention_backend-{backend}/train.yaml
stage{N}_te_fl_prefer-{impl}__attention_backend-{backend}__deepseek_v3/train.yaml
```

Where:
- `{impl}` = flagos | reference | vendor
- `{backend}` = flash | fused | unfused

### 2. Run Batch Tests

Launch all 12 training configurations for a stage.

**Steps:**
1. Read the stage's config directory
2. For each config YAML:
   - Launch via FlagScale runner: `cd FlagScale && python -m flagscale.launcher.runner <config_path>`
   - Capture output to `stage{N}_runs/logs/{model}_{impl}_{backend}.log`
   - Track exit codes
3. Generate summary: `stage{N}_runs/logs/summary.txt` with PASS/FAIL for each config

**Parallel execution:**
Run configs in parallel when possible (different models/configs don't conflict).

### 3. Clean Up Old Stages

Remove old stage directories while preserving important artifacts.

**Steps:**
1. Ask user which stages to remove (default: keep only latest)
2. For each stage to remove:
   - Archive logs to `backup/stage{N}_logs.tar.gz`
   - Remove `stage{N}_runs/` directory
   - Remove `FlagScale/run_configs/stage{N}_*/` configs
3. Report disk space freed

**Safety:** Always confirm before deletion. Preserve summary.txt and final logs.

### 4. Compare Stages

Compare configurations or results between two stages.

**Config diff:**
```bash
diff -r FlagScale/run_configs/stage9_* FlagScale/run_configs/stage10_*
```

**Results comparison:**
- Parse summary.txt from both stages
- Show pass/fail differences
- Compare timing/memory metrics if available

### 5. Migrate Stage Configs

Upgrade old stage configs to new standard (e.g., stage9 → stage10 format).

**Steps:**
1. For each config in old stage:
   - Load YAML
   - Add `eval_interval: 1000` under `model:`
   - Remove `legacy_tokenizer: true` from `data.tokenizer:`
   - Reorder `te_fl_prefer` to come after `transformer_impl`
   - Update stage number in paths
2. Write to new stage config directory
3. Validate YAML syntax

## File Structure

```
repo_update/
├── FlagScale/
│   └── run_configs/
│       ├── stage9_te_fl_prefer-flagos__attention_backend-fused/
│       │   ├── train.yaml          # Main config
│       │   └── train/32b.yaml      # Model-specific overrides
│       └── stage10_te_fl_prefer-flagos__attention_backend-fused/
│           ├── train.yaml
│           └── train/32b.yaml
├── stage9_runs/
│   ├── Qwen3-32b-Stage9-flagos-fused/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── tensorboard/
│   │   └── wandb/
│   └── logs/
│       └── summary.txt
└── stage10_runs/
    └── logs/
        └── summary.txt
```

## Config Template (Stage 10 Standard)

### Qwen3-32b Config

```yaml
defaults:
  - _self_
  - train: 32b

experiment:
  exp_name: Qwen3-32b-Stage{N}-{impl}-{backend}
  seed: 42
  save_steps: 10000
  load: null
  exp_dir: <WORKSPACE_DIR>/stage{N}_runs/${experiment.exp_name}
  ckpt_format: torch
  task:
    type: train
    backend: megatron
    entrypoint: flagscale/train/megatron/train_gpt.py
  runner:
    per_node_task: false
    no_shared_fs: false
    rdzv_backend: static
    hostfile: <WORKSPACE_DIR>/host_single
    ssh_port: 7878
  cmds:
    before_start: ulimit -n 1048576 && source <CONDA_ENV_ACTIVATE_CMD>
  envs:
    LOGLEVEL: "INFO"
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    CUDA_DEVICE_MAX_CONNECTIONS: 1

action: run

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
```

### Model Config (train/32b.yaml)

```yaml
system:
  no_shared_fs: ${experiment.runner.no_shared_fs}
  num_workers: 2
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 1
  context_parallel_size: 1
  sequence_parallel: true
  use_distributed_optimizer: true
  overlap_grad_reduce: true
  overlap_param_gather: true
  precision:
    bf16: true
    attention_softmax_in_fp32: true
    accumulate_allreduce_grads_in_fp32: true
  logging:
    log_interval: 1
    tensorboard_log_interval: 1
    wandb_project: ${experiment.exp_name}
    wandb_exp_name: ${experiment.exp_name}
    log_timers_to_tensorboard: true
    log_validation_ppl_to_tensorboard: true
    log_throughput: true
    log_params_norm: true
    log_num_zeros_in_grad: true
    log_memory_to_tensorboard: true
  checkpoint:
    save_interval: ${experiment.save_steps}
    load: ${experiment.load}
    ckpt_format: ${experiment.ckpt_format}

model:
  transformer_impl: transformer_engine
  te_fl_prefer: {impl}
  attention_backend: {backend}
  num_layers: 16
  hidden_size: 5120
  ffn_hidden_size: 25600
  num_attention_heads: 64
  kv_channels: 128
  group_query_attention: true
  num_query_groups: 8
  seq_length: 4096
  max_position_embeddings: 40960
  norm_epsilon: 1.0e-06
  use_rotary_position_embeddings: true
  rotary_base: 1000000
  swiglu: true
  normalization: RMSNorm
  qk_layernorm: true
  init_method_std: 0.02
  attention_dropout: 0.0
  hidden_dropout: 0.0
  untie_embeddings_and_output_weights: true
  no_position_embedding: true
  no_rope_fusion: true
  disable_bias_linear: true
  seed: ${experiment.seed}
  finetune: false
  micro_batch_size: 1
  global_batch_size: 8
  eval_iters: 0
  eval_interval: 1000  # Stage 10 addition
  train_iters: 20
  optimizer:
    clip_grad: 1.0
    weight_decay: 0.1
    adam_beta1: 0.9
    adam_beta2: 0.95
    lr_scheduler:
      lr: 0.003
      min_lr: 0.0003
      lr_warmup_fraction: 0.1
      lr_decay_style: WSD
      lr_wsd_decay_style: cosine
      lr_wsd_decay_iters: 10

data:
  reset_position_ids: true
  reset_attention_mask: true
  data_path: <DATA_PATH>
  split: 1
  no_mmap_bin_files: true
  tokenizer:
    tokenizer_type: QwenTokenizerFS  # No legacy_tokenizer in Stage 10
    tokenizer_path: <TOKENIZER_PATH>
    vocab_size: 151851
    make_vocab_size_divisible_by: 64
```

## Running Tests

Launch a single config:
```bash
cd FlagScale
python -m flagscale.launcher.runner \
  --config-path ../run_configs/stage{N}_te_fl_prefer-{impl}__attention_backend-{backend} \
  --config-name train
```

Launch all configs for a stage:
```bash
# Create a runner script
for config in FlagScale/run_configs/stage{N}_*/train.yaml; do
  config_dir=$(dirname $config)
  config_name=$(basename $config_dir)
  echo "=== [$config_name] START $(date) ===" >> stage{N}_runs/logs/${config_name}.log
  cd FlagScale && python -m flagscale.launcher.runner \
    --config-path ../$config_dir \
    --config-name train \
    >> ../stage{N}_runs/logs/${config_name}.log 2>&1
  echo "=== [$config_name] EXIT_CODE=$? $(date) ===" >> stage{N}_runs/logs/${config_name}.log
  cd ..
done
```

## Best Practices

1. **Always use stage10 format for new stages** - includes eval_interval and removes legacy_tokenizer
2. **Test one config before batch runs** - verify environment and paths work
3. **Archive before cleanup** - preserve logs and summaries
4. **Use descriptive stage numbers** - increment sequentially (stage11, stage12, etc.)
5. **Check disk space** - each stage run can be 10-50GB depending on checkpoints

## Common Issues

**Issue**: Config not found
- Check `FlagScale/run_configs/` path exists
- Verify YAML syntax with `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`

**Issue**: CUDA OOM during parallel runs
- Run configs sequentially instead of parallel
- Reduce `micro_batch_size` or `global_batch_size`

**Issue**: Checkpoint directory conflicts
- Ensure each config has unique `exp_dir`
- Clean old checkpoints before rerunning

## Output Format

When creating a unified stage, report:
```
Created Stage {N} with 12 configurations:
✓ Qwen3-32b: flagos × [flash, fused, unfused]
✓ Qwen3-32b: reference × [flash, fused, unfused]
✓ Qwen3-32b: vendor × [flash, fused, unfused]
✓ DeepSeek-V3: [flagos, reference, vendor] × unfused

Configs: FlagScale/run_configs/stage{N}_*/
Run dir: stage{N}_runs/

To launch all tests:
  bash scripts/run_stage{N}.sh
```

When running batch tests, show progress and final summary:
```
Running Stage {N} tests...
[1/12] qwen3_flagos_flash... PASS (23.4s)
[2/12] qwen3_flagos_fused... PASS (24.1s)
...
[12/12] deepseek_v3_vendor_unfused... PASS (45.2s)

Summary: 12/12 PASS (0 FAIL)
Results: stage{N}_runs/logs/summary.txt
```
