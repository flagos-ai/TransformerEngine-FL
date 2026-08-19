# Qwen3.5 Ascend NPU 优化交接说明

## 1. 交付目标

本次交付面向 FlagScale Qwen3.5-35B-A3B、64K 序列训练，提供一套已经跑通基础功能并完成有限数值对比的 Ascend NPU 优化基线，供后续继续进行算子和端到端性能优化。

当前结果不能视为严格精度验证：尚未完成完整模型、真实训练配置、多步 loss 收敛和长时间训练稳定性验收。

本次不包含量化和 FP8 优化。

## 2. 交付提交

本次改动横跨三个仓库，需要配套使用：

| 仓库 | 分支 | 提交 | 主要内容 |
| --- | --- | --- | --- |
| TransformerEngine-FL | `main` | `c5976768` | MoE permutation、chunk-sort 及必要的 NPU 算子接入 |
| Megatron-LM-FL | `ascend_qw` | `6e7255559` | Gated DeltaNet 的 Ascend 加速及插件化接入 |
| FlagScale | `main` | `c68ca655` | 64K smoke 配置、启动脚本和 `empty_cache` 热路径优化 |

提交中没有包含训练日志、profile、备份文件和一次性调试脚本。

## 3. 主要优化点

### 3.1 MoE token permutation

- 为 Ascend 接入 mask-map permute/unpermute 的 NPU 实现。
- 接入 TransformerEngineNPU 的 chunk-sort 实现，支持 64K/top-8 场景中约 13 万 token 的实际输入规模。
- 支持moe_permute_fusion=True`，避免回到包含 AICPU `argsort` 的非融合分发路径。

### 3.2 Gated DeltaNet

- 将 Q/K L2Norm 和 GDN recurrence 切换到 fla_npu 优化实现。
- NPU 相关代码放入 Megatron-LM-FL 的 Ascend plugin，减少对模型核心代码的侵入。

## 4. 使用方法

### 4.1 环境要求

- Ascend NPU 与匹配版本的 CANN。
- 匹配版本的 PyTorch、torch_npu。
- TransformerEngineNPU，可通过 `import transformer_engine_npu` 导入。
- 同时使用本次交付的 TransformerEngine-FL、Megatron-LM-FL 和 FlagScale 提交。

源码运行时建议设置：

```bash
export PYTHONPATH=/workspace/TransformerEngine-FL:/workspace/TransformerEngineNPU:${PYTHONPATH}
``
```

## 5. 当前验证情况

### 5.1 TransformerEngine-FL

```bash
cd /workspace/TransformerEngine-FL
pytest -q \
  tests/plugin/backend/npu/test_backend_npu.py \
  tests/plugin/plugin/test_manager.py \
  transformer_engine/plugin/tests/test_npu_moe_mask_permutation.py
```

结果：`50 passed`。这些是针对当前接口和有限输入的单元测试，不代表完整训练精度已经通过验收。

覆盖内容包括：

- mask-map permute/unpermute。
- chunk-sort 前向和反向。
- 131181-token 输入。
- BF16 activation 和 FP32 probability 的有限样例数值对比。

### 5.2 Megatron-LM-FL

```bash
cd /workspace/Megatron-LM-FL
pytest -q tests/test_gdn_plugin_dispatch.py
```

结果：`2 passed`，覆盖 Ascend plugin 选择以及 Q/K L2Norm 的有限 BF16 前向、反向数值对比；没有覆盖完整 GDN recurrence 的严格数值验收。

### 5.3 FlagScale

64K 配置已经在 10.55.0.164 的 `test-lyz-train` 容器中完成 `dryrun`，确认：

```text
seq_length=65536
moe_router_fusion=false
moe_permute_fusion=true
```

`dryrun` 只确认配置能够正确组合，不验证训练数值。

### 5.4 尚未完成的精度验收

- 完整 Qwen3.5-35B-A3B 配置的 NPU/基准实现逐步 loss 对比。
- 多个随机种子和多个训练 step 的 loss、梯度及参数更新对比。
- GDN recurrence 在真实训练 shape 下的完整前向、反向对比。
- 长时间训练的收敛性、NaN/Inf 和稳定性检查。
- checkpoint 保存、恢复后的连续性验证。

## 6. 当前性能基线

测试环境：10.55.0.164，`test-lyz-train` 容器，16 NPU，Qwen3.5 64K smoke 配置。

|  | 稳定迭代时间 | 吞吐 |
| --- | ---: | ---: |
 | 2387.9 ms/iter | 约 66.1 TFLOP/s/GPU |

当前观测比历史结果慢约 1.88%，但历史结果并非本次在相同代码和环境下重新运行得到的严格 A/B，因此只能作为参考，不能直接归因到单个优化。
