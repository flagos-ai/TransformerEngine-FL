# TransformerEngine-FL Upstream Sync Plan: v2.14 → v2.17

## 概览

| 项目 | 值 |
|------|------|
| Fork 当前基线 | upstream `v2.14` tag (commit `f031cf87`) — PR #62 已合入 main |
| 目标 | upstream `v2.17` tag (commit `2e559f06`) |
| 增量规模 | 239 commits, 482 files, +83,356 / -20,066 lines |
| csrc pybind 变化 | 34 行 `.def()` 增删（约 15-17 个 API 新增/修改） |
| main 上 v2.14 之后 fork 新增 | ~30 commits（vendor backends、CICD、FlagOS op 等） |
| 参考 PR | [#62](https://github.com/flagos-ai/TransformerEngine-FL/pull/62) (v2.9→v2.14 同步) |
| Skill 参考 | PR #67 `skills/te-fl-upstream-sync/` |

---

## Stage 1: Repo Setup & Branch Preparation

**目标**: 创建三分支结构（dev/base/main），为后续 merge 做准备。

**命令**:
```bash
cd /share/project/zhaoyingli/flagos/TransformerEngine-FL

# 添加 upstream remote（如未添加）
git remote add upstream https://github.com/NVIDIA/TransformerEngine.git 2>/dev/null || true
git fetch upstream --tags

# 创建 dev 分支 = v2.17 tag
git checkout -b dev v2.17

# 创建 base 分支 = v2.14 tag（PR #62 同步点）
git checkout -b base v2.14

# 回到 main
git checkout main
```

**验证**:
- `dev` → commit `2e559f06`
- `base` → commit `f031cf87`
- `main` → fork 最新

**产出**: SYNC_POINT.md

---

## Stage 2: Identify Plugin Changes

**目标**: 记录 fork 相对于 base 的所有专有变更，作为 Stage 3 解冲突的参考。

**命令**:
```bash
# Plugin 目录（全部是新增文件）
git diff --name-status base..main -- 'transformer_engine/plugin/'

# CUDA patches / TE_DEVICE_TYPE
git diff --name-status base..main -- 'transformer_engine/__init__.py'

# Build 系统
git diff base..main -- setup.py CMakeLists.txt pyproject.toml

# Python layer patches
git diff --name-status base..main -- 'transformer_engine/pytorch/'

# 生成完整 diff 留档
git diff base..main > /tmp/plugin_changes_full.diff
```

**产出**: `PLUGIN_CHANGES.md`（结构化记录新增文件、修改文件、build 变更）

---

## Stage 3: Merge & Conflict Resolution

**目标**: 将 upstream v2.17 合入 fork main，按优先级解决冲突。

**冲突分级**:

| 优先级 | 范围 | 策略 |
|--------|------|------|
| P0 | `transformer_engine/pytorch/` 中 plugin 相关调用 | 保留 fork plugin dispatch, 接受 upstream 实现演进 |
| P1 | `setup.py`, `build_tools/`, `3rdparty/` | 保留 plugin 编译目标, 合入 upstream 依赖/版本更新 |
| P2 | `.github/`, `tests/`, `qa/`, `docs/`, `README` | 以 upstream 为主, 补充 fork CI 扩展 |

**P0 解冲突原则**:
1. 读取 main 版本，保留新增的 fork 内容（plugin dispatch 调用、TE_DEVICE_TYPE 替换）
2. 接受 upstream 的功能新增和重构
3. 检查安全/关键 bug fix

---

### Stage 3.1: 创建 merge 分支并执行 merge ✅

```bash
git checkout main
git checkout -b merge/dev-to-main-$(date +%Y%m%d)
git merge dev --no-ff -m "merge(dev): integrate upstream v2.17"
```

**结果**: 分支 `merge/dev-to-main-20260807`，冲突总数 **228 个文件**:
- 173 个 fork 没改过 → Stage 3.2 自动解决
- 55 个 fork 改过 → 手动解（Stage 3.3-3.10）

---

### Stage 3.2: 自动解冲突（fork 未修改的 173 个文件）

**分类**:

| 分类 | 文件数 | 说明 | Stage 4 影响 |
|------|--------|------|-------------|
| Plugin 相关 | 81 | csrc pybind(12) + cpp_extensions(1) + ops/(6) + tensor/(6) + common/(56) | ⚠️ 需作为 Stage 4 API Sync 输入 |
| 非 Plugin 相关 | 92 | jax/(28) + tests/(37) + docs/(5) + build_tools/wheel(4) + qa/(5) + examples/(4) + 其他 | ✅ 无需额外处理 |

**Plugin 相关 81 文件细分**:
- A. csrc pybind (12): pybind.cpp, extensions/*.cpp, quantizer.cpp → 直接影响 plugin ops.py
- B. cpp_extensions (1): fused_attn.py → FA Python wrapper 签名变化
- C. ops/ (6): __init__, _common, basic/__init__, fused/__init__, backward_activation_bias, fuser
- D. tensor/ (6): _quantization_helpers, storage/*_storage.py, utils.py
- E. common/ (56): CUDA kernels (cast, fused_attn, fused_router, hadamard, gemm, normalization, multi_tensor, swizzle, transpose, triton)

**命令**:
```bash
for file in $(git diff --name-only --diff-filter=U); do
  if ! git diff base..main -- "$file" | grep -q '^[+-]'; then
    git checkout --theirs "$file" && git add "$file"
    echo "AUTO-RESOLVED (theirs): $file"
  fi
done

# Stage 4 准备: 导出 plugin 相关文件的 upstream diff
git diff base..dev -- transformer_engine/pytorch/csrc/ > /tmp/stage4_csrc_diff.diff
git diff base..dev -- transformer_engine/common/ > /tmp/stage4_common_diff.diff
```

---

### Stage 3.3: 解冲突 P2 — 低风险文件

涉及: `.github/`, `docs/`, `README.rst`, `.gitignore`, `qa/` 脚本
策略: 以 upstream 为主，保留 fork CI 扩展

```bash
for file in $(git diff --name-only --diff-filter=U | grep -E '^\.(github|gitignore)|^docs/|^README|^qa/'); do
  git checkout --theirs "$file" && git add "$file"
  echo "P2 RESOLVED (theirs): $file"
done
```

---

### Stage 3.4: 解冲突 P1 — Build 系统

涉及: `setup.py`, `build_tools/pytorch.py`, `build_tools/utils.py`, `3rdparty/`, `.gitmodules`

3rdparty submodule 决策:
- cudnn-frontend → 取 upstream v2.17 (e46d7082)
- cutlass → 保留 fork (e64a9136)
- googletest → 保留 fork (a0f06a70)
- nccl → 接受 upstream 新增 (a6b5de08)
- .gitmodules → 合并（保留原有 + 新增 nccl 条目）

setup.py / build_tools: 保留 fork plugin 编译目标，合入 upstream 依赖更新

---

### Stage 3.5: 解冲突 P0-A — __init__.py / common / debug (6 files)

- `transformer_engine/__init__.py`
- `transformer_engine/common/__init__.py`
- `transformer_engine/debug/features/fake_quant.py`
- `transformer_engine/debug/features/log_fp8_tensor_stats.py`
- `transformer_engine/debug/features/per_tensor_scaling.py`
- `transformer_engine/debug/features/utils/stats_buffer.py`

策略: 保留 TE_DEVICE_TYPE 定义和 plugin import，接受 upstream 新增功能代码。

---

### Stage 3.6: 解冲突 P0-B — pytorch/attention/ (5 files)

- `attention/dot_product_attention/backends.py`
- `attention/dot_product_attention/context_parallel.py`
- `attention/dot_product_attention/dot_product_attention.py`
- `attention/dot_product_attention/utils.py`
- `attention/multi_head_attention.py`

策略: 保留 fork `plugin.ops.get_attention_backend()` dispatch + FlashAttentionBase，接受 upstream FA3/新参数演进。

---

### Stage 3.7: 解冲突 P0-C — pytorch/module/ (5 files)

- `module/base.py`
- `module/grouped_linear.py`
- `module/layernorm_linear.py`
- `module/layernorm_mlp.py`
- `module/linear.py`

策略: 保留 fork `plugin.ops.xxx()` gemm/normalization dispatch，接受 upstream quantization/precision 演进。

---

### Stage 3.8: 解冲突 P0-D — pytorch/ops/ (11 files)

- `ops/basic/activation.py`, `basic_linear.py`, `bias.py`, `grouped_linear.py`, `swiglu.py`
- `ops/fused/forward_grouped_mlp.py`, `forward_linear_bias_activation.py`, `forward_linear_bias_add.py`, `forward_linear_scale_add.py`, `userbuffers_backward_linear.py`, `userbuffers_forward_linear.py`

策略: 保留 fork plugin op dispatch，接受 upstream 新增 quantization path / fused op 逻辑。

---

### Stage 3.9: 解冲突 P0-E — pytorch/tensor/ (6 files)

- `tensor/float8_blockwise_tensor.py`, `float8_tensor.py`, `grouped_tensor.py`, `mxfp8_tensor.py`, `nvfp4_tensor.py`
- `tensor/storage/grouped_tensor_storage.py`

策略: 保留 fork TE_DEVICE_TYPE 替换，接受 upstream tensor 实现演进。

---

### Stage 3.10: 解冲突 P0-F — pytorch/ 其他文件 (11 files)

- `pytorch/__init__.py`, `cpp_extensions/gemm.py`, `cpu_offload.py`, `distributed.py`
- `optimizers/fused_adam.py`, `permutation.py`, `quantization.py`, `setup.py`
- `transformer.py`, `triton/permutation.py`, `utils.py`

策略: 逐文件检查 fork patch 内容，保留 plugin dispatch + TE_DEVICE_TYPE，接受 upstream 逻辑。

---

### Stage 3.11: 提交 merge commit

```bash
# 验证无残留冲突标记
grep -rn '<<<<<<<\|=======\|>>>>>>>' transformer_engine/ tests/ qa/ setup.py build_tools/ && echo "CONFLICT MARKERS FOUND!" || echo "CLEAN"

# pre-commit
pre-commit run --all-files
git add -A

# 提交
git commit --no-edit
git log --oneline -3
```

---

## Stage 4: Plugin API Sync

**目标**: 让 plugin 层 1:1 覆盖 upstream csrc pybind 接口变化。

### 4.1 Diff csrc pybind

```bash
git diff base..dev -- transformer_engine/pytorch/csrc/ > /tmp/csrc_diff.diff

# 提取 ADDED APIs
grep -E '^\+.*\.def\("' /tmp/csrc_diff.diff | grep -v '^\+\+\+' | \
  sed 's/.*\.def("\([^"]*\)".*/\1/' | sort -u > /tmp/added_apis.txt

# 提取 REMOVED APIs
grep -E '^\-.*\.def\("' /tmp/csrc_diff.diff | grep -v '^\-\-\-' | \
  sed 's/.*\.def("\([^"]*\)".*/\1/' | sort -u > /tmp/removed_apis.txt

# 分类
comm -23 /tmp/added_apis.txt /tmp/removed_apis.txt  # 纯新增
comm -13 /tmp/added_apis.txt /tmp/removed_apis.txt  # 纯删除
comm -12 /tmp/added_apis.txt /tmp/removed_apis.txt  # 修改（签名变化）
```

### 4.2 更新文件列表

| 文件 | 操作 |
|------|------|
| `plugin/core/ops.py` | 新增/修改抽象方法 |
| `plugin/core/backends/vendor/cuda/cuda.py` | CUDA 参考实现（先改这个） |
| `plugin/core/backends/vendor/cuda/register_ops.py` | 新增 OpImpl 注册 |
| 其余 5 个 vendor（enflame/hygon/iluvatar/metax/musa） | 复制 CUDA 签名 |
| 各 vendor `register_ops.py` | 同步 OpImpl |
| `FlashAttentionBase` + 各 vendor `flash_attention.py` | 同步 forward 签名 |

### 4.3 class-object 参数检测

检查 `AttentionParams` 等 dataclass 是否新增字段:
```bash
diff <(git show base:transformer_engine/pytorch/attention/dot_product_attention/utils.py | \
  sed -n '/^class AttentionParams/,/^class \|^def /p' | grep -E "^\s+\w+\s*:") \
  <(git show dev:transformer_engine/pytorch/attention/dot_product_attention/utils.py | \
  sed -n '/^class AttentionParams/,/^class \|^def /p' | grep -E "^\s+\w+\s*:")
```

### 4.4 Full-surface pybind 覆盖审计

```bash
# 提取所有 pybind export
grep -oP 'm\.def\("\K[^"]+' transformer_engine/pytorch/csrc/extensions/pybind.cpp | sort -u > /tmp/all_exports.txt

# 对比 plugin ops.py
grep -oP 'def \K\w+' transformer_engine/plugin/core/ops.py | grep -v '^_' | sort -u > /tmp/plugin_ops.txt

# 找缺失
comm -23 /tmp/all_exports.txt /tmp/plugin_ops.txt
```

### 4.5 关键规则（PR #62 经验）

- 所有 vendor backend **必须用显式参数签名**，禁止 `*args/**kwargs`
- enum 参数需要 `tex.DType(int(dtype))` 转换
- quantizer 对象需要 `quantizer.dtype = tex.DType(int(qdtype))` 归一化
- 新增 op 必须同时更新 `register_ops.py`

---

## Stage 5: Patch CUDA Hardcoding

**目标**: 将 upstream v2.14→v2.17 新引入的 `"cuda"` 硬编码替换为 `TE_DEVICE_TYPE`。

**扫描**:
```bash
git diff base..dev -- transformer_engine/pytorch/ ':(exclude)transformer_engine/pytorch/csrc/' \
  | grep '^+' | grep -v '^+++' \
  | grep -E 'device.*"cuda"|torch\.device\("cuda"\)|get_autocast_dtype.*"cuda"|\.device\.type.*==.*"cuda"' \
  > /tmp/cuda_string_candidates.txt
```

**替换规则**:

| 模式 | 替换为 |
|------|--------|
| `device="cuda"` | `device=TE_DEVICE_TYPE` |
| `torch.device("cuda")` | `torch.device(TE_DEVICE_TYPE)` |
| `torch.get_autocast_dtype("cuda")` | `torch.get_autocast_dtype(TE_DEVICE_TYPE)` |
| `.device.type == "cuda"` | `.device.type == TE_DEVICE_TYPE` |

**不动**:
- `torch.cuda.*` API 调用（由 vendor patches.py 运行时处理）
- `torch.cuda.CUDAGraph`、`torch.version.cuda`
- 注释/docstring 中的 `"cuda"`
- CUDA-specific guard 中的检测逻辑

**每个修改的文件需添加 import**:
```python
from transformer_engine import TE_DEVICE_TYPE
```

---

## Stage 6: Detect & Fix Stale References

**目标**: 检查 fork 专有代码中引用被 upstream 重命名/移动的符号。

### 6.1 找出 upstream 重命名/删除

```bash
# 被删除的函数/类定义
git diff base..dev -- '*.py' | grep -E '^\-(def |class )' | \
  sed 's/^-//;s/(.*//;s/def //;s/class //;s/://;s/ //g' | sort -u > /tmp/removed_symbols.txt

# 被新增的
git diff base..dev -- '*.py' | grep -E '^\+(def |class )' | \
  sed 's/^+//;s/(.*//;s/def //;s/class //;s/://;s/ //g' | sort -u > /tmp/added_symbols.txt

# 真正消失的（被删且未重新添加）
comm -23 /tmp/removed_symbols.txt /tmp/added_symbols.txt > /tmp/gone_symbols.txt

# 文件重命名/删除
git diff --diff-filter=R --name-status -M base..dev > /tmp/renamed_files.txt
git diff --diff-filter=D --name-only base..dev > /tmp/deleted_files.txt
```

### 6.2 在 fork 专有代码中搜索

```bash
# fork 专有文件 = HEAD 有但 dev 没有的变更
git diff --name-only dev..HEAD -- '*.py' > /tmp/fork_files.txt

# 逐个搜索消失的符号
while IFS= read -r symbol; do
  MATCHES=$(grep -rn "$symbol" $(cat /tmp/fork_files.txt) 2>/dev/null)
  [ -n "$MATCHES" ] && echo "⚠️ STALE: $symbol" && echo "$MATCHES"
done < /tmp/gone_symbols.txt
```

### 6.3 修复

- 找到新名称: `git diff base..dev -- <file> | grep -A5 -B5 "old_name"`
- 找到新路径: `git ls-tree -r --name-only dev | grep "<basename>"`

---

## Stage 7: Build & Import Verification

**目标**: 确认合并后的代码能编译和 import。

```bash
git submodule update --init --recursive
pip install --no-build-isolation -e . 2>&1 | tee build.log
python -c "from transformer_engine import pytorch; print('OK')"
```

**预期输出**:
```
[CUDA] Successfully loaded CUDA libs
[TE-FL manager.py INFO] OpManager initialized: N ops with M implementations
[TE-FL manager.py INFO] Registered impl_ids: ['default.flagos', 'reference.torch', 'vendor.cuda']
```

**失败处理**: 根据 traceback 回溯到 Stage 3-6 修复。

---

## Stage 8: Unit & Integration Tests

**分层执行**:

| Level | 命令 | 验证内容 |
|-------|------|----------|
| L1 | `pytest transformer_engine/plugin/tests/ -k plugin -v` | Plugin 注册/dispatch |
| L2 | `pytest tests/pytorch/ -v` | 上游集成测试 |
| L2.5a | `TE_PATH=$(pwd) bash qa/L0_pytorch_debug_unittest/test.sh` | Debug 单测 |
| L2.5b | `TE_PATH=$(pwd) bash qa/L0_pytorch_unittest/test.sh` | PyTorch 单测 |
| L2.5c | `TE_PATH=$(pwd) bash qa/L1_pytorch_distributed_unittest/test.sh` | 分布式测试 |
| L2.6 | `python transformer_engine/plugin/tests/run_all_tests.py` | Plugin 全量测试 |
| L3 | `python tests/pytorch/test_sanity.py` | E2E sanity |

**前置检查**: CI 脚本引用的测试文件是否因 upstream rename 而失效:
```bash
grep -oP '(?<=\$TE_PATH/)tests/pytorch/[^\s"]+\.py' qa/L0_pytorch_unittest/test.sh | \
  sort -u | while read f; do [ ! -f "$f" ] && echo "MISSING: $f"; done
```

**PR #62 踩过的坑**:
- `test_float8tensor.py` → `test_quantized_tensor.py`（upstream rename）
- OP API 签名不匹配（如 `fused_topk_with_score_function_bwd` 缺参数）
- MetaX 平台特有跳过项需维护

---

## Stage 5: Patch CUDA Hardcoding

**目标**: 将 upstream v2.14→v2.17 新引入的 `"cuda"` 硬编码替换为 `TE_DEVICE_TYPE`。

**扫描**:
```bash
git diff base..dev -- transformer_engine/pytorch/ ':(exclude)transformer_engine/pytorch/csrc/' \
  | grep '^+' | grep -v '^+++' \
  | grep -E 'device.*"cuda"|torch\.device\("cuda"\)|get_autocast_dtype.*"cuda"|\.device\.type.*==.*"cuda"' \
  > /tmp/cuda_string_candidates.txt
```

**替换规则**:

| 模式 | 替换为 |
|------|--------|
| `device="cuda"` | `device=TE_DEVICE_TYPE` |
| `torch.device("cuda")` | `torch.device(TE_DEVICE_TYPE)` |
| `torch.get_autocast_dtype("cuda")` | `torch.get_autocast_dtype(TE_DEVICE_TYPE)` |
| `.device.type == "cuda"` | `.device.type == TE_DEVICE_TYPE` |

**不动**:
- `torch.cuda.*` API 调用（由 vendor patches.py 运行时处理）
- `torch.cuda.CUDAGraph`、`torch.version.cuda`
- 注释/docstring 中的 `"cuda"`
- CUDA-specific guard 中的检测逻辑

**每个修改的文件需添加 import**:
```python
from transformer_engine import TE_DEVICE_TYPE
```

---

## Stage 6: Detect & Fix Stale References

**目标**: 检查 fork 专有代码中引用被 upstream 重命名/移动的符号。

### 6.1 找出 upstream 重命名/删除

```bash
# 被删除的函数/类定义
git diff base..dev -- '*.py' | grep -E '^\-(def |class )' | \
  sed 's/^-//;s/(.*//;s/def //;s/class //;s/://;s/ //g' | sort -u > /tmp/removed_symbols.txt

# 被新增的
git diff base..dev -- '*.py' | grep -E '^\+(def |class )' | \
  sed 's/^+//;s/(.*//;s/def //;s/class //;s/://;s/ //g' | sort -u > /tmp/added_symbols.txt

# 真正消失的（被删且未重新添加）
comm -23 /tmp/removed_symbols.txt /tmp/added_symbols.txt > /tmp/gone_symbols.txt

# 文件重命名/删除
git diff --diff-filter=R --name-status -M base..dev > /tmp/renamed_files.txt
git diff --diff-filter=D --name-only base..dev > /tmp/deleted_files.txt
```

### 6.2 在 fork 专有代码中搜索

```bash
# fork 专有文件 = HEAD 有但 dev 没有的变更
git diff --name-only dev..HEAD -- '*.py' > /tmp/fork_files.txt

# 逐个搜索消失的符号
while IFS= read -r symbol; do
  MATCHES=$(grep -rn "$symbol" $(cat /tmp/fork_files.txt) 2>/dev/null)
  [ -n "$MATCHES" ] && echo "⚠️ STALE: $symbol" && echo "$MATCHES"
done < /tmp/gone_symbols.txt
```

### 6.3 修复

- 找到新名称: `git diff base..dev -- <file> | grep -A5 -B5 "old_name"`
- 找到新路径: `git ls-tree -r --name-only dev | grep "<basename>"`

---

## Stage 7: Build & Import Verification

**目标**: 确认合并后的代码能编译和 import。

```bash
git submodule update --init --recursive
pip install --no-build-isolation -e . 2>&1 | tee build.log
python -c "from transformer_engine import pytorch; print('OK')"
```

**预期输出**:
```
[CUDA] Successfully loaded CUDA libs
[TE-FL manager.py INFO] OpManager initialized: N ops with M implementations
[TE-FL manager.py INFO] Registered impl_ids: ['default.flagos', 'reference.torch', 'vendor.cuda']
```

**失败处理**: 根据 traceback 回溯到 Stage 3-6 修复。

---

## Stage 8: Unit & Integration Tests

**分层执行**:

| Level | 命令 | 验证内容 |
|-------|------|----------|
| L1 | `pytest transformer_engine/plugin/tests/ -k plugin -v` | Plugin 注册/dispatch |
| L2 | `pytest tests/pytorch/ -v` | 上游集成测试 |
| L2.5a | `TE_PATH=$(pwd) bash qa/L0_pytorch_debug_unittest/test.sh` | Debug 单测 |
| L2.5b | `TE_PATH=$(pwd) bash qa/L0_pytorch_unittest/test.sh` | PyTorch 单测 |
| L2.5c | `TE_PATH=$(pwd) bash qa/L1_pytorch_distributed_unittest/test.sh` | 分布式测试 |
| L2.6 | `python transformer_engine/plugin/tests/run_all_tests.py` | Plugin 全量测试 |
| L3 | `python tests/pytorch/test_sanity.py` | E2E sanity |

**前置检查**: CI 脚本引用的测试文件是否因 upstream rename 而失效:
```bash
grep -oP '(?<=\$TE_PATH/)tests/pytorch/[^\s"]+\.py' qa/L0_pytorch_unittest/test.sh | \
  sort -u | while read f; do [ ! -f "$f" ] && echo "MISSING: $f"; done
```

**PR #62 踩过的坑**:
- `test_float8tensor.py` → `test_quantized_tensor.py`（upstream rename）
- OP API 签名不匹配（如 `fused_topk_with_score_function_bwd` 缺参数）
- MetaX 平台特有跳过项需维护

---

## Stage 9: Merge to Main (Tree Replacement)

**目标**: 将完成验证的 merge 分支通过 tree replacement 策略合入 main，提交 PR。

**为什么用 tree replacement 而不是普通 merge**:
`-X theirs` 只解决冲突，非冲突变更仍会合并两侧。当 fork 和 upstream 独立添加了相同 patch 时，
git 会保留两份副本导致重复代码。Tree replacement 完全避免此问题。

**命令**:
```bash
git checkout main && git pull origin main
git checkout -b merge-to-main-$(date +%Y%m%d)

# Tree replacement: 记录两个 parent 但用 merge 分支的 tree
git merge -s ours merge/dev-to-main-YYYYMMDD --no-edit
git read-tree -m -u merge/dev-to-main-YYYYMMDD

# pre-commit 格式化
pre-commit run --all-files
git add -A && git commit --amend --no-edit
```

**清理中间文件**:
```bash
for f in SYNC_POINT.md MERGE_RECORD.md UPSTREAM_SYNC.md; do
  [ -f "$f" ] && git rm "$f"
done
git diff --cached --quiet || git commit -m "chore: remove intermediate sync record files"
```

**验证**:
```bash
git diff merge/dev-to-main-YYYYMMDD HEAD --stat
git log --oneline --graph -5
pip install --no-build-isolation -e .
python -c "import transformer_engine; print('OK')"
```

---

## Stage 10: FlagScale E2E Training Validation

**目标**: 在 FlagScale 真实训练场景下验证合并后的 TE-FL。

**测试矩阵**（复用 PR #62 标准）:

### Qwen3-32B（16 layers, 20 iters, 1 node × 8 GPUs）

| Config | 预期 |
|--------|------|
| vendor-flash / vendor-fused / vendor-unfused | PASS |
| flagos-flash / flagos-unfused | PASS |
| reference-flash / reference-unfused | PASS |

### DeepSeek-V3 16BA3B（18 layers + 1 mtp, 20 iters）

| Config | 预期 |
|--------|------|
| vendor-unfused / flagos-unfused / reference-unfused | PASS |

**成功标准**: 至少一个组合跑完 20 步无错误，loss 下降。

---

## 风险与注意事项

| 风险 | 应对 |
|------|------|
| NVFP4/MXFP8 新 op 多 | Stage 4 重点关注 |
| FA3 参数变化 | Stage 4 FlashAttentionBase 同步 |
| 新 vendor stale refs | Stage 6 重点扫描 |
| Build system 演进 | Stage 3 P1 仔细处理 |

## 时间估算

| 阶段 | 预计耗时 |
|------|----------|
| Stage 1-2 | 0.5 天 |
| Stage 3 (merge) | 2-3 天 |
| Stage 4 (Plugin API) | 2-3 天 |
| Stage 5-6 | 1 天 |
| Stage 7-8 (build+tests) | 1-2 天 |
| Stage 9-10 | 1 天 |
| **总计** | **7-10 天** |

## Rollback

任何时候出问题: `git revert -m 1 <merge-commit-sha>`
