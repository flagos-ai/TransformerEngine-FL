# TransformerEngine-FL v2.14 → v2.17 升级结论记录

本文件记录每个 Stage 完成后的结论、关键发现和决策。

---

## Stage 1: Repo Setup & Branch Preparation ✅

**完成时间**: 2025-01

**结论**:
- 三分支结构就绪
  - `base` = upstream v2.14 tag (`f031cf87`)
  - `dev` = upstream v2.17 tag (`2e559f06`)
  - `main` = fork 最新 (`dea7cd6c`)
- upstream remote 已添加: `https://github.com/NVIDIA/TransformerEngine.git`
- 增量规模确认: 239 commits, 482 files, +83,356/-20,066 lines

---

## Stage 2: Identify Plugin Changes ✅

**完成时间**: 2025-01

### Fork 专有变更总览

- **base..main**: 64 commits, 233 files, +44,026/-634 lines
- Fork 变更全部是增量式（无删除/重命名 upstream 文件）

### 文件分类

| 区域 | 文件数 | 类型 | 冲突风险 |
|------|--------|------|----------|
| `transformer_engine/plugin/` | ~75 | 全部新增 | 🟢 低 |
| `transformer_engine/pytorch/` | 43 修改 | patch | 🔴 高 |
| `transformer_engine/debug/` | 4 | 修改 | 🟡 中 |
| `transformer_engine/__init__.py` + `common/` | 2 | 修改 | 🟡 中 |
| `tests/` | 47 | 全部新增 | 🟢 低 |
| `.github/` | 31 | 新增+修改 | 🟢 低 |
| `qa/` | 13 | 新增+修改 | 🟡 中 |
| `build_tools/` + `setup.py` | 3 | 修改 | 🟡 中 |
| `3rdparty/` submodules | 3 | 版本更新 | 🟡 中 |
| `.gitignore` | 1 | 修改 | 🟢 低 |

### pytorch/ 双向冲突预检

| 类型 | 文件数 | 处理方式 |
|------|--------|----------|
| 必然冲突（双方修改） | 38 | 手动解冲突 |
| fork-only 修改（upstream未动） | 6 | 自动保留 |
| upstream-only 修改（fork未动） | 60 | 自动接受 |

**38个冲突文件核心模式**: fork 加了 plugin dispatch patch，upstream 改了底层实现，通常不在同一行但 git 会因上下文偏移报冲突。

### 3rdparty Submodule 版本决策

| submodule | base | fork | upstream v2.17 | 决策 |
|-----------|------|------|----------------|------|
| cudnn-frontend | `7b9b711c` | `1d6f6d9b` | `e46d7082` | 取 upstream v2.17 |
| cutlass | `57e3cfb4` | `e64a9136` | `57e3cfb4` | 保留 fork |
| googletest | `f8d7d77c` | `a0f06a70` | `f8d7d77c` | 保留 fork |
| nccl | N/A | N/A | `a6b5de08` | 接受 upstream 新增 |

### 关键发现

1. Fork pytorch/ 修改本质是 plugin dispatch patch，与 upstream 实现演进通常不冲突同一行
2. 6 个 fork-only 文件完全不会冲突
3. upstream v2.17 新增了 `3rdparty/nccl` submodule

---

## Stage 3-10

*待完成*
