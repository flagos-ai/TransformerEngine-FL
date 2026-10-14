# TransformerEngine-FL Upgrade Skills

这组 skills 用于维护 TransformerEngine-FL 与任意上游 TransformerEngine release 之间的升级流程。它们不绑定具体版本；每次运行通过 `base`、`fork`、`target` 三个 ref 参数确定范围。

## 用途

这套流程覆盖：

- FL fork 设计基线和差异分类；
- plugin manager、registry、policy、discovery 和各硬件 backend；
- `transformer_engine_torch` 动态 alias、enum、callable public contract；
- PyTorch attention/device 等侵入式 runtime patch；
- C++/CUDA 构建、wheel、CMake 和 third-party submodule；
- GitHub Actions、硬件 CI、QA 和测试矩阵；
- 升级后的证据汇总、回滚和交付。

## 目录

| Skill | 作用 |
|---|---|
| `te-upgrade-orchestrator` | 按 gate 串联完整升级流程 |
| `te-classify-fork-delta` | 生成 fork 差异清单和设计基线 |
| `te-integrate-upstream-conflicts` | 分类和解决 upstream/fork 冲突 |
| `te-audit-plugin-api` | 审计 plugin/native/Python API 和动态 alias |
| `te-preserve-runtime-patches` | 保护 runtime 侵入式修改及跨模块调用链 |
| `te-integrate-build-submodules` | 审计构建、打包、CUDA 和 submodule |
| `te-audit-cicd` | 审计 CI/CD、硬件矩阵和 QA 入口 |
| `te-run-upgrade-test-matrix` | 执行 import、GPU、backend 和回归测试 |
| `te-finalize-upstream-upgrade` | 汇总证据并生成交付决策 |

## 推荐用法

在仓库根目录开始，先创建工作分支，不要直接在 `main` 上操作：

```bash
git switch -c chore/te-upgrade-<date>
```

然后按以下顺序使用：

```text
1. te-classify-fork-delta
2. te-integrate-upstream-conflicts
3. te-audit-plugin-api
4. te-preserve-runtime-patches
5. te-integrate-build-submodules
6. te-audit-cicd
7. te-run-upgrade-test-matrix
8. te-finalize-upstream-upgrade
```

也可以直接使用 `te-upgrade-orchestrator`，它会检查每个阶段的 artifact、ref 一致性、P0 决策和用户审批。

## 参数和 artifact

典型参数：

```text
base   = 上游共同基线 ref
fork   = 当前 FL 分支/ref
target = 要升级到的上游 ref
```

所有中间结果放在：

```text
/share/project/zhaoyingli/flagos/temp/<run-name>/
```

不要把审计结果或生成文件放到 `/tmp`。每个阶段都应记录：输入 refs、输出 artifact、owner、验收命令和 `preserve/adapt/drop` 决策。

## 关键验收原则

- fork-owned plugin、runtime、build 和 CI 文件不能因 whole-file/tree replacement 被删除；
- 动态 `transformer_engine_torch` 必须与目标 release 的 Python/native public contract 对齐；
- enum、sentinel、import-time callable 和签名必须逐项检查；
- runtime 修改按“定义—导入—alias—调用”链验证；
- 每个 backend、CI surface、submodule 和测试入口都有 owner 或明确排除理由；
- GPU、reference、plugin manager 和 staged import 测试结果必须分开记录；
- 未经用户批准，不执行 merge-to-main、push、PR、tag 或 release。

## 本仓库当前实例

本仓库外部 artifact 中保存了本次 FL 设计基线和升级证据；仓库内的 skills 保持版本无关，只提供可复用流程和验收标准。
