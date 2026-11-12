# Enflame 成品镜像 L2 CI 实测（2026-11-12）

## 验证对象与执行边界

- 镜像：harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-enflame-ci
- 固定 digest：sha256:e3a20e738e1702270cc07f001296b449e475519ac0beb908d9710f7c73707de6
- 被测源码：00c7dc6acc3cd19e4fc41e0a95b037eef2352ec5。后续报告提交仅修改文档，不属于本次被测源码。
- 硬件：用户提供的 8 卡 Enflame S60 / ZIXIAOC200 主机，TOPS / 驱动 1.10.6。distributed 组按现有配置启动 2 个进程。
- 请求范围：enflame、L2。GitHub Actions 的 plan.json 与机器复跑 plan.json 内容一致；plan_hash 为 c2561dd360ceec8d1b84ece83a1fed5bfa259d1281a366727b5123de2388f55d。

已触发 [GitHub Actions run 43161598490](https://github.com/BrianPei/TransformerEngine-FL/actions/runs/43161598490)。请求解析、计划生成、checkout-and-config 成功；设备 preflight 等待 sy-8g-cicd-te runner，尚未执行设备测试。因此下面是用户提供机器上的 CI 计划复跑结果，不能表述为 GitHub Actions 全绿。

机器复跑使用固定 digest、同一份源码归档和 plan，每个组启动一个全新容器，沿用 enflame.yml 的 privileged、host IPC、100 GB shm、ulimit 和 8 个 GCU 设备映射。调用仓库原有 setup_enflame.sh、run_ci_test_group.py、测试 launcher 和 group-report；失败组最多重试 2 次。不修改 TE、测试或厂商包，不运行安装补丁来掩盖失败。实际控制器脚本随原始日志保存。

## 结果统计

| 执行组 | 结果 | 最终 JUnit：通过 / 跳过 / 失败 | 执行次数 | 耗时（秒） |
| --- | --- | --- | --- | --- |
| preflight | 通过 | 非 pytest 组 | 1 | 0.57 |
| pytorch_unit | 通过 | 126 / 16 / 0 | 1 | 208.80 |
| pytorch_debug | 通过 | 3 / 0 / 0 | 1 | 57.25 |
| pytorch_distributed | 通过 | 10 / 3 / 0 | 1 | 57.96 |
| pytorch_onnx | 失败，exit 1 | 3 / 1 / 5 | 3 | 92.61 |
| pytorch_mcore_integration（默认 GitHub 源） | 拉取阻塞后主动停止，容器 exit 137；报告为 cancelled | 未到训练阶段，无 JUnit | 1（未进入重试） | 480.45 |
| MCore 补充诊断（同 commit 本地源） | 失败，exit 1，训练前 ECCL 错误 | 非 pytest 组 | 1 | 46.78 |

pytest 最终去重统计为 **142 passed、20 skipped、5 failed，共 167 项**。失败组重试会覆盖同名 JUnit 文件，因此按最后一次结果计数；ONNX 三次尝试均为同样 5 项失败，日志内累计出现 15 次用例失败，不计为 15 个唯一失败用例。MCore 是独立 shell 集成组，其失败不能混入 pytest 用例数量。

### ONNX：5 个唯一失败用例，1 类根因

都位于 tests/pytorch/test_onnx_export.py：

- test_export_layernorm_recipe[precision0-None]
- test_export_layernorm_recipe[precision1-None]
- test_export_layernorm_zero_centered_gamma
- test_export_layernorm_normalization[LayerNorm]
- test_export_layernorm_normalization[RMSNorm]

共同异常：

    onnxruntime.capi.onnxruntime_pybind11_state.Fail
    Unsupported model IR version: 10, max supported IR version: 9

镜像继承 onnx==1.17.0、onnxruntime==1.17.0；当前导出模型 IR 为 10，而 Runtime 仅支持 IR 9。这属于导出与推理运行时兼容冲突，不是 pytest 或 coverage 缺失。厂商 enflame-modelopt==3.6.20260615+torch.2.11.0 精确依赖 onnxruntime==1.17.0，不能直接升级后宣称依赖一致。此前独立升级实验的结果见[依赖审计](enflame-runtime-rd-dependencies.md)，不属于本次固定 digest 的测试结果。

### MCore：原始流程与补充诊断分列

原始流程使用脚本默认 GitHub URL 拉取 Megatron-LM-FL，持续停在 git clone。容器内独立连通性探测报告 curl (28) SSL connection timeout；运行 480.45 秒后主动停止该新建容器，Docker exit 137。这是网络阻塞后的人工终止，不是等满配置超时，也不是完成集成测试后的返回码；原始组没有完成重试或训练。为保留真实状态，停止后根据实测返回码与 execute=cancelled 补生成 group-report，未把它标记为通过或伪造测试失败。

为区分源码传输与运行时问题，另起同 digest 的全新容器，将 MCORE_REPO_URL 指向本地 bare mirror，固定同一 commit 175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82。除此之外沿用默认参数：12 层、hidden 512、8 heads、seq 1024、micro batch 4、global batch 32、bf16、单进程、FP8 auto（未启用）、exit-interval 4。该补充诊断运行 1 次，46.78 秒，exit 1。

C++ dataset helper 编译与设备分布式初始化完成，训练前的 program_start_global 执行 int64 MIN all_reduce 时失败：

    AllReduce : invalid type 8 for reduction operation 3 prot 0
    torch.distributed.all_reduce(program_start_global, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.DistBackendError: ECCL ... ecclInvalidArgument
    ProcessGroupECCL.cpp:2718

随后 torchrun 报 ChildFailedError，这是上述 ECCL 异常的包装，不另算一类根因。未完成训练，也未达到 4 步退出目标。补充诊断改变了源码传输方式，不能取代原始流程结果。

## 新增依赖分类

以基础镜像与成品镜像已安装包清单做差集，基础 89 个 Python distributions、成品 128 个：新增 38 个第三方包及 TE 本体 1 个；已有包版本变化 0、移除 0，无新增系统包。

| 镜像层 | 新增数量（含传递依赖） | 主要直接用途 / 工具 |
| --- | --- | --- |
| release：运行 | 8 | einops、onnxscript、nvdlfw-inspect、importlib-metadata 及其传递依赖 |
| dev：研发 | 24 | build、wheel、pre-commit、pylint、cpplint、clang-format、vermin、tensorboard 及配套依赖 |
| ci：测试执行 | 6 | pytest 8.2.1、coverage 7.17.2、pytest-cov 7.1.0、torchao 0.13.0；传递新增 iniconfig 2.3.1、pluggy 1.7.0 |
| TE 本体 | 1 | transformer_engine 2.17.0+fl.0.3.0 |

精确版本及直接 / 传递需求见[依赖分类明细](enflame-runtime-rd-dependencies.md#新增依赖分类与基础镜像差集)，配方见 [Dockerfile](../docker/enflame/Dockerfile)。分类代表镜像安装层，不代表用途互斥；例如 dev 层 tensorboard 也被 MCore CI 使用。Black 26.3.1、GCC/G++、Git、CMake、Ninja、pybind11、six、regex 等均继承，不计新增。

## 覆盖边界和非终止性告警

平台 gate 读取齐全的 6 份组报告，结果 exit 1：ONNX 为 test_failed，原始 MCore 为 cancelled / 未执行完；其余 4 组 passed。补充 MCore 诊断不替换原始报告。合并 4 个 pytest 组的覆盖率为 24.26%，仅表示当前配置与 exclusions 下采集到的范围；未运行历史基线比较，不能据此声称覆盖率门禁通过。

现有配置明确不支持 normalization、quantization、vendor_backend 三个逻辑测试组。沿用现有 exclusions，不将未执行范围记为通过，也不把它们计成新增失败。

导入时的 Cannot import torch_gcu or triton_gcu.triton、ONNX 的 CUDAExecutionProvider 不可用、transfer_to_gcu 的 JIT 降级，以及 Apex 缺失时回退 Torch Norm 均保存在日志中。它们与终止性异常分别记录；本次没有以这些警告单独增加失败计数，也没有证据说明需要为此向镜像追加 Apex/CUDA 包。

此前 TE Linear 独立 smoke 的 cuda/gcu 设备检查错误不属于本次 L2 选中用例，单列于依赖审计，不能混入本次 5 个 pytest 失败。

## 原始证据与复现

- 远端目录：/data/te-enflame-ci-run-20261112
- 本地归档目录：/Users/meicheng/Desktop/CICD/enflame-ci-run-20261112
- results/test-summary.json：组级结果与 JUnit 去重计数。
- results/failures.json：5 个失败用例的原始 message、traceback、JUnit 路径。
- results/*.log：每组完整输出，含重试；mcore-offline.log 是补充诊断。
- results/github-connectivity.log、mcore-network-stop-time：连通性探测与主动停止时间。
- results/platform-report.json、platform-gate.log、coverage.json / coverage.xml：平台汇总和覆盖率。
- results/container-identities.txt：每个测试容器的镜像 ID 和退出码；全部固定同一成品镜像。
- source/logs：JUnit；plan.json、matrices.json：原始执行范围。
- execute_group.py、run_matrix.py、aggregate_ci.py、mcore_offline.py、finalize_ci.py：本次复跑与停止状态记录控制器。
- dependency-delta.json、image-packages.json：新增依赖差集与镜像包清单。

Actions 请求可用 gh workflow run ci-requested.yml，在 BrianPei/TransformerEngine-FL 的 codex/enflame-ci-image ref 上传入 platforms=enflame、test_level=L2、checkout_ref=codex/enflame-ci-image。精确复现本次结果时应检出上述被测 SHA，固定上述镜像 digest，并保留 plan 与控制器中的参数；本次未执行线上 FlagOps 上报、覆盖率历史比较或 artifact 上传服务。
