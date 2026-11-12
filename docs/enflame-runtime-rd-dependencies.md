# Enflame CI 镜像依赖与验证

- 基础镜像：`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-enflame-tops1.10.6:2.3.0`
- 基础镜像 digest：`sha256:69dabebaf536625255183c389c53f2cf6c2de0727b0989140dff4451941d301a`
- 成品镜像：`harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-enflame-ci`
- 成品 digest：`sha256:e3a20e738e1702270cc07f001296b449e475519ac0beb908d9710f7c73707de6`
- 配方：[docker/enflame/Dockerfile](../docker/enflame/Dockerfile)，提供 base / release / dev / ci 阶段。
- 源码对照：main @ `b99c2fa89ba0931bdf8d724e3da8278620574297`；安装 wheel：`transformer_engine==2.17.0+fl.0.3.0`。
- 硬件：8 卡 Enflame S60 / ZIXIAOC200，驱动与 efsmi 1.10.6。验证日期：2026-11-12。

## 运行依赖审计

除 TE 本体外，基础镜像缺少以下 4 项直接运行依赖。开发工具和 CI 专用包不计入此表。

| 包名 | 安装版本 | 用途 | 补装的传递依赖 |
| --- | --- | --- | --- |
| einops | 0.8.2 | 张量变换，源码声明的运行依赖 | 无 |
| onnxscript | 0.7.2 | PyTorch ONNX 导出模块导入 | onnx-ir 1.1.0、ml_dtypes 0.5.4 |
| nvdlfw-inspect | 0.2.2 | TE debug / inspect | PyYAML 已有 |
| importlib-metadata | 10.2.0 | 包元数据声明、插件发现兼容路径 | lazy-importer 0.1.0、zipp 4.1.2 |

基础镜像已有：Python 3.12.13、torch 2.11.0+cpu、torch_gcu 2.11.0+3.8.20260713、numpy 1.26.4、flag_gems 5.4.0、flagtree 0.7.0+enflame3.6、flagcx 0.14.0+tops1.10.6、pydantic 2.14.1、pydantic_core 2.50.1、onnx 1.17.0、onnxruntime 1.17.0、onnxruntime_extensions 0.15.2、transformers 4.57.6、huggingface_hub 0.36.2、PyYAML 6.0.1。

基础镜像已有 GCC/G++、Git、CMake、Ninja、pybind11 和 Python 头文件。FlagTree 位于 `/opt/flagtree`，提供 Triton 3.6.0；新增 .pth 让直接运行 Python 时也可导入，并将实际解释器的 python3-config 链接到 /flagos/bin。将基础 Shell 中的 GCU 运行环境显式写入 ENV，保证非交互调用一致。

开发和 CI 层增加 pytest 8.2.1、coverage 7.17.2、pytest-cov 7.1.0、torchao 0.13.0、build、wheel、pre-commit、pylint、cpplint、clang-format、vermin、tensorboard 等。完整精确版本固化在 Dockerfile 的 constraints 中。保留 `black==26.3.1`，因为 torch_gcu 对它有精确依赖；仓库 pre-commit 的其他 Black 版本应使用隔离环境。

## 新增依赖分类（与基础镜像差集）

以下按 Dockerfile 安装阶段区分。运行层新增 8 个第三方包，研发层新增 24 个，CI 层新增 6 个，共新增 38 个第三方包；TE 本体另计 1 个。继承已有包不计为新增。传递依赖一并列出，避免与直接需求数混淆。

| 分类 | 直接需求 | 新增传递依赖 |
| --- | --- | --- |
| 运行（4 + 4） | einops==0.8.2、onnxscript==0.7.2、nvdlfw_inspect==0.2.2、importlib_metadata==10.2.0 | onnx-ir==1.1.0、ml_dtypes==0.5.4、lazy-importer==0.1.0、zipp==4.1.2 |
| 研发（10 + 14） | build==1.6.1、wheel==0.48.0、pre_commit==4.7.0、pylint==3.3.4、cpplint==1.6.0、clang-format==18.1.6、vermin==1.9.2、tensorboard==2.21.0、pyproject_hooks==1.3.3、astroid==3.3.11 | absl-py==2.5.1、cfgv==3.5.0、distlib==0.4.4、grpcio==1.85.1、identify==2.6.20、isort==6.1.0、Markdown==3.11.3、mccabe==0.7.0、nodeenv==1.12.0、python-discovery==1.9.5、tensorboard-data-server==0.7.2、tomlkit==0.15.2、virtualenv==21.15.6、Werkzeug==3.1.9 |
| CI（4 + 2） | pytest==8.2.1、coverage==7.17.2、pytest-cov==7.1.0、torchao==0.13.0 | iniconfig==2.3.1、pluggy==1.7.0 |

分类是镜像层归属，不代表用途互斥。例如 tensorboard 放在 dev 层供研发观测，MCore CI 也会用到；astroid、pyproject_hooks 是研发工具传递依赖，但配方中显式安装并固定版本。Black 已在基础镜像中，不计新增；GCC/G++、Git、CMake、Ninja、pybind11、ONNX Runtime、transformers 等亦为继承。

## 验证结果与限制

| 项目 | 结果 |
| --- | --- |
| Dockerfile ci 阶段构建 | 通过；镜像 ID 前缀 c62c8f712e74 |
| pip check | 通过，保留厂商原始组件版本 |
| 8 卡 GCU 矩阵乘法与反向传播 | 审计环境及成品镜像全部 8 卡通过，逐卡独立进程并显式 set_device |
| TE wheel 和 checkout 导入 | 通过，Triton 3.6.0 可直接导入 |
| policy / manager / reference lifecycle、activation、dropout、GEMM | 审计环境及成品镜像各 102 passed |
| Enflame debug 组 | 3 passed、21 deselected |
| Enflame unittest 组 | 126 passed、16 skipped、10 deselected |
| Enflame distributed 组，2 进程 | 10 passed、3 skipped |
| ONNX 组 | 3 passed、1 skipped、57 deselected、5 failed，见下文 |
| MCore GPT 冒烟 | 失败于训练前 ECCL all_reduce，未完成训练，见下文 |
| TE Linear float32 / bfloat16 冒烟 | 第一项 float32 forward 即失败，bfloat16 未执行，见下文 |

当前源码和 wheel 的 TE Linear 在 prepare_forward 中要求设备类型 cuda，而 torch_gcu 的实际设备类型为 gcu，报 `TransformerEngine needs cuda. Got input on device: gcu:0`。未通过修改源码或屏蔽检查绕过。导入时还会输出 `Cannot import torch_gcu or triton_gcu.triton`；这不代表硬件 smoke 失败，也不能据此宣称 vendor 扩展已可用。

ONNX 的 5 项 LayerNorm 导出测试失败于 ONNX Runtime 加载 IR 10 模型；厂商预装 Runtime 1.17.0 仅支持 IR 9。独立升级实验使用 onnxruntime 1.31.1 后，ONNX 组达到 8 passed、1 skipped、57 deselected，但违反 `enflame-modelopt==3.6.20260615+torch.2.11.0` 对 `onnxruntime==1.17.0` 的精确依赖。最终镜像保留 1.17.0，避免破坏厂商工具链或使 pip check 失败。此冲突需另行协调厂商依赖，不能称为全量 CI 通过。

MCore 对照 commit `175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82`，使用仓库原有 Enflame integration launcher，独立本地镜像仓库提供源码，不修改已有 MCore checkout。冒烟缩小为 2 层、hidden 128、4 heads、seq 128、micro/global batch 1，FP8 auto（未启用）。C++ dataset helper 编译、分布式初始化成功，随后 `program_start_global` 的 int64 MIN all_reduce 报 `invalid type 8 for reduction operation 3` / `ecclInvalidArgument`。因此没有达到 4 步训练退出目标。

所有测试沿用已有 Enflame exclusions，包括 fused optimizer 的已知驱动风险用例；没有修改测试、TE 源码或厂商包来隐藏失败。CI 配置仅替换 ci_image 地址。

## 复现

在仓库根目录构建：

```sh
docker build --target ci --network host \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -f docker/enflame/Dockerfile \
  -t harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-enflame-ci .
```

硬件测试需参考 [.github/configs/enflame.yml](../.github/configs/enflame.yml) 暴露 8 个 GCU 设备，使用 privileged、host IPC 和对应 shm/ulimit。镜像中的 TE wheel 用于直接导入；CI 设置 PYTHONPATH 后使用被检出的源码版本。

原始构建、依赖报告、硬件日志和测试日志保存在远端 `/data/te-enflame-audit-20261112`，包括 build.log、push.log、image-packages.json、plugin-unit.log、debug.log、unittest.log、distributed.log、onnx.log、onnx-upgraded.log、mcore.log 和 coverage.xml。临时升级实验与最终镜像分开，成品版本以 image-packages.json 为准。
