# Build from repository root:
# docker build --target ci -f docker/hygon/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-hygon-ci .
# Single-file base/release/dev/ci recipe, matching docker/iluvatar/Dockerfile.
ARG HYGON_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-hygon-dtk26.10@sha256:1a22c8198f87d7224008fc95dd3f569259635ac6d162989d4db2e85cbb041cd4
FROM ${HYGON_BASE_IMAGE} AS base
ARG HYGON_BASE_IMAGE
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=2.17.0-hygon-ci
ARG SOURCE_URL=https://github.com/flagos-ai/TransformerEngine-FL
LABEL org.opencontainers.image.title="transformer-engine-fl" \
      org.opencontainers.image.description="Hygon BW1000 DTK 26.10 TE environment" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${HYGON_BASE_IMAGE}" \
      io.flagos.accelerator.backend="hygon" \
      io.flagos.image.stage="base"
ENV VIRTUAL_ENV=/flagos \
    PATH=/flagos/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/hyhal/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/lib:/opt/dtk/.hyhal/lib64 \
    HIP_PATH=/opt/dtk/hip ROCM_PATH=/opt/dtk DTKROOT=/opt/dtk HYHAL_PATH=/opt/hyhal \
    TE_FL_SKIP_CUDA=1 TE_FL_PREFER=reference NVTE_FRAMEWORK=pytorch NVTE_WITH_NCCL_EP=0 \
    NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 NVTE_UnfusedDPA_Emulate_FP8=1
# Preserve the audited vendor stack and dependency versions.
RUN mkdir -p /opt/te-image \
    && printf '%s\n' \
        flag_gems==5.4.0 \
        flagcx==0.14.1+dtk2610 \
        flagtree==0.7.0+hcu3.6 \
        numpy==1.26.4 \
        torch==2.13.0+dtk2610.2610261432.gefb4e1 \
        torchaudio==2.11.0+dtk2610.torch2130.2610221109.g1da530 \
        torchvision==0.28.0+dtk2610.torch2130.2610212003.gac691b \
        absl-py==2.5.1 \
        annotated-doc==0.0.6 \
        annotated-types==0.8.0 \
        anyio==4.16.0 \
        astroid==3.3.11 \
        black==24.4.2 \
        build==1.6.1 \
        certifi==2026.10.23 \
        cfgv==3.5.0 \
        charset-normalizer==3.5.2 \
        clang-format==18.1.6 \
        click==8.5.1 \
        coverage==7.17.2 \
        cpplint==1.6.0 \
        dill==0.4.1 \
        distlib==0.4.4 \
        expecttest==0.3.0 \
        flatbuffers==25.12.19 \
        grpcio==1.85.1 \
        h11==0.16.0 \
        hf-xet==1.8.0 \
        httpcore2==2.16.0 \
        httpx2==2.16.0 \
        huggingface_hub==2.4.0 \
        identify==2.6.20 \
        idna==3.20 \
        importlib_metadata==10.2.0 \
        iniconfig==2.3.1 \
        isort==6.1.0 \
        lazy-importer==0.1.0 \
        markdown-it-py==4.3.0 \
        Markdown==3.11.3 \
        mccabe==0.7.0 \
        mdurl==0.1.2 \
        ml_dtypes==0.5.4 \
        mypy_extensions==1.2.0 \
        nodeenv==1.12.0 \
        nvdlfw_inspect==0.2.2 \
        onnx-ir==1.1.0 \
        onnx==1.23.2 \
        onnxruntime==1.31.1 \
        onnxruntime_extensions==0.15.2 \
        onnxscript==0.7.2 \
        pathspec==1.1.1 \
        platformdirs==4.14.5 \
        pluggy==1.7.0 \
        pre_commit==4.7.0 \
        protobuf==7.36.2 \
        psutil==7.2.2 \
        pydantic==2.14.3 \
        pydantic_core==2.50.4 \
        Pygments==2.21.0 \
        pylint==3.3.4 \
        pyproject_hooks==1.3.3 \
        pytest-cov==7.1.0 \
        pytest==8.2.1 \
        python-discovery==1.9.5 \
        regex==2026.11.9 \
        requests==2.34.2 \
        rich==15.0.0 \
        safetensors==0.8.0 \
        shellingham==1.5.4 \
        six==1.17.0 \
        tensorboard-data-server==0.7.2 \
        tensorboard==2.21.0 \
        tokenizers==0.23.3 \
        tomlkit==0.15.2 \
        torchao==0.13.0 \
        tqdm==4.70.1 \
        transformers==5.22.2 \
        truststore==0.10.5 \
        typer==0.29.1 \
        typing-inspection==0.4.5 \
        urllib3==2.8.0 \
        vermin==1.9.2 \
        virtualenv==21.15.6 \
        Werkzeug==3.1.9 \
        wheel==0.48.0 \
        zipp==4.1.2 > /opt/te-image/constraints.txt
ENV PIP_CONSTRAINT=/opt/te-image/constraints.txt
# Metadata checks only: builds must not initialize GPU drivers.
RUN python -c 'import sys, importlib.metadata as m; assert sys.version_info[:2] == (3,12); expected={"torch":"2.13.0+dtk2610.2610261432.gefb4e1","flagtree":"0.7.0+hcu3.6","flag_gems":"5.4.0","flagcx":"0.14.1+dtk2610"}; actual={k:m.version(k) for k in expected}; assert actual == expected, actual; print(actual)' \
    && python -c 'import pathlib,site,sys; (pathlib.Path(site.getsitepackages()[0])/"flagtree_runtime.pth").write_text("/opt/flagtree\n"); target=pathlib.Path(sys.base_prefix)/"bin/python3-config"; assert target.is_file(); pathlib.Path("/flagos/bin/python3-config").symlink_to(target)'
WORKDIR /workspace
CMD ["bash"]

FROM base AS release
LABEL io.flagos.image.stage="release"
ARG TE_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-thead/packages/transformer-engine/2.17.0+fl.0.3.0/transformer_engine-2.17.0+fl.0.3.0-py3-none-any.whl#sha256=d6abfe990299995bafcc70cbd4c28f9cad176f34064e4b887c3e0c051e74d778
RUN python -m pip install --no-cache-dir  \
        pydantic==2.14.3 importlib_metadata==10.2.0 onnx==1.23.2 onnxscript==0.7.2 \
        nvdlfw_inspect==0.2.2 "${TE_WHEEL_URL}" \
    && python -m pip check

FROM release AS dev
LABEL io.flagos.image.stage="dev"
# GCC/G++, Python headers, Git, CMake, Ninja and pybind11 are inherited.
RUN command -v gcc g++ git cmake ninja python3-config \
    && python -m pip install --no-cache-dir  \
        cpplint==1.6.0 wheel==0.48.0 build==1.6.1 pyproject_hooks==1.3.3 \
        pylint==3.3.4 black==24.4.2 clang-format==18.1.6 astroid==3.3.11 \
        isort==6.1.0 mccabe==0.7.0 six==1.17.0 psutil==7.2.2 \
        tensorboard==2.21.0 grpcio==1.85.1 protobuf==7.36.2 tensorboard-data-server==0.7.2 \
        pre_commit==4.7.0 vermin==1.9.2 absl-py==2.5.1 cfgv==3.5.0 \
        click==8.5.1 dill==0.4.1 identify==2.6.20 Markdown==3.11.3 \
        mypy_extensions==1.2.0 nodeenv==1.12.0 pathspec==1.1.1 platformdirs==4.14.5 \
        tomlkit==0.15.2 virtualenv==21.15.6 distlib==0.4.4 python-discovery==1.9.5 \
        Werkzeug==3.1.9 \
    && python -m pip check

FROM dev AS ci
LABEL io.flagos.image.stage="ci"
RUN python -m pip install --no-cache-dir  \
        pytest==8.2.1 torchao==0.13.0 pluggy==1.7.0 wheel==0.48.0 \
        build==1.6.1 expecttest==0.3.0 coverage==7.17.2 pytest-cov==7.1.0 \
        onnxruntime==1.31.1 onnxruntime_extensions==0.15.2 transformers==5.22.2 huggingface_hub==2.4.0 \
        click==8.5.1 hf-xet==1.8.0 httpx2==2.16.0 httpcore2==2.16.0 \
        tokenizers==0.23.3 requests==2.34.2 charset-normalizer==3.5.2 idna==3.20 \
        urllib3==2.8.0 anyio==4.16.0 certifi==2026.10.23 h11==0.16.0 \
        importlib_metadata==10.2.0 lazy-importer==0.1.0 protobuf==7.36.2 regex==2026.11.9 \
        safetensors==0.8.0 tqdm==4.70.1 truststore==0.10.5 zipp==4.1.2 \
        flatbuffers==25.12.19 iniconfig==2.3.1 nvdlfw_inspect==0.2.2 onnx==1.23.2 \
        ml_dtypes==0.5.4 onnxscript==0.7.2 onnx-ir==1.1.0 pydantic==2.14.3 \
        pydantic_core==2.50.4 annotated-types==0.8.0 typing-inspection==0.4.5 pyproject_hooks==1.3.3 \
        typer==0.29.1 annotated-doc==0.0.6 rich==15.0.0 Pygments==2.21.0 \
        markdown-it-py==4.3.0 mdurl==0.1.2 shellingham==1.5.4 \
    && python -m pip check \
    && python -m coverage --version \
    && python -c 'import importlib.metadata as m; assert m.version("transformer_engine")=="2.17.0+fl.0.3.0"; assert m.version("pytest")=="8.2.1"'
# CI runs its checked-out revision through PYTHONPATH; do not bake a Git checkout
# or alter vendor packages to hide source compatibility failures.
