# Build from repository root; CI targets NVIDIA A100 (sm80).
# docker build --target ci -f docker/cuda/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-cuda-ci .
ARG CUDA_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-nvidia-cuda13.3@sha256:95e04fab2b5c2ffb6f3d0142d313afbc3bfef5622c28ac14aed5598c01f32cd9
FROM ${CUDA_BASE_IMAGE} AS base
ARG CUDA_BASE_IMAGE
ARG BUILD_DATE=unknown
ARG VCS_REF=b99c2fa89ba0931bdf8d724e3da8278620574297
ARG IMAGE_VERSION=2.17.0-cuda-ci
LABEL org.opencontainers.image.title="transformer-engine-fl" \
      org.opencontainers.image.description="NVIDIA A100 CUDA 13.3 TE development and CI environment" \
      org.opencontainers.image.source="https://github.com/flagos-ai/TransformerEngine-FL" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${CUDA_BASE_IMAGE}" \
      io.flagos.accelerator.backend="cuda" \
      io.flagos.image.stage="base"
ENV VIRTUAL_ENV=/flagos \
    CUDA_HOME=/usr/local/cuda-13.3 \
    NVCC=/usr/local/cuda-13.3/bin/nvcc \
    PATH=/flagos/bin:/usr/local/cuda-13.3/bin:${PATH} \
    PYTHONPATH=/opt/flagtree:/opt/triton \
    CUDNN_PATH=/flagos/lib/python3.12/site-packages/nvidia/cudnn \
    TE_LIB_PATH=/flagos/lib/python3.12/site-packages/transformer_engine \
    CPATH=/flagos/lib/python3.12/site-packages/nvidia/nccl/include:/flagos/lib/python3.12/site-packages/nvidia/cudnn/include \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_NCCL_EP=0 NVTE_CUDA_ARCHS=80 \
    PIP_CONSTRAINT=/opt/te-image/constraints.txt
# Preserve every inherited Python package and lock the complete addition closure.
# FlagTree supplies the active Triton implementation; /opt/triton also exposes
# the inherited 3.7.1 distribution metadata required by PyTorch. Neither is replaced.
RUN mkdir -p /opt/te-image \
    && printf '%s\n' \
        absl-py==2.5.1 \
        annotated-types==0.8.0 \
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
        cuda-bindings==13.5.0 \
        cuda-pathfinder==1.8.6 \
        cuda-toolkit==13.0.3.0 \
        dill==0.4.1 \
        distlib==0.4.4 \
        einops==0.8.2 \
        expecttest==0.3.0 \
        filelock==4.2.9 \
        flag_gems==5.4.0 \
        flagcx==0.14.1+cuda13.3 \
        flagtree==0.7.0 \
        flatbuffers==25.12.19 \
        fsspec==2026.9.0 \
        greenlet==3.5.6 \
        grpcio==1.85.1 \
        hf-xet==1.8.0 \
        huggingface_hub==0.36.2 \
        identify==2.6.20 \
        idna==3.20 \
        importlib_metadata==10.2.0 \
        iniconfig==2.3.1 \
        isort==6.1.0 \
        Jinja2==3.1.6 \
        lazy-importer==0.1.0 \
        Markdown==3.11.3 \
        MarkupSafe==3.0.4 \
        mccabe==0.7.0 \
        ml_dtypes==0.5.4 \
        mpmath==1.3.0 \
        mypy_extensions==1.2.0 \
        networkx==3.7 \
        ninja==1.13.0 \
        nodeenv==1.12.0 \
        numpy==1.26.4 \
        nvdlfw_inspect==0.2.2 \
        nvidia-cublas==13.1.1.3 \
        nvidia-cuda-cupti==13.0.85 \
        nvidia-cuda-nvrtc==13.0.88 \
        nvidia-cuda-runtime==13.0.96 \
        nvidia-cudnn-cu13==9.20.0.48 \
        nvidia-cufft==12.0.0.61 \
        nvidia-cufile==1.15.1.6 \
        nvidia-curand==10.4.0.35 \
        nvidia-cusolver==12.0.4.66 \
        nvidia-cusparse==12.6.3.3 \
        nvidia-cusparselt-cu13==0.8.1 \
        nvidia-nccl-cu13==2.29.7 \
        nvidia-nvjitlink==13.5.37 \
        nvidia-nvshmem-cu13==3.4.5 \
        nvidia-nvtx==13.0.85 \
        onnx==1.23.2 \
        onnx-ir==1.1.0 \
        onnxruntime==1.31.1 \
        onnxruntime_extensions==0.15.2 \
        onnxscript==0.7.2 \
        packaging==26.3 \
        pathspec==1.1.1 \
        pillow==13.0.0 \
        pip==26.3 \
        platformdirs==4.14.5 \
        pluggy==1.7.0 \
        pre_commit==4.7.0 \
        protobuf==7.36.2 \
        psutil==7.2.2 \
        pybind11==3.0.3 \
        pydantic==2.14.3 \
        pydantic_core==2.50.4 \
        pylint==3.3.4 \
        pyproject_hooks==1.3.3 \
        pytest==8.2.1 \
        pytest-cov==7.1.0 \
        python-discovery==1.9.5 \
        PyYAML==6.0.1 \
        regex==2026.11.9 \
        requests==2.34.2 \
        safetensors==0.8.0 \
        setuptools==84.2.1 \
        setuptools-scm==9.2.2 \
        six==1.17.0 \
        SQLAlchemy==2.0.57 \
        sympy==1.14.0 \
        tensorboard==2.21.0 \
        tensorboard-data-server==0.7.2 \
        tokenizers==0.22.2 \
        tomlkit==0.15.2 \
        torch==2.13.0+cu130 \
        torchao==0.13.0 \
        torchaudio==2.11.0+cu130 \
        torchvision==0.28.0+cu130 \
        tqdm==4.70.1 \
        transformers==4.57.6 \
        triton==3.7.1 \
        typing_extensions==4.16.0 \
        typing-inspection==0.4.5 \
        urllib3==2.8.0 \
        vermin==1.9.2 \
        virtualenv==21.15.6 \
        Werkzeug==3.1.9 \
        wheel==0.48.0 \
        zipp==4.1.2 > /opt/te-image/constraints.txt
# Metadata checks do not initialize CUDA during an image build.
RUN python -c 'import sys, importlib.metadata as m; assert sys.version_info[:3] == (3,12,13); expected={"torch":"2.13.0+cu130","numpy":"1.26.4","flag_gems":"5.4.0","flagtree":"0.7.0","flagcx":"0.14.1+cuda13.3"}; actual={k:m.version(k) for k in expected}; assert actual == expected, actual; print(actual)' \
    && python -c 'import pathlib,site,sys; (pathlib.Path(site.getsitepackages()[0])/"flagos_runtime.pth").write_text("/opt/flagtree\n/opt/triton\n"); target=pathlib.Path(sys.base_prefix)/"bin/python3-config"; assert target.is_file(); pathlib.Path("/flagos/bin/python3-config").symlink_to(target)'
WORKDIR /workspace
CMD ["bash"]

FROM base AS release
LABEL io.flagos.image.stage="release"
# This published wheel provides FlagOS/Reference Python paths, not CUDA binaries.
ENV TE_FL_SKIP_CUDA=1 TE_FL_PREFER=reference
ARG TE_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-thead/packages/transformer-engine/2.17.0+fl.0.3.0/transformer_engine-2.17.0+fl.0.3.0-py3-none-any.whl#sha256=d6abfe990299995bafcc70cbd4c28f9cad176f34064e4b887c3e0c051e74d778
RUN python -m pip install --no-cache-dir \
        pydantic==2.14.3 importlib_metadata==10.2.0 einops==0.8.2 onnx==1.23.2 onnxscript==0.7.2 \
        nvdlfw_inspect==0.2.2 "${TE_WHEEL_URL}" \
    && python -m pip check

FROM release AS dev
LABEL io.flagos.image.stage="dev"
# The runtime holds libcublas and NCCL. Match development packages to those
# libraries instead of upgrading the CUDA runtime with a rolling meta-package.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cuda-nvcc-13-3=13.3.33-1 cuda-cudart-dev-13-3=13.3.29-1 \
        libcublas-dev-13-3=13.5.1.27-1 libcusparse-dev-13-3=12.8.1.7-1 \
        libcusolver-dev-13-3=12.2.2.18-1 libcurand-dev-13-3=10.4.3.29-1 \
        libcufft-dev-13-3=12.3.0.29-1 cuda-nvrtc-dev-13-3=13.3.33-1 \
        libnvjitlink-dev-13-3=13.3.33-1 cuda-cupti-dev-13-3=13.3.75-1 cuda-nvml-dev-13-3=13.3.29-1 \
        cuda-profiler-api-13-3=13.3.27-1 \
        libnuma-dev build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# CMake, Git, Ninja, pybind11 and Python development headers are inherited.
# CPATH exposes the existing pip NCCL headers, including builds with NCCL EP off.
RUN command -v gcc g++ cmake git ninja nvcc python3-config \
    && python -m pip install --no-cache-dir \
        wheel==0.48.0 build==1.6.1 cpplint==1.6.0 pylint==3.3.4 black==24.4.2 \
        clang-format==18.1.6 pre_commit==4.7.0 vermin==1.9.2 six==1.17.0 regex==2026.11.9 \
        psutil==7.2.2 tensorboard==2.21.0 \
    && python -m pip check

FROM dev AS native-build
ARG TE_SOURCE_REF=b99c2fa89ba0931bdf8d724e3da8278620574297
ARG MAX_JOBS=32
ENV TE_FL_SKIP_CUDA=0 SKIP_CUDA_BUILD=0 TE_FL_PREFER=vendor \
    NVTE_WITH_CUDA=1 NVTE_WITH_MACA=0 NVTE_NO_LOCAL_VERSION=1 \
    NVTE_BUILD_THREADS_PER_JOB=1
# Keep the native extension and Python sources at the same revision. Build
# without visible GPUs, then copy only its wheel into the final CI image.
RUN git init /tmp/te-source \
    && git -C /tmp/te-source remote add origin https://github.com/flagos-ai/TransformerEngine-FL.git \
    && git -C /tmp/te-source fetch --depth 1 origin "${TE_SOURCE_REF}" \
    && git -C /tmp/te-source checkout --detach FETCH_HEAD \
    && git -C /tmp/te-source submodule update --init --recursive \
    && cd /tmp/te-source \
    && CUDA_VISIBLE_DEVICES= MAX_JOBS="${MAX_JOBS}" python -m pip wheel . -v \
        --no-deps --no-build-isolation --wheel-dir /opt/te-native-wheels

FROM dev AS ci
LABEL io.flagos.image.stage="ci"
ENV TE_FL_SKIP_CUDA=0 SKIP_CUDA_BUILD=0 TE_FL_PREFER=vendor \
    NVTE_WITH_CUDA=1 NVTE_WITH_MACA=0
COPY --from=native-build /opt/te-native-wheels/ /opt/te-image/native-wheels/
RUN python -m pip install --no-cache-dir \
        pytest==8.2.1 expecttest==0.3.0 coverage==7.17.2 pytest-cov==7.1.0 onnxruntime==1.31.1 \
        onnxruntime_extensions==0.15.2 transformers==4.57.6 torchao==0.13.0 requests==2.34.2 \
    && python -m pip install --no-cache-dir --no-deps --force-reinstall /opt/te-image/native-wheels/*.whl \
    && python -m pip check \
    && python -m coverage --version \
    && python -c 'import importlib.metadata as m; d=m.distribution("transformer_engine"); assert d.version == "2.17.0"; assert any("transformer_engine_torch_nv" in str(p) and str(p).endswith(".so") for p in d.files)'
# CI installs its PR wheel on top; the image does not contain a source checkout.
