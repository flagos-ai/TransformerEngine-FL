# Build from repository root:
# docker build --target ci -f docker/iluvatar/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformerengine-fl:iluvatar-corex4.5.0-ci-recipe .
# Single-file base/dev/ci/release recipe, following vllm-plugin-FL.
# Keep the validated TE CoreX matrix; do not inherit vLLM-specific dependencies.
ARG ILUVATAR_BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/transformerengine-fl:iluvatar-corex4.5.0-ci
FROM ${ILUVATAR_BASE_IMAGE} AS base
ARG ILUVATAR_BASE_IMAGE
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=unknown
ARG IMAGE_STAGE=base
ARG SOURCE_URL=https://github.com/BrianPei/TransformerEngine-FL
LABEL org.opencontainers.image.title="transformerengine-fl" \
      org.opencontainers.image.description="Iluvatar BI-V150 CoreX 4.5 TE environment" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${ILUVATAR_BASE_IMAGE}" \
      io.flagos.accelerator.backend="iluvatar" \
      io.flagos.image.stage="${IMAGE_STAGE}" \
      io.flagos.torch.version="2.10.0+corex.4.5.0.20260804" \
      io.flagos.driver.version="4.5.0"
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/te-ci \
    PATH=/opt/te-ci/bin:/usr/local/corex/bin:${PATH} \
    GEMS_VENDOR=iluvatar \
    TE_FL_SKIP_CUDA=1 \
    NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \
    NVTE_FRAMEWORK=pytorch \
    NVTE_WITH_NCCL_EP=0 \
    PIP_CONSTRAINT=/opt/te-image/constraints.txt
RUN mkdir -p /opt/te-image \
    && printf '%s\n' 'numpy==1.26.4' 'scipy==1.15.3' 'torch==2.10.0+corex.4.5.0.20260804' 'torchvision==0.25.0+corex.4.5.0.20260804' 'triton==3.2.0+corex.4.5.0.20260804' 'ixte==5.0.0+corex.4.5.0.20260804' > /opt/te-image/constraints.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
# Metadata validation only: image builds must not initialize GPU drivers.
RUN python -c 'import sys, importlib.metadata as m; assert sys.version_info[:3] == (3,12,3); expected={"torch":"2.10.0+corex.4.5.0.20260804","torchvision":"0.25.0+corex.4.5.0.20260804","triton":"3.2.0+corex.4.5.0.20260804","ixte":"5.0.0+corex.4.5.0.20260804","flag_gems":"5.4.0"}; actual={k:m.version(k) for k in expected}; assert actual == expected, actual; print(actual)'
# The inherited image may contain an old TE installation. CI setup installs the
# checked-out revision; the image build neither clones nor pins TE source.
WORKDIR /workspace
CMD ["bash"]

FROM base AS release
# Runtime environment only; the base may already contain development packages.
LABEL io.flagos.image.stage="release"
WORKDIR /workspace

FROM base AS dev
LABEL io.flagos.image.stage="dev"
RUN python -m pip install --no-cache-dir --no-deps \
    setuptools==84.2.1 wheel==0.48.0 pybind11==3.1.0 \
    pytest==8.4.2 coverage==7.17.0 pytest-cov==7.1.0 \
    pluggy==1.7.0 iniconfig==2.3.1 packaging==26.3 Pygments==2.21.0

FROM dev AS ci
LABEL io.flagos.image.stage="ci"
RUN python -m pip install --no-cache-dir --no-deps \
    absl-py==2.5.1 annotated-types==0.8.0 certifi==2026.10.23 charset-normalizer==3.5.2 \
    coverage==7.17.0 einops==0.8.2 expecttest==0.3.0 filelock==4.2.4 \
    flatbuffers==25.12.19 fsspec==2026.9.0 greenlet==3.5.6 grpcio==1.85.0 \
    hf-xet==1.7.1 huggingface_hub==0.36.2 idna==3.20 importlib_metadata==10.1.0 \
    iniconfig==2.3.1 Jinja2==3.1.6 Markdown==3.11.2 MarkupSafe==3.0.4 \
    ml_dtypes==0.5.4 mpmath==1.3.0 networkx==3.7 numpy==1.26.4 \
    nvdlfw_inspect==0.2.2 onnx==1.23.2 onnx-ir==1.1.0 onnxruntime==1.31.1 \
    onnxruntime_extensions==0.15.2 onnxscript==0.7.2 packaging==26.3 pillow==13.0.0 \
    pip==24.0 pluggy==1.7.0 protobuf==7.36.2 psutil==7.2.2 \
    pybind11==3.1.0 pydantic==2.14.1 pydantic_core==2.50.1 Pygments==2.21.0 \
    pytest==8.4.2 pytest-cov==7.1.0 PyYAML==6.0.1 regex==2026.9.29 \
    requests==2.34.2 safetensors==0.8.0 scipy==1.15.3 setuptools==84.2.1 \
    six==1.17.0 SQLAlchemy==2.0.55 sympy==1.14.0 tensorboard==2.21.0 \
    tensorboard-data-server==0.7.2 tokenizers==0.22.2 torchao==0.13.0 tqdm==4.70.1 \
    transformers==4.57.6 typing-inspection==0.4.4 typing_extensions==4.16.0 urllib3==2.8.0 \
    Werkzeug==3.1.9 wheel==0.48.0 zipp==4.1.2 \
    && python -m pip check \
    && python -c 'import importlib.metadata as m; assert m.version("coverage")=="7.17.0"; assert m.version("pytest-cov")=="7.1.0"' \
    && python -m coverage --version

