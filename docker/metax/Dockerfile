# Build from the repository root:
# docker build -f docker/metax/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-metax-ci .
ARG METAX_BASE_IMAGE=harbor.baai.ac.cn/flagos-base/flagos-base-metax-maca3.8.1.3:2.3.0@sha256:5951295dd62343881e123d8e13766a4686fbc275cf510de713bdd8b78b681e6f
FROM ${METAX_BASE_IMAGE}
ARG METAX_BASE_IMAGE
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
ARG PYPI_INDEX=https://pypi.org/simple
ARG METAX_TE_WHEEL=https://repos.metax-tech.com/r/maca-pypi/packages/transformer-engine/2.13.0+2b7ba8da.maca3.7.1.0.torch2.8/transformer_engine-2.13.0+2b7ba8da.maca3.7.1.0.torch2.8-cp312-cp312-linux_x86_64.whl#sha256=513dfd3e3f668860a356d9dc68268143c0e3ee53524cb07fc72b19fe7d7d1ebe
ARG TE_FL_WHEEL=https://resource.flagos.net/repository/flagos-pypi-thead/packages/transformer-engine/2.17.0+fl.0.3.0/transformer_engine-2.17.0+fl.0.3.0-py3-none-any.whl#sha256=d6abfe990299995bafcc70cbd4c28f9cad176f34064e4b887c3e0c051e74d778
LABEL org.opencontainers.image.title="TransformerEngine-FL MetaX CI" \
      org.opencontainers.image.source="https://github.com/BrianPei/TransformerEngine-FL" \
      org.opencontainers.image.base.name="${METAX_BASE_IMAGE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="2.17.0+fl.0.3.0" \
      io.flagos.accelerator.backend="metax"

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/te-ci \
    PATH=/opt/te-ci/bin:/opt/maca/tools/cu-bridge/bin:${PATH} \
    PLATFORM=metax \
    GEMS_VENDOR=metax \
    TE_FL_SKIP_CUDA=1 \
    NVTE_WITH_MACA=1 \
    NVTE_WITH_NCCL_EP=0 \
    NVTE_FRAMEWORK=pytorch \
    MACA_HOME=/opt/maca \
    CUDA_HOME=/opt/maca \
    TE_METAX_TE_HOME=/opt/metax-te \
    PIP_CONSTRAINT=/opt/te-image/constraints.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv build-essential cmake ninja-build \
        ca-certificates curl git openssh-server libgl1 libglib2.0-0 libnuma1 libibverbs1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/te-ci \
    && ln -sf /opt/maca/tools/cu-bridge/bin/cucc /opt/maca/tools/cu-bridge/bin/nvcc

# Keep dependency pins in this standalone Dockerfile.
RUN mkdir -p /opt/te-image \
    && printf '%s\n' \
        'torch==2.10.0+metax3.8.1.0' \
        'torchvision==0.25.0+metax3.8.1.0' \
        'triton==3.6.0+metax3.8.1.0' \
        'flash-attn==2.6.3+metax3.8.1.0torch2.10' \
        'apex==0.1+metax3.8.1.0' \
        'numpy==1.26.4' \
        'pytest==8.2.1' \
        > /opt/te-image/constraints.txt \
    && printf '%s\n' \
        'torch @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/torch/2.10.0+metax3.8.1.0/torch-2.10.0+metax3.8.1.0-cp312-cp312-linux_x86_64.whl#sha256=e17cbcc675ea6e82b9a6dc4dcd6f3ae5d75d1ac6d1c428f97f244bb4344d0b54' \
        'triton @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/triton/3.6.0+metax3.8.1.0/triton-3.6.0+metax3.8.1.0-cp312-cp312-linux_x86_64.whl#sha256=f73dbb1b9bac8de6a17a6afb623329a372b0a2a10b1593ba7b19e0541fab20fe' \
        'torchvision @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/torchvision/0.25.0+metax3.8.1.0/torchvision-0.25.0+metax3.8.1.0-cp312-cp312-linux_x86_64.whl#sha256=2fb0078a2b4259bc52cd9d17f5e2bdd0f3378a5aea4810d838eccc978e40f2b6' \
        'flash-attn @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/flash-attn/2.6.3+metax3.8.1.0torch2.10/flash_attn-2.6.3+metax3.8.1.0torch2.10-cp312-cp312-linux_x86_64.whl#sha256=7b12945e844565f7fe933ba2460bd11a23e004812caa6e028ac313b2ef0ee169' \
        'apex @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/apex/0.1+metax3.8.1.0/apex-0.1+metax3.8.1.0-cp312-cp312-linux_x86_64.whl#sha256=57dfdc64caad6e05e43233d8706f49279bbb43ab411509c0491aa56ebc3bb158' \
        'flag-gems @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/flag-gems/5.4.0/flag_gems-5.4.0-py3-none-any.whl#sha256=09cb175e29f4fdfc78b87df46c4500d5a9b451e12cd64dda9a7d669f522bf5e5' \
        'numpy==1.26.4' \
        'pytest==8.2.1' \
        'coverage==7.6.1' \
        'pytest-cov==5.0.0' \
        'expecttest==0.3.0' \
        'pybind11==3.0.3' \
        'ninja==1.13.0' \
        'pydantic==2.12.4' \
        'packaging==26.3' \
        'importlib-metadata==8.7.0' \
        'einops==0.8.1' \
        'nvdlfw-inspect==0.2.2' \
        'onnx==1.19.1' \
        'onnxscript==0.5.6' \
        'onnxruntime==1.23.2' \
        'onnxruntime-extensions==0.14.0' \
        'transformers==4.57.1' \
        'torchao==0.13.0' \
        'tensorboard==2.20.0' \
        'six==1.17.0' \
        'regex==2025.11.3' \
        'psutil==7.1.3' \
        'PyYAML==6.0.1' \
        > /opt/te-image/requirements.txt
RUN python -m pip install --no-cache-dir --index-url "${PYPI_INDEX}" \
        pip==24.0 setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install --no-cache-dir --index-url "${PYPI_INDEX}" \
        -r /opt/te-image/requirements.txt

# The vendor wheel and TE-FL both own transformer_engine/. Keep the native
# vendor libraries outside site-packages; the MetaX plugin loads them by path.
# This vendor wheel targets torch 2.8. On C550 with torch 2.10, vendor GEMM
# falls back to FlagGems; FP32/BF16 forward and backward were validated.
RUN python -m pip install --no-cache-dir --no-deps --target "${TE_METAX_TE_HOME}" "${METAX_TE_WHEEL}" \
    && python -m pip install --no-cache-dir --index-url "${PYPI_INDEX}" "${TE_FL_WHEEL}" \
    && python -m pip check \
    && python -m pip freeze --all > /opt/te-image/requirements.lock \
    && python -c 'import importlib.metadata as m; from pathlib import Path; assert m.version("torch") == "2.10.0+metax3.8.1.0"; assert m.version("transformer_engine") == "2.17.0+fl.0.3.0"; assert list(Path("/opt/metax-te/transformer_engine").glob("transformer_engine_torch*.so"))'

WORKDIR /workspace
CMD ["bash"]
