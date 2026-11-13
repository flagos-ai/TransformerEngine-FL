# Build from repository root:
# docker build --target ci -f docker/iluvatar/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-iluvatar-ci .
# Single-file base/release/dev/ci recipe, matching docker/hygon/Dockerfile.
# Audited offline wheels include the repacked IXTE namespace adapter, which is
# absent from the runtime base. Keep this artifact source pinned by digest.
ARG ILUVATAR_ARTIFACT_IMAGE=harbor.baai.ac.cn/flagos-dev/transformer-engine-fl@sha256:c4981e71166516921b6a5023584c88f745bb9c7aa7b42aea03b5a2c6bb0f8c2f
ARG ILUVATAR_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-iluvatar-corex4.5.0@sha256:6122f544f0f000f95b5739e6a2c4cadbf1293b3ace0f8c9ebca5715510ed1831
FROM ${ILUVATAR_ARTIFACT_IMAGE} AS artifacts
FROM ${ILUVATAR_BASE_IMAGE} AS base
ARG ILUVATAR_BASE_IMAGE
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=2.17.0-iluvatar-ci
ARG SOURCE_URL=https://github.com/flagos-ai/TransformerEngine-FL
LABEL org.opencontainers.image.title="transformer-engine-fl" \
      org.opencontainers.image.description="Iluvatar BI-V150 CoreX 4.5 TE environment" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${ILUVATAR_BASE_IMAGE}" \
      io.flagos.accelerator.backend="iluvatar" \
      io.flagos.image.stage="base"
SHELL ["/bin/bash", "-c"]
ENV VIRTUAL_ENV=/flagos \
    PATH=/flagos/bin:/usr/local/corex/bin:${PATH} \
    IX_HOME=/usr/local/corex CUDA_HOME=/usr/local/corex GEMS_VENDOR=iluvatar \
    TE_FL_SKIP_CUDA=1 NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_NCCL_EP=0 \
    TE_FL_PREFER=flagos TE_FL_PER_OP="generic_gemm=reference;multi_tensor_unscale_l2norm=reference" \
    TE_FL_STRICT=0 NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 \
    PYTHON_BIN=/flagos/bin/python PYTHONDONTWRITEBYTECODE=1
# Preserve the complete audited wheelhouse, constraints and IXTE provenance.
COPY --from=artifacts /opt/te-audit/wheelhouse/ /opt/te-audit/wheelhouse/
COPY --from=artifacts /opt/te-audit/constraints.txt /opt/te-audit/constraints.txt
COPY --from=artifacts /opt/te-audit/runtime-baseline.json /opt/te-audit/runtime-baseline.json
COPY --from=artifacts /opt/te-audit/final-versions.json /opt/te-audit/final-versions.json
COPY --from=artifacts /opt/te-audit/ixte-provenance.json /opt/te-audit/ixte-provenance.json
ENV PIP_CONSTRAINT=/opt/te-audit/constraints.txt
# Metadata checks only: image builds must not initialize GPU drivers.
RUN python -c 'import sys, importlib.metadata as m; assert sys.version_info[:3] == (3,12,13); expected={"torch":"2.10.0+corex.4.5.0.20260804","torchvision":"0.25.0+corex.4.5.0.20260804","flagtree":"0.7.1a1+iluvatar3.6","flag_gems":"5.4.0","flagcx":"0.14.0+corex4.5.0","numpy":"1.26.4"}; actual={k:m.version(k) for k in expected}; assert actual == expected, actual; print(actual)'
# Make FlagTree available to direct Python processes, as on Hygon. Execute the
# original config scripts by absolute path so their prefix detection stays valid.
RUN python -c 'exec('"'"'from pathlib import Path\nimport site\nimport sys\n(Path(site.getsitepackages()[0]) / "flagtree_runtime.pth").write_text("/opt/flagtree\\n")\nfor name in ("python3-config", "python3.12-config"):\n    source = Path(sys.base_prefix) / "bin" / name\n    assert source.is_file(), source\n    target = Path("/usr/local/bin") / name\n    target.write_text("#!/bin/sh\\nexec " + str(source) + \'"'"' "$@"\\n\'"'"')\n    target.chmod(0o755)\n'"'"')' \
    && python3-config --includes && python3-config --extension-suffix
WORKDIR /workspace
CMD ["bash"]

FROM base AS release
LABEL io.flagos.image.stage="release"
RUN python -c 'import hashlib,pathlib; p=pathlib.Path("/opt/te-audit/wheelhouse/transformer_engine-2.17.0+fl.0.3.0-py3-none-any.whl"); assert hashlib.sha256(p.read_bytes()).hexdigest()=="d6abfe990299995bafcc70cbd4c28f9cad176f34064e4b887c3e0c051e74d778"'
RUN python -m pip install --no-index --find-links /opt/te-audit/wheelhouse \
        annotated-types==0.8.0 \
        einops==0.8.2 \
        importlib-metadata==10.1.0 \
        ixte==5.0.0+corex.4.5.0.20260804 \
        ml-dtypes==0.5.4 \
        nvdlfw-inspect==0.2.2 \
        onnx==1.23.2 \
        onnx-ir==1.1.0 \
        onnxscript==0.7.2 \
        protobuf==7.36.2 \
        pydantic==2.14.1 \
        pydantic-core==2.50.1 \
        transformer-engine==2.17.0+fl.0.3.0 \
        typing-inspection==0.4.4 \
        wheel==0.48.0 \
        zipp==4.1.2 \
    && python -m pip check

FROM release AS dev
LABEL io.flagos.image.stage="dev"
# Compiler, Python headers, Git, CMake, Ninja and pybind11 are inherited.
RUN command -v gcc g++ make git cmake ninja python3-config
RUN python -m pip install --no-index --find-links /opt/te-audit/wheelhouse \
        astroid==3.3.11 \
        black==24.4.2 \
        cfgv==3.5.0 \
        clang-format==18.1.6 \
        click==8.5.1 \
        cpplint==1.6.0 \
        dill==0.4.1 \
        distlib==0.4.4 \
        identify==2.6.20 \
        isort==6.1.0 \
        mccabe==0.7.0 \
        mypy-extensions==1.2.0 \
        nodeenv==1.12.0 \
        pathspec==1.1.1 \
        platformdirs==4.14.5 \
        pre-commit==4.7.0 \
        pylint==3.3.4 \
        python-discovery==1.9.5 \
        tomlkit==0.15.2 \
        vermin==1.9.2 \
        virtualenv==21.15.6 \
    && python -m pip check

FROM dev AS ci
LABEL io.flagos.image.stage="ci"
RUN python -m pip install --no-index --find-links /opt/te-audit/wheelhouse \
        absl-py==2.5.1 \
        certifi==2026.10.23 \
        charset-normalizer==3.5.2 \
        coverage==7.17.0 \
        expecttest==0.3.0 \
        flatbuffers==25.12.19 \
        grpcio==1.85.0 \
        hf-xet==1.7.1 \
        huggingface-hub==0.36.2 \
        idna==3.20 \
        iniconfig==2.3.1 \
        markdown==3.11.2 \
        onnxruntime==1.31.1 \
        onnxruntime-extensions==0.15.2 \
        pluggy==1.7.0 \
        psutil==7.2.2 \
        pygments==2.21.0 \
        pytest==8.4.2 \
        pytest-cov==7.1.0 \
        regex==2026.9.29 \
        requests==2.34.2 \
        safetensors==0.8.0 \
        scipy==1.15.3 \
        six==1.17.0 \
        tensorboard==2.21.0 \
        tensorboard-data-server==0.7.2 \
        tokenizers==0.22.2 \
        torchao==0.13.0 \
        tqdm==4.70.1 \
        transformers==4.57.6 \
        urllib3==2.8.0 \
        werkzeug==3.1.9 \
    && python -m pip check
RUN python -m coverage --version \
    && python -c 'import importlib.metadata as m; assert m.version("transformer_engine")=="2.17.0+fl.0.3.0"; assert m.version("pytest")=="8.4.2"; import triton; assert triton.__version__=="3.6.0"'
# CI installs the checked-out revision in setup_iluvatar.sh. Do not copy a Git
# checkout into the image or change test exclusions to hide compatibility failures.
