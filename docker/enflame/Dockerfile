# Build from repository root:
# docker build --target ci -f docker/enflame/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-enflame-ci .
# Audited base tag: flagos-runtime-enflame-tops1.10.6:2.3.0
ARG ENFLAME_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-enflame-tops1.10.6@sha256:69dabebaf536625255183c389c53f2cf6c2de0727b0989140dff4451941d301a
FROM ${ENFLAME_BASE_IMAGE} AS base
ARG ENFLAME_BASE_IMAGE
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=2.17.0-enflame-ci
LABEL org.opencontainers.image.title="transformer-engine-fl" \
      org.opencontainers.image.description="Enflame GCU TOPS 1.10.6 TE environment" \
      org.opencontainers.image.source="https://github.com/flagos-ai/TransformerEngine-FL" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${ENFLAME_BASE_IMAGE}" \
      io.flagos.accelerator.backend="enflame" io.flagos.image.stage="base"
ENV VIRTUAL_ENV=/flagos \
    PATH=/flagos/bin:${PATH} \
    TORCH_GCU_ENABLE_INT64_AND_UINT64=1 ENABLE_I64_CHECK=0 \
    TORCHGCU_INDUCTOR_ENABLE=1 ENFLAME_PT_OP_DEBUG_CONFIG=fallback_cpu=all_ops \
    TE_FL_SKIP_CUDA=1 TE_FL_PREFER=reference NVTE_FRAMEWORK=pytorch \
    NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1 \
    NVTE_UnfusedDPA_Emulate_FP8=1 PLATFORM=enflame
# Pin the complete audited Python environment, preserving the vendor stack.
# torch_gcu requires black==26.3.1; repository hooks use isolated environments.
RUN mkdir -p /opt/te-image \
    && printf '%s\n' \
        absl-py==2.5.1 \
        accelerate==1.15.0 \
        aiohappyeyeballs==2.7.1 \
        aiohttp==3.14.5 \
        aiosignal==1.4.0 \
        annotated-types==0.8.0 \
        astroid==3.3.11 \
        attrs==26.1.0 \
        black==26.3.1 \
        build==1.6.1 \
        certifi==2026.10.23 \
        cfgv==3.5.0 \
        charset-normalizer==3.5.2 \
        clang-format==18.1.6 \
        click==8.5.0 \
        coloredlogs==15.0.1 \
        compressed-tensors==0.13.0 \
        coverage==7.17.2 \
        cpplint==1.6.0 \
        datasets==3.6.0 \
        dill==0.3.8 \
        distlib==0.4.4 \
        einops==0.8.2 \
        enflame-modelopt==3.6.20260615+torch.2.11.0 \
        expecttest==0.1.3 \
        filelock==4.2.4 \
        flag_gems==5.4.0 \
        flagcx==0.14.0+tops1.10.6 \
        flagtree==0.7.0+enflame3.6 \
        flash_attn==2.7.2+torch.2.11.0.gcu.3.8.20260706 \
        flatbuffers==25.12.19 \
        frozenlist==1.8.0 \
        fsspec==2025.3.0 \
        greenlet==3.5.6 \
        grpcio==1.85.1 \
        hf-xet==1.7.0 \
        huggingface_hub==0.36.2 \
        humanfriendly==10.0 \
        identify==2.6.20 \
        idna==3.20 \
        importlib_metadata==10.2.0 \
        iniconfig==2.3.1 \
        isort==6.1.0 \
        Jinja2==3.1.6 \
        lark==1.1.4 \
        lazy-importer==0.1.0 \
        loguru==0.7.3 \
        looseversion==1.3.0 \
        Markdown==3.11.3 \
        MarkupSafe==3.0.4 \
        mccabe==0.7.0 \
        ml_dtypes==0.5.4 \
        mpmath==1.3.0 \
        multidict==7.2.1 \
        multiprocess==0.70.16 \
        mypy_extensions==1.1.0 \
        networkx==3.7 \
        ninja==1.13.0 \
        nodeenv==1.12.0 \
        numpy==1.26.4 \
        nvdlfw_inspect==0.2.2 \
        onnx==1.17.0 \
        onnx-ir==1.1.0 \
        onnxconverter-common==1.16.0 \
        onnxoptimizer==0.3.13 \
        onnxruntime==1.17.0 \
        onnxruntime_extensions==0.15.2 \
        onnxscript==0.7.2 \
        packaging==26.3 \
        pandas==3.0.6 \
        parameterized==0.9.0 \
        pathspec==1.1.1 \
        pillow==13.0.0 \
        pip==26.2.2 \
        platformdirs==4.13.5 \
        pluggy==1.7.0 \
        pre_commit==4.7.0 \
        propcache==0.5.4 \
        protobuf==7.36.2 \
        psutil==7.2.2 \
        pyarrow==26.0.0 \
        pybind11==3.0.3 \
        pydantic==2.14.1 \
        pydantic_core==2.50.1 \
        pyefml==1.10.6 \
        pylint==3.3.4 \
        pyproject_hooks==1.3.3 \
        pytest==8.2.1 \
        pytest-cov==7.1.0 \
        python-dateutil==3.0.0 \
        python-discovery==1.9.5 \
        pytokens==0.4.1 \
        PyYAML==6.0.1 \
        regex==2026.9.29 \
        requests==2.34.2 \
        safetensors==0.8.0 \
        sentencepiece==0.2.2 \
        setuptools==81.0.0 \
        setuptools-scm==9.2.2 \
        six==1.17.0 \
        SQLAlchemy==2.0.55 \
        sympy==1.14.0 \
        tensorboard==2.21.0 \
        tensorboard-data-server==0.7.2 \
        tokenizers==0.22.2 \
        tomlkit==0.15.2 \
        tops_extension==3.6.20260529+torch.2.11.0 \
        topstx==1.10.6 \
        torch==2.11.0+cpu \
        torch_gcu==2.11.0+3.8.20260713 \
        torchao==0.13.0 \
        torchaudio==2.11.0+cpu \
        torchcodec==3.8.20260624+torch.2.11.0 \
        torchvision==0.26.0+cpu \
        tqdm==4.70.1 \
        transformer_engine==2.17.0+fl.0.3.0 \
        transformers==4.57.6 \
        typing-inspection==0.4.4 \
        typing_extensions==4.16.0 \
        tzdata==2026.5 \
        urllib3==2.8.0 \
        vermin==1.9.2 \
        virtualenv==21.15.6 \
        Werkzeug==3.1.9 \
        wheel==0.48.0 \
        xxhash==4.0.2 \
        yarl==1.25.1 \
        zipp==4.1.2 > /opt/te-image/constraints.txt
ENV PIP_CONSTRAINT=/opt/te-image/constraints.txt
# Direct Python processes need FlagTree on sys.path too.
# Metadata-only checks: never initialize accelerator drivers during build.
RUN python -c 'import sys,importlib.metadata as m; sys.path.insert(0,"/opt/flagtree"); assert sys.version_info[:2]==(3,12); expected={"torch":"2.11.0+cpu","torch_gcu":"2.11.0+3.8.20260713","flagtree":"0.7.0+enflame3.6","flag_gems":"5.4.0","flagcx":"0.14.0+tops1.10.6","black":"26.3.1"}; actual={k:m.version(k) for k in expected}; assert actual==expected,actual' \
    && python -c 'import pathlib,site,sys; (pathlib.Path(site.getsitepackages()[0])/"flagtree_runtime.pth").write_text("/opt/flagtree\n"); target=pathlib.Path(sys.base_prefix)/"bin/python3-config"; assert target.is_file(); link=pathlib.Path("/flagos/bin/python3-config"); link.unlink(missing_ok=True); link.symlink_to(target)'
WORKDIR /workspace
CMD ["bash"]

FROM base AS release
ARG PIP_INDEX_URL=https://pypi.org/simple
LABEL io.flagos.image.stage="release"
ARG TE_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-thead/packages/transformer-engine/2.17.0+fl.0.3.0/transformer_engine-2.17.0+fl.0.3.0-py3-none-any.whl#sha256=d6abfe990299995bafcc70cbd4c28f9cad176f34064e4b887c3e0c051e74d778
RUN python -m pip install --no-cache-dir \
        einops==0.8.2 \
        importlib_metadata==10.2.0 \
        lazy-importer==0.1.0 \
        ml_dtypes==0.5.4 \
        nvdlfw_inspect==0.2.2 \
        onnx-ir==1.1.0 \
        onnxscript==0.7.2 \
        zipp==4.1.2 "${TE_WHEEL_URL}" \
    && python -m pip check

FROM release AS dev
ARG PIP_INDEX_URL=https://pypi.org/simple
LABEL io.flagos.image.stage="dev"
RUN bash -c 'for tool in gcc g++ git cmake ninja python3-config; do command -v "$tool" || exit 1; done' \
    && python -m pip install --no-cache-dir \
        absl-py==2.5.1 \
        astroid==3.3.11 \
        build==1.6.1 \
        cfgv==3.5.0 \
        clang-format==18.1.6 \
        cpplint==1.6.0 \
        distlib==0.4.4 \
        grpcio==1.85.1 \
        identify==2.6.20 \
        isort==6.1.0 \
        Markdown==3.11.3 \
        mccabe==0.7.0 \
        nodeenv==1.12.0 \
        pre_commit==4.7.0 \
        pylint==3.3.4 \
        pyproject_hooks==1.3.3 \
        python-discovery==1.9.5 \
        tensorboard==2.21.0 \
        tensorboard-data-server==0.7.2 \
        tomlkit==0.15.2 \
        vermin==1.9.2 \
        virtualenv==21.15.6 \
        Werkzeug==3.1.9 \
        wheel==0.48.0 \
    && python -m pip check

FROM dev AS ci
ARG PIP_INDEX_URL=https://pypi.org/simple
LABEL io.flagos.image.stage="ci"
RUN python -m pip install --no-cache-dir \
        coverage==7.17.2 \
        iniconfig==2.3.1 \
        pluggy==1.7.0 \
        pytest==8.2.1 \
        pytest-cov==7.1.0 \
        torchao==0.13.0 \
    && python -m pip check \
    && python -m coverage --version \
    && python -c 'import importlib.metadata as m; assert m.version("transformer_engine")=="2.17.0+fl.0.3.0"; assert m.version("pytest")=="8.2.1"'
# CI selects the checked-out source via PYTHONPATH. The image does not patch TE
# device checks or claim that every Enflame hardware test passes.
