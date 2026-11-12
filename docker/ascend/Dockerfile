# Build from the repository root with --target ci.
ARG NPU_IMAGE=harbor.baai.ac.cn/flagos-dev/transformerengine-fl@sha256:f143e7f715c939b7fa000b2a06f0b58a3272a99d6ed75c5b50eb2eb6777d77fe
ARG ASCEND_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c@sha256:274909c4a51f99d7aef06349555cd8481e068f40b3213c3facc2d1c90cf155e4
FROM ${NPU_IMAGE} AS npu
FROM ${ASCEND_BASE_IMAGE} AS base
ENV TE_FL_SKIP_CUDA=1 NVTE_FRAMEWORK=pytorch NVTE_WITH_CUDA=0 \
    NVTE_WITH_MACA=0 NVTE_WITH_NCCL_EP=0 TE_WITH_NCCL=0
COPY --from=npu /usr/local/python3.11.15/lib/python3.11/site-packages/transformer_engine_npu /flagos/lib/python3.11/site-packages/transformer_engine_npu
COPY --from=npu /usr/local/python3.11.15/lib/python3.11/site-packages/transformer_engine_npu-2.13.0.dist-info /flagos/lib/python3.11/site-packages/transformer_engine_npu-2.13.0.dist-info
RUN mkdir -p /opt/te-image && printf '%s\n' \
    'torch==2.10.0+cpu' 'torch-npu==2.10.0' 'torchaudio==2.10.0+cpu' \
    'torchvision==0.25.0+cpu' 'numpy==1.26.4' 'flag_gems==5.4.0' \
    'flagtree==0.7.0+ascend3.5' 'flagcx==0.14.0+cann9.0.0.910c' \
    > /opt/te-image/constraints.txt
ENV PIP_CONSTRAINT=/opt/te-image/constraints.txt PIP_PROGRESS_BAR=off
WORKDIR /workspace
CMD ["bash"]

FROM base AS release
ARG TE_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-thead/packages/transformer-engine/2.17.0+fl.0.3.0/transformer_engine-2.17.0+fl.0.3.0-py3-none-any.whl#sha256=d6abfe990299995bafcc70cbd4c28f9cad176f34064e4b887c3e0c051e74d778
RUN python -m pip install --no-cache-dir \
    pydantic==2.14.3 einops==0.8.2 importlib-metadata==10.2.0 \
    onnx==1.23.2 onnxscript==0.7.2 nvdlfw-inspect==0.2.2 \
    attrs==26.1.0 decorator==5.3.1 scipy==1.17.1 cloudpickle==3.1.2 tornado==6.6.1 \
    "${TE_WHEEL_URL}"

FROM release AS dev
RUN python -m pip install --no-cache-dir \
    build==1.6.1 wheel==0.48.0 black==24.4.2 clang-format==18.1.6 \
    pre-commit==4.7.0 vermin==1.9.2 cpplint==1.6.0 pylint==3.3.4 \
    six==1.17.0 regex==2026.11.9 psutil==7.2.2 tensorboard==2.21.0

FROM dev AS ci
RUN python -m pip install --no-cache-dir \
    pytest==8.2.1 expecttest==0.3.0 coverage==7.17.2 pytest-cov==7.1.0 \
    onnxruntime==1.31.1 onnxruntime-extensions==0.15.2 \
    transformers==5.22.2 torchao==0.13.0
