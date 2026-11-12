# syntax=docker/dockerfile:1
# Build from the repository root:
# docker build -f docker/metax/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-metax-ci .
ARG METAX_BASE_IMAGE=harbor.baai.ac.cn/flagos-base/flagos-base-metax-maca3.7.2.1:2.3.0@sha256:e2682f632e4dda707bbb4d8900b9c2d75733f056b701171dce6bd2ea417f4166
# Optional: reuse a previously validated dependency image by digest.
# The default still builds every dependency from the MACA base above.
ARG METAX_RUNTIME_IMAGE=dependencies
FROM ${METAX_BASE_IMAGE} AS dependencies
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
        'torch==2.8.0+metax3.7.2.0' \
        'torchvision==0.15.1+metax3.7.2.0' \
        'triton==3.0.0+metax3.7.2.0' \
        'flash-attn==2.6.3+metax3.7.2.0torch2.8' \
        'apex==0.1+metax3.7.2.0' \
        'numpy==1.26.4' \
        'pytest==8.2.1' \
        > /opt/te-image/constraints.txt \
    && printf '%s\n' \
        'torch @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/torch/2.8.0+metax3.7.2.0/torch-2.8.0+metax3.7.2.0-cp312-cp312-linux_x86_64.whl#sha256=8bb172671d6a44c741f671d62f5d0f84c37a884b918af7276c9fbde8638a3d26' \
        'triton @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/triton/3.0.0+metax3.7.2.0/triton-3.0.0+metax3.7.2.0-cp312-cp312-linux_x86_64.whl#sha256=4ea454ac5df9bcd0ac856207bbfffb3fb0a398f73fb7cfaf4cb21712a3041162' \
        'torchvision @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/torchvision/0.15.1+metax3.7.2.0/torchvision-0.15.1+metax3.7.2.0-cp312-cp312-linux_x86_64.whl#sha256=40594b08694cbbe3232ee4780f1cce162389f2a8702b559bdc7b00e4b60d4748' \
        'flash-attn @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/flash-attn/2.6.3+metax3.7.2.0torch2.8/flash_attn-2.6.3+metax3.7.2.0torch2.8-cp312-cp312-linux_x86_64.whl#sha256=99958f6407e53ce1d650050074b288a574213e3c33b07addeadd5bf1603f88b9' \
        'apex @ https://resource.flagos.net/repository/flagos-pypi-metax/packages/apex/0.1+metax3.7.2.0/apex-0.1+metax3.7.2.0-cp312-cp312-linux_x86_64.whl#sha256=519ec7c50068852714edb3382894f7fb9ffa56c5ff64ca54e310faf0cb47b135' \
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
# Keep the native extension and the MACA PyTorch wheel on torch 2.8.
RUN python -m pip install --no-cache-dir --no-deps --target "${TE_METAX_TE_HOME}" "${METAX_TE_WHEEL}" \
    && python -m pip install --no-cache-dir --index-url "${PYPI_INDEX}" "${TE_FL_WHEEL}" \
    && python -m pip check \
    && python -m pip freeze --all > /opt/te-image/requirements.lock \
    && python -c 'import importlib.metadata as m; from pathlib import Path; assert m.version("torch") == "2.8.0+metax3.7.2.0"; assert m.version("transformer_engine") == "2.17.0+fl.0.3.0"; assert list(Path("/opt/metax-te/transformer_engine").glob("transformer_engine_torch*.so"))'

FROM ${METAX_RUNTIME_IMAGE} AS runtime
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# TE-FL attention imports the legacy transformer_engine_metax namespace,
# while MetaX TE >= 2.13 ships transformer_engine/. Provide the actual vendor
# Python modules under the legacy namespace without overwriting TE-FL.
# Both the native vendor extension and the runtime use the torch 2.8 ABI.
RUN python - <<'PY'
from pathlib import Path
import re
import shutil
import sysconfig
home=Path('/opt/metax-te')
site=Path(sysconfig.get_paths()['purelib'])
dest=site/'transformer_engine_metax'
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(home/'transformer_engine',dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.so','__pycache__','csrc','include'))
for p in dest.rglob('*.py'):
 s=p.read_text()
 s=re.sub(r'\btransformer_engine\b','transformer_engine_metax',s)
 s=s.replace('transformer_engine_torch','transformer_engine_torch_metax')
 p.write_text(s)
# Keep the vendor root lazy so backend discovery does not import PyTorch
# attention while TE-FL itself is still initializing.
(dest/'__init__.py').write_text('__version__ = "2.13.0"\n')
p=dest/'pytorch/__init__.py'
p.write_text(p.read_text().replace('load_framework_extension("torch")', ''))
p=dest/'common/__init__.py'
s=p.read_text()
marker = 'if "NVTE_PROJECT_BUILDING" not in os.environ'
assert marker in s, 'Unexpected vendor core initialization layout'
s=s.split(marker)[0]
p.write_text(s)
# The wheel is isolated rather than registered as transformer_engine_metax
# in pip. Skip its normal distribution checks and automatic framework load;
# the extension bridge below owns loading, without replacing TE-FL dispatch.
with (dest/'common/__init__.py').open('a') as f:
 f.write('\n@functools.lru_cache(maxsize=None)\ndef get_te_path():\n    return Path("/opt/metax-te")\n')
(site/'transformer_engine_torch_metax.py').write_text('''import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import torch
pkg = Path("/opt/metax-te/transformer_engine")
ctypes.CDLL(str(pkg / "libtransformer_engine.so"), mode=ctypes.RTLD_GLOBAL)
owner = sys.modules.pop("transformer_engine_torch", None)
flags = sys.getdlopenflags()
try:
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
    spec = importlib.util.spec_from_file_location("transformer_engine_torch", str(next(pkg.glob("transformer_engine_torch*.so"))))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
finally:
    sys.setdlopenflags(flags)
    if owner is not None:
        sys.modules["transformer_engine_torch"] = owner
    else:
        sys.modules.pop("transformer_engine_torch", None)
sys.modules[__name__] = native
''')
# The original shared library lives in the original package directory.
(dest/'libtransformer_engine.so').symlink_to(home/'transformer_engine/libtransformer_engine.so')
# v2.17 adds padding controls absent from this vendor forward signature.
# Only default (unpadded) semantics can be forwarded to the older API.
with (dest/'pytorch/attention/dot_product_attention/backends.py').open('a') as f:
 f.write('''
class _LegacyFlashAttention(FlashAttention):
    def forward(self, *args, pad_between_seqs=False, cu_seqlens_q_padded=None,
                cu_seqlens_kv_padded=None, **kwargs):
        if pad_between_seqs or cu_seqlens_q_padded is not None or cu_seqlens_kv_padded is not None:
            raise NotImplementedError("MetaX TE 2.13 does not support padded sequence controls")
        return super().forward(*args, **kwargs)

FlashAttention = _LegacyFlashAttention
''')

assert (dest/'pytorch/attention/dot_product_attention/backends.py').is_file()
print('Installed vendor attention namespace:', dest)
PY

WORKDIR /workspace
CMD ["bash"]
