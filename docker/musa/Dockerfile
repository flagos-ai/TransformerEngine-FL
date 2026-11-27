# Build from the repository root:
# docker build --target ci -f docker/musa/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformerengine-fl:20261127-musa-dev .
ARG MUSA_BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/transformerengine-fl@sha256:285810557f4fe9d6c64cce19d6b644631ad42966f4beba77742f95932232a8cf
FROM ${MUSA_BASE_IMAGE} AS ci

ARG MUSA_BASE_IMAGE
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=20261127-musa-dev
ARG SOURCE_URL=https://github.com/flagos-ai/TransformerEngine-FL

LABEL org.opencontainers.image.title="transformer-engine-fl" \
      org.opencontainers.image.description="MooreThreads MUSA S5000 clean CI environment" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${MUSA_BASE_IMAGE}" \
      io.flagos.accelerator.backend="musa" \
      io.flagos.image.stage="ci"

ENV PLATFORM=mthreads \
    MUSA_HOME=/usr/local/musa \
    CUDA_HOME=/usr/local/musa \
    LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib \
    TORCH_DEVICE_BACKEND_AUTOLOAD=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Remove all legacy sitecustomize hooks (both TE monkey-patches and PyTorch
# startup wrappers), Python operator fallbacks, stale editable finder hooks
# (keeping transformer_engine dist-info so importlib.metadata.version("transformer_engine")
# resolves for source checkouts), and stale Triton/FlagGems runtime caches
# containing broken libmusa.so.4 links.
# Keep /root/.flagtree, /root/.flaggems/config_cache, and /root/.triton/{json,pybind11}
# because they contain offline MThreads LLVM/Triton compiler toolchains.
RUN rm -rf \
        /etc/python3.10/sitecustomize.py \
        /usr/lib/python3.10/sitecustomize.py \
        /usr/local/lib/python3.10/dist-packages/transformer_engine_musa \
        /usr/local/lib/python3.10/dist-packages/transformer_engine_musa_torch.py \
        /usr/local/lib/python3.10/dist-packages/__editable__.transformer_engine-2.14.0+29b5a10e.pth \
        /usr/local/lib/python3.10/dist-packages/__editable___transformer_engine_2_14_0_29b5a10e_finder.py \
        /root/.triton/cache \
        /root/.triton/autotune \
        /root/.flaggems/code_cache \
        /root/.flaggems/oplist.log \
        /root/.cache/pip \
        /tmp/* \
    && mkdir -p /root/.triton /root/.flaggems /workspace

# Install clean native extension loaders only (no sitecustomize or Python fallbacks).
RUN set -eu \
    && SITE=/usr/local/lib/python3.10/dist-packages \
    && PKG="${SITE}/transformer_engine_musa" \
    && mkdir -p "${PKG}/pytorch" \
    && printf '%s\n' \
        '"""Container-only compatibility package for MooreThreads TransformerEngine MUSA."""' \
        '' \
        'import ctypes' \
        'import os' \
        '' \
        '_BASE = "/opt/mt_transformerengine_runtime"' \
        '_LIB = os.path.join(_BASE, "libtransformer_engine.so")' \
        'if os.path.exists(_LIB):' \
        '    ctypes.CDLL(_LIB, mode=ctypes.RTLD_GLOBAL)' \
        > "${PKG}/__init__.py" \
    && printf '%s\n' \
        '"""Reserved MUSA namespace; no attention proxy or operator fallback installed."""' \
        > "${PKG}/pytorch/__init__.py" \
    && printf '%s\n' \
        '"""Container-only pure loader for MooreThreads TransformerEngine MUSA torch extension."""' \
        '' \
        'import ctypes' \
        'import glob' \
        'import importlib.util' \
        'import os' \
        'import sys' \
        '' \
        '_BASE = "/opt/mt_transformerengine_runtime"' \
        '_LIB = os.path.join(_BASE, "libtransformer_engine.so")' \
        'if os.path.exists(_LIB):' \
        '    ctypes.CDLL(_LIB, mode=ctypes.RTLD_GLOBAL)' \
        '' \
        '_candidates = sorted(glob.glob(os.path.join(_BASE, "transformer_engine_torch*.so")))' \
        'if not _candidates:' \
        '    raise ImportError(f"Cannot find transformer_engine_torch extension under {_BASE}")' \
        '' \
        '_saved = sys.modules.get("transformer_engine_torch")' \
        '_spec = importlib.util.spec_from_file_location("transformer_engine_torch", _candidates[0])' \
        '_native = importlib.util.module_from_spec(_spec)' \
        '_spec.loader.exec_module(_native)' \
        'if _saved is None:' \
        '    if sys.modules.get("transformer_engine_torch") is _native:' \
        '        sys.modules.pop("transformer_engine_torch", None)' \
        'elif _saved is not _native:' \
        '    sys.modules["transformer_engine_torch"] = _saved' \
        '' \
        '' \
        'def __getattr__(name):' \
        '    return getattr(_native, name)' \
        '' \
        '' \
        'def __dir__():' \
        '    return sorted(set(globals()) | set(dir(_native)))' \
        '' \
        '' \
        'for _name in dir(_native):' \
        '    if not _name.startswith("__"):' \
        '        globals()[_name] = getattr(_native, _name)' \
        > "${SITE}/transformer_engine_musa_torch.py" \
    && chmod 0644 \
        "${PKG}/__init__.py" \
        "${PKG}/pytorch/__init__.py" \
        "${SITE}/transformer_engine_musa_torch.py" \
    && test -s "${PKG}/__init__.py" \
    && test -s "${PKG}/pytorch/__init__.py" \
    && test -s "${SITE}/transformer_engine_musa_torch.py"

# Metadata and static checks only: image builds must not initialize GPU drivers.
RUN set -eu \
    && printf '%s\n' \
        'import importlib.metadata as m' \
        'import importlib.util' \
        'from pathlib import Path' \
        'import py_compile' \
        'import re' \
        '' \
        'assert not Path("/etc/python3.10/sitecustomize.py").exists()' \
        'assert not Path("/usr/lib/python3.10/sitecustomize.py").exists()' \
        '' \
        'for py_file in (' \
        '    "/usr/local/lib/python3.10/dist-packages/transformer_engine_musa_torch.py",' \
        '    "/usr/local/lib/python3.10/dist-packages/transformer_engine_musa/__init__.py",' \
        '    "/usr/local/lib/python3.10/dist-packages/transformer_engine_musa/pytorch/__init__.py",' \
        '):' \
        '    py_compile.compile(py_file, doraise=True)' \
        '' \
        'assert importlib.util.find_spec("transformer_engine_musa") is not None' \
        'assert importlib.util.find_spec("transformer_engine_musa.pytorch") is not None' \
        'assert importlib.util.find_spec("transformer_engine_musa_torch") is not None' \
        '' \
        'triton_init = Path("/usr/local/lib/python3.10/dist-packages/triton/__init__.py").read_text()' \
        'triton_ver = re.search(r"__version__\s*=\s*[\x22\x27]([^\x22\x27]+)", triton_init)' \
        'assert triton_ver and triton_ver.group(1) == "3.1.0", triton_ver' \
        '' \
        'runtime_dir = Path("/opt/mt_transformerengine_runtime")' \
        'assert (runtime_dir / "libtransformer_engine.so").is_file()' \
        'assert list(runtime_dir.glob("transformer_engine_torch*.so"))' \
        'assert Path("/root/.flagtree/mthreads/mthreadsTritonPlugin.so").is_file()' \
        'assert Path("/root/.flaggems/config_cache/TunedConfig_mthreads_triton_3_1.db").is_file()' \
        'assert not Path("/root/.triton/cache").exists()' \
        '' \
        'required = {' \
        '    "transformer_engine": "2.14.0+29b5a10e",' \
        '    "torch": "2.5.0",' \
        '    "torch_musa": "2.5.0",' \
        '    "flag_gems": "4.2.0",' \
        '    "flagtree": "0.4.0",' \
        '    "pytest": "8.2.1",' \
        '    "expecttest": "0.3.0",' \
        '    "nvdlfw-inspect": "0.2.2",' \
        '    "onnxruntime": "1.23.2",' \
        '    "onnxruntime-extensions": "0.15.2",' \
        '    "coverage": "7.13.1",' \
        '    "pytest-cov": "7.0.0",' \
        '    "six": "1.17.0",' \
        '    "regex": "2025.9.1",' \
        '}' \
        'actual = {name: m.version(name) for name in required}' \
        'assert actual == required, (actual, required)' \
        'print("Verified MUSA CI image dependencies:", actual)' \
        | python3 -

WORKDIR /workspace
CMD ["bash"]
