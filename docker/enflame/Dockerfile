# Extend the user-selected TE development image; build from repository root:
# docker build -f docker/enflame/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-enflame-ci .
# manual-20260809-enflame-dev, pinned to preserve its Torch/GCU/ECCL/TE stack.
ARG ENFLAME_BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/transformerengine-fl@sha256:9f7d14da2e77229cd3284b6590bbe5e92f5f9be7e787d3b7fe762dcc9a7660eb
FROM ${ENFLAME_BASE_IMAGE} AS ci
ARG ENFLAME_BASE_IMAGE
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG PIP_INDEX_URL=https://pypi.org/simple
LABEL org.opencontainers.image.base.name="${ENFLAME_BASE_IMAGE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      io.flagos.image.stage="ci" \
      io.flagos.accelerator.backend="enflame"
ARG FLAGTREE_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-enflame/packages/flagtree/0.7.0+enflame3.6/flagtree-0.7.0+enflame3.6-cp312-cp312-linux_x86_64.whl#sha256=94b3aaf9f9aefa02f535ee5339f16c5304c52a1de5e7e749f742f7b4bf01332d
ARG FLAG_GEMS_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-enflame/packages/flag-gems/5.4.0/flag_gems-5.4.0-py3-none-any.whl#sha256=09cb175e29f4fdfc78b87df46c4500d5a9b451e12cd64dda9a7d669f522bf5e5
# Avoid executing the inherited accelerator startup hooks during image build.
# Isolate FlagTree's Triton files from the Triton already present in the base.
RUN python3 -S -c 'import sys,runpy; sys.path[:0]=["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages"]; sys.argv=["pip",*sys.argv[1:]]; runpy.run_module("pip",run_name="__main__")' \
        install --break-system-packages --no-cache-dir --no-deps \
        --target /opt/flagtree "${FLAGTREE_WHEEL_URL}" \
    && python3 -S -c 'import sys,runpy; sys.path[:0]=["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages"]; sys.argv=["pip",*sys.argv[1:]]; runpy.run_module("pip",run_name="__main__")' \
        install --break-system-packages --no-cache-dir --no-deps "${FLAG_GEMS_WHEEL_URL}" \
    && python3 -S -c 'import sys,runpy; sys.path[:0]=["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages"]; sys.argv=["pip",*sys.argv[1:]]; runpy.run_module("pip",run_name="__main__")' \
        install --break-system-packages --no-cache-dir \
        SQLAlchemy==2.0.55 greenlet==3.5.6 PyYAML==6.0.1 \
    && python3 -S -c 'from pathlib import Path; Path("/usr/local/lib/python3.12/dist-packages/00_flagtree_runtime.pth").write_text("import sys; sys.path.insert(0, "+repr("/opt/flagtree")+")"+chr(10))' \
    && python3 -S -c 'import sys; sys.path[:0]=["/opt/flagtree","/usr/local/lib/python3.12/dist-packages"]; import importlib.metadata as m; assert m.version("flag_gems")=="5.4.0"; assert m.version("flagtree")=="0.7.0+enflame3.6"'
# Inherited packages have pre-existing pip-check conflicts; do not claim that
# this image upgrades the base TE to 2.17 or provides a native scale kernel.
