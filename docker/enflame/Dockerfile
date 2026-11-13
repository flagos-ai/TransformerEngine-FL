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
ARG FLAG_GEMS_WHEEL_URL=https://files.pythonhosted.org/packages/08/50/944776a89835961de5d03c942efc58237492df9c584765e7044ed617d099/flag_gems-5.0.2-py3-none-any.whl#sha256=d7c5a6121bf2d3f7daddde43bfa7cae739de6cb5b294dcfe7ae6173045c35552
# Avoid executing the inherited accelerator startup hooks during image build.
# Isolate FlagTree's Triton files from the Triton already present in the base.
RUN python3 -S -c 'import sys,runpy; sys.path[:0]=["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages"]; sys.argv=["pip",*sys.argv[1:]]; runpy.run_module("pip",run_name="__main__")' \
        install --break-system-packages --no-cache-dir --no-deps \
        --target /opt/flagtree "${FLAGTREE_WHEEL_URL}" \
    && python3 -S -c 'import sys,runpy; sys.path[:0]=["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages"]; sys.argv=["pip",*sys.argv[1:]]; runpy.run_module("pip",run_name="__main__")' \
        install --break-system-packages --no-cache-dir --no-deps "${FLAG_GEMS_WHEEL_URL}" \
    && python3 -S -c 'import sys,runpy; sys.path[:0]=["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages"]; sys.argv=["pip",*sys.argv[1:]]; runpy.run_module("pip",run_name="__main__")' \
        install --break-system-packages --no-cache-dir \
        SQLAlchemy==2.0.48 greenlet==3.5.6 PyYAML==6.0.3 packaging==26.0 \
    && python3 -S -c 'from pathlib import Path; Path("/usr/local/lib/python3.12/dist-packages/00_flagtree_runtime.pth").write_text("import sys; sys.path.insert(0, "+repr("/opt/flagtree")+")"+chr(10))' \
    && python3 -S -c 'import sys; sys.path[:0]=["/opt/flagtree","/usr/local/lib/python3.12/dist-packages"]; import importlib.metadata as m; assert m.version("flag_gems")=="5.0.2"; assert m.version("flagtree")=="0.7.0+enflame3.6"'
# Let the standalone migration package supply the vendor TE patches. The
# inherited torch_gcu inserts its own directory ahead of dist-packages and
# otherwise shadows that package with a migration tree that has no TE patches.
# Archive stale dist-info without uninstalling overlapping Torch package files.
RUN python3 -S -c 'from pathlib import Path; import shutil; \
    site = Path("/usr/local/lib/python3.12/dist-packages"); \
    init = site / "torch_gcu/__init__.py"; source = init.read_text(); \
    old = "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"; \
    assert source.count(old) == 1, "Unexpected torch_gcu startup; review migration precedence"; \
    assert (site / "migration/patches/transformer_engine/v2_9_0/__init__.py").is_file(); \
    init.write_text(source.replace(old, "sys.path.append(os.path.dirname(os.path.abspath(__file__)))")); \
    bootstrap = site / "enflame_te_bootstrap.py"; source = bootstrap.read_text(); \
    old = "    os.environ.setdefault(\"TE_FL_PREFER\", \"reference\")" + chr(10); \
    assert source.count(old) == 1, "Unexpected inherited TE preference hook"; \
    bootstrap.write_text(source.replace(old, "")); \
    archive = Path("/opt/enflame/legacy-dist-info"); archive.mkdir(parents=True, exist_ok=True); \
    packages = [("torch", "2.9.1+cpu", "2.10.0+cpu"), ("torch_gcu", "2.9.1+4.0.20260714", "2.10.0+3.7.20260408")]; \
    assert all((site / (name + "-" + active + ".dist-info/METADATA")).is_file() and active in (site / name / "version.py").read_text() for name, stale, active in packages); \
    [shutil.move(site / (name + "-" + stale + ".dist-info"), archive) for name, stale, active in packages]'
# CI clones its own Megatron checkout. Do not inherit the unrelated workspace.
ENV PYTHONPATH=""
RUN rm -rf /workspace/Megatron-LM-FL
# Other inherited packages still have pip-check conflicts; the TE version is
# inherited from the pinned base image.
