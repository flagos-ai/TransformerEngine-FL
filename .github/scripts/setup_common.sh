#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON_BIN:-python3}"

cd "$WORKSPACE"
"$PYTHON" -m pip install --disable-pip-version-check \
    --no-deps --force-reinstall \
    --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/ \
    flag-gems==5.4.0

"$PYTHON" - <<'PY'
from importlib.metadata import version

installed = version("flag-gems")
if installed != "5.4.0":
    raise RuntimeError(f"Expected FlagGems 5.4.0, found {installed}")
print(f"FlagGems: {installed}")
PY

"$PYTHON" tests/pytorch/test_sanity_import.py
