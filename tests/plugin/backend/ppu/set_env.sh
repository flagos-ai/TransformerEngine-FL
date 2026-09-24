#!/usr/bin/env bash
set -euo pipefail
# PPU exposes its accelerator through the vendor torch.cuda compatibility API.
PPU_TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TE_PATH="$(cd "$PPU_TEST_DIR/../../../.." && pwd)"
# Fixed runtime options are supplied by ppu.yml container_options.
case ":${PYTHONPATH:-}:" in
    *":$TE_PATH:"*) ;;
    *) PYTHONPATH="$TE_PATH${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export PYTHONPATH
export XML_LOG_DIR="${XML_LOG_DIR:-$TE_PATH/logs/ppu}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$XML_LOG_DIR"
