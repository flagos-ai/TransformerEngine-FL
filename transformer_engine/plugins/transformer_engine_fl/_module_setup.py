# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Module setup for transformer_engine_fl plugin system.

This module handles the registration of transformer_engine_fl modules in sys.modules
with both full and short names to support relative imports in backends.
"""

import sys
from pathlib import Path


def setup_module_aliases():
    """
    Register transformer_engine_fl modules under both full and short names.

    This allows backends to use relative imports like:
        from ...ops import TEFLBackendBase
        from ...types import OpImpl, BackendImplKind

    And ensures they work correctly regardless of how the module is imported.
    """
    # Get the current package
    current_package = sys.modules.get("transformer_engine.plugins.transformer_engine_fl")
    if current_package is None:
        return

    # Register the main package under short name
    sys.modules["transformer_engine_fl"] = current_package

    # List of submodules to register
    submodule_names = [
        "ops",
        "logger",
        "types",
        "logger_manager",
        "policy",
        "operator_registry",
        "registry",
        "discovery",
    ]

    # Register each submodule under short name
    for name in submodule_names:
        full_name = f"transformer_engine.plugins.transformer_engine_fl.{name}"
        short_name = f"transformer_engine_fl.{name}"

        if full_name in sys.modules and short_name not in sys.modules:
            sys.modules[short_name] = sys.modules[full_name]

    # Register backends package
    backends_full = "transformer_engine.plugins.transformer_engine_fl.backends"
    backends_short = "transformer_engine_fl.backends"
    if backends_full in sys.modules and backends_short not in sys.modules:
        sys.modules[backends_short] = sys.modules[backends_full]

    # Register parent plugins package if needed
    if "transformer_engine.plugins" not in sys.modules:
        import types
        plugins_dir = Path(__file__).parent.parent
        plugins_pkg = types.ModuleType("transformer_engine.plugins")
        plugins_pkg.__path__ = [str(plugins_dir)]
        sys.modules["transformer_engine.plugins"] = plugins_pkg


def register_as_transformer_engine_torch():
    """
    Register the tefl module as transformer_engine_torch.

    This provides backward compatibility with code that expects
    transformer_engine_torch to be available.
    """
    # Only register if not already present
    if "transformer_engine_torch" in sys.modules:
        return

    try:
        from .ops import get_tefl_module
        tefl_module = get_tefl_module()
        sys.modules["transformer_engine_torch"] = tefl_module
    except Exception as e:
        import traceback
        print(f"[TEFL Setup] Warning: Could not register transformer_engine_torch: {e}")
        traceback.print_exc()

        # Create a minimal placeholder module to avoid import errors
        # This allows the system to at least import without crashing
        import types
        placeholder = types.ModuleType("transformer_engine_torch")
        placeholder.__doc__ = "Placeholder module - TEFL backend not available"
        sys.modules["transformer_engine_torch"] = placeholder
