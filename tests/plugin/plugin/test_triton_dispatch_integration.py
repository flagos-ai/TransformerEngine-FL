# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

import importlib
import inspect
import sys

import pytest

import transformer_engine
from transformer_engine.plugin.core.manager import (
    get_default_manager,
    reset_default_manager,
)
from transformer_engine.plugin.core.triton_dispatch import is_installed
from transformer_engine.plugin.core.triton_dispatch.entry import ensure_native_registered
from transformer_engine.plugin.core.triton_dispatch.manifest import (
    TARGET_APIS,
    TARGET_OPS,
)


def test_all_manifest_apis_are_wrapped_and_registered():
    assert is_installed()
    # Natives are registered onto whichever manager is current, lazily on
    # first dispatch, so a sibling suite's reset_default_manager() leaves a
    # fresh manager with nothing registered yet. Drive that registration the
    # way an entry does rather than assuming import-time state survives.
    manager = get_default_manager()
    ensure_native_registered(manager)

    for module_name, api_names in TARGET_APIS.items():
        module = importlib.import_module(module_name)
        for api_name in api_names:
            entry = getattr(module, api_name)
            op_name = TARGET_OPS[(module_name, api_name)]
            native = manager.registry.get_implementation(op_name, "te.native")
            assert native is not None
            assert entry is not native.fn
            assert inspect.unwrap(entry) is native.fn


def test_kernel_objects_and_check_deterministic_are_not_wrapped():
    permutation = importlib.import_module("transformer_engine.pytorch.triton.permutation")
    mhc = importlib.import_module("transformer_engine.pytorch.triton.mhc")

    assert not inspect.isfunction(permutation._permute_kernel)
    assert inspect.unwrap(mhc.check_deterministic) is mhc.check_deterministic


def test_reset_default_manager_recovers_native_registration():
    module_name = "transformer_engine.pytorch.triton.pad"
    api_name = "pad_columnwise_scale_inv"
    op_name = TARGET_OPS[(module_name, api_name)]
    entry = getattr(sys.modules[module_name], api_name)

    reset_default_manager()
    manager = get_default_manager()
    assert manager.registry.get_implementation(op_name, "te.native") is None

    ensure_native_registered(manager)
    assert manager.registry.get_implementation(op_name, "te.native").fn is inspect.unwrap(entry)


def test_reload_is_rejected_before_module_execution():
    module = importlib.import_module("transformer_engine.pytorch.triton.pad")
    original = module.pad_columnwise_scale_inv

    with pytest.raises(ImportError, match="Reload is not supported"):
        importlib.reload(module)

    assert module.pad_columnwise_scale_inv is original
