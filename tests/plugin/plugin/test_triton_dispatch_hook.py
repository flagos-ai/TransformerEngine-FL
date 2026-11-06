# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

from transformer_engine.plugin.core.triton_dispatch import hook
from transformer_engine.plugin.core.triton_dispatch.manifest import TARGET_APIS


TARGET = "transformer_engine.pytorch.triton.pad"


class RecordingLoader:
    def __init__(self, fail=False):
        self.created = 0
        self.executed = 0
        self.fail = fail

    def create_module(self, spec):
        self.created += 1
        return ModuleType(spec.name)

    def exec_module(self, module):
        self.executed += 1
        if self.fail:
            raise RuntimeError("loader failed")


class RecordingFinder:
    def __init__(self, loader):
        self.loader = loader
        self.calls = []

    def find_spec(self, fullname, path=None, target=None):
        self.calls.append((fullname, path, target))
        return ModuleSpec(fullname, self.loader, origin="editable://target")


def test_finder_delegates_to_later_finder_and_wraps_loader(monkeypatch):
    dispatch_finder = hook.TritonDispatchFinder()
    loader = RecordingLoader()
    downstream = RecordingFinder(loader)
    path = ["editable-path"]
    monkeypatch.setattr(hook.sys, "meta_path", [dispatch_finder, downstream])

    spec = dispatch_finder.find_spec(TARGET, path, None)

    assert downstream.calls == [(TARGET, path, None)]
    assert spec.origin == "editable://target"
    assert isinstance(spec.loader, hook.TritonDispatchLoader)
    assert spec.loader.create_module(spec).__name__ == TARGET
    assert loader.created == 1


def test_non_target_module_is_transparent(monkeypatch):
    dispatch_finder = hook.TritonDispatchFinder()
    downstream = RecordingFinder(RecordingLoader())
    monkeypatch.setattr(hook.sys, "meta_path", [dispatch_finder, downstream])

    assert dispatch_finder.find_spec("unrelated.module") is None
    assert downstream.calls == []


def test_reload_rejected_before_delegation(monkeypatch):
    dispatch_finder = hook.TritonDispatchFinder()
    loader = RecordingLoader()
    downstream = RecordingFinder(loader)
    module = ModuleType(TARGET)
    monkeypatch.setattr(hook.sys, "meta_path", [dispatch_finder, downstream])

    with pytest.raises(ImportError, match="Reload is not supported"):
        dispatch_finder.find_spec(TARGET, target=module)

    assert downstream.calls == []
    assert loader.executed == 0


def test_loader_failure_leaves_no_dispatch_state(monkeypatch):
    loader = RecordingLoader(fail=True)
    wrapped = hook.TritonDispatchLoader(loader)
    module = ModuleType(TARGET)
    module.__spec__ = ModuleSpec(TARGET, wrapped)
    remembered = []
    monkeypatch.setattr(hook, "remember_natives", lambda natives: remembered.append(natives))

    with pytest.raises(RuntimeError, match="loader failed"):
        wrapped.exec_module(module)

    assert remembered == []
    assert not hasattr(module, hook._PROCESSED_MARKER)


def test_manifest_mismatch_is_reported():
    module = ModuleType(TARGET)
    api_name = TARGET_APIS[TARGET][0]

    def public_api():
        return None

    public_api.__module__ = TARGET
    setattr(module, api_name, public_api)

    def extra_api():
        return None

    extra_api.__module__ = TARGET
    module.extra_api = extra_api

    with pytest.raises(ImportError, match="unexpected=\\['extra_api'\\]"):
        hook._build_replacements(module)


def test_install_checks_all_preloaded_targets_before_mutation(monkeypatch):
    meta_path = []
    monkeypatch.setattr(hook.sys, "meta_path", meta_path)
    monkeypatch.setitem(hook.sys.modules, TARGET, ModuleType(TARGET))

    with pytest.raises(ImportError, match="after target modules were loaded"):
        hook.install_finder()

    assert meta_path == []


def test_install_is_idempotent(monkeypatch):
    meta_path = []
    monkeypatch.setattr(hook.sys, "meta_path", meta_path)
    for module_name in TARGET_APIS:
        monkeypatch.delitem(hook.sys.modules, module_name, raising=False)

    first = hook.install_finder()
    second = hook.install_finder()

    assert first is second
    assert meta_path == [first]


def test_registration_failure_rolls_back_captured_natives(monkeypatch):
    wrapped = hook.TritonDispatchLoader(RecordingLoader())
    module = ModuleType(TARGET)
    module.__spec__ = ModuleSpec(TARGET, wrapped)
    api_name = TARGET_APIS[TARGET][0]

    def public_api():
        return None

    public_api.__module__ = TARGET
    setattr(module, api_name, public_api)
    remembered = []
    forgotten = []
    monkeypatch.setattr(hook, "remember_natives", lambda natives: remembered.append(natives))
    monkeypatch.setattr(hook, "forget_natives", lambda natives: forgotten.append(natives))
    monkeypatch.setattr(
        hook,
        "ensure_native_registered",
        lambda manager: (_ for _ in ()).throw(RuntimeError("registration failed")),
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        wrapped.exec_module(module)

    assert forgotten == remembered
    assert not hasattr(module, hook._PROCESSED_MARKER)
