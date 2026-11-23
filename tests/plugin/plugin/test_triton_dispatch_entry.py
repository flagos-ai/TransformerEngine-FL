# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from transformer_engine.plugin.core.manager import OpManager
from transformer_engine.plugin.core.policy import reset_global_policy
from transformer_engine.plugin.core.triton_dispatch import entry
from transformer_engine.plugin.core.types import BackendImplKind, OpImpl

# Every environment variable SelectionPolicy reads (plugin/core/policy.py).
_POLICY_ENV = (
    "TE_FL_PREFER",
    "TE_FL_PREFER_VENDOR",
    "TE_FL_STRICT",
    "TE_FL_DENY_VENDORS",
    "TE_FL_ALLOW_VENDORS",
    "TE_FL_PER_OP",
)


@pytest.fixture
def isolated_entry_state(monkeypatch):
    # Selection depends on the global policy, and sibling suites leave one
    # installed (test_policy_selection sets prefer=reference without
    # restoring it), which would silently change which impl wins here.
    # Startup hooks can also seed TE_FL_* inside the interpreter (Enflame's
    # enflame_te_bootstrap.pth does setdefault("TE_FL_PREFER", "reference")),
    # so unsetting them outside the process is not enough.
    for name in _POLICY_ENV:
        monkeypatch.delenv(name, raising=False)
    reset_global_policy()
    monkeypatch.setattr(entry, "_native_functions", {})
    monkeypatch.setattr(entry, "_native_generation", 0)
    monkeypatch.setattr(entry, "_registered_managers", entry.weakref.WeakKeyDictionary())
    # Keyed by op_name only, so a name reused across tests would otherwise
    # carry the previous test's resolved function. Nothing hits today because
    # a fresh manager's registry generation is still 0 when an entry reads it
    # (ensure_native_registered runs inside the entry, and the cache is only
    # written afterwards), but that is an implementation detail to not rely on.
    monkeypatch.setattr(entry, "_fast_path", {})
    yield
    reset_global_policy()


def test_entry_dispatches_native_without_call_retry(isolated_entry_state):
    manager = OpManager()

    def native(value):
        return value + 1

    entry.remember_natives({"triton_test_op": native})
    wrapped = entry.make_entry("triton_test_op", native)

    with patch.object(manager, "ensure_initialized"):
        with patch.object(entry, "get_default_manager", return_value=manager):
            assert wrapped(3) == 4

    impl = manager.registry.get_implementation("triton_test_op", "te.native")
    assert impl.fn is native
    assert impl.kind == BackendImplKind.REFERENCE


def test_entry_registers_natives_on_replacement_manager(isolated_entry_state):
    managers = [OpManager(), OpManager()]

    def native(value):
        return value

    entry.remember_natives({"triton_test_op": native})
    wrapped = entry.make_entry("triton_test_op", native)

    for manager in managers:
        with patch.object(manager, "ensure_initialized"):
            with patch.object(entry, "get_default_manager", return_value=manager):
                assert wrapped(7) == 7
        assert manager.registry.get_implementation("triton_test_op", "te.native").fn is native


def test_concurrent_first_registration_is_idempotent(isolated_entry_state):
    manager = OpManager()

    def native(value):
        return value

    entry.remember_natives({"triton_test_op": native})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: entry.ensure_native_registered(manager), range(32)))

    impls = manager.registry.get_implementations("triton_test_op")
    assert [impl.impl_id for impl in impls] == ["te.native"]


def test_later_capture_advances_registration_generation(isolated_entry_state):
    manager = OpManager()

    def first():
        return 1

    def second():
        return 2

    entry.remember_natives({"triton_first": first})
    entry.ensure_native_registered(manager)
    entry.remember_natives({"triton_second": second})
    entry.ensure_native_registered(manager)

    assert manager.registry.get_implementation("triton_first", "te.native").fn is first
    assert manager.registry.get_implementation("triton_second", "te.native").fn is second


def test_registering_batch_after_collision_does_not_leave_partial_state(
    isolated_entry_state,
):
    manager = OpManager()

    def first():
        return "first"

    def second():
        return "second"

    def conflicting():
        return "conflict"

    manager.registry.register_impl(
        OpImpl(
            op_name="triton_second",
            impl_id="te.native",
            kind=BackendImplKind.REFERENCE,
            fn=conflicting,
        )
    )
    entry.remember_natives({"triton_first": first, "triton_second": second})

    with pytest.raises(RuntimeError, match="Unexpected te.native"):
        entry.ensure_native_registered(manager)

    assert manager.registry.get_implementation("triton_first", "te.native") is None


def test_conflicting_native_registration_is_not_swallowed(isolated_entry_state):
    manager = OpManager()

    def native():
        return "native"

    def conflicting():
        return "conflict"

    manager.registry.register_impl(
        OpImpl(
            op_name="triton_test_op",
            impl_id="te.native",
            kind=BackendImplKind.REFERENCE,
            fn=conflicting,
        )
    )
    entry.remember_natives({"triton_test_op": native})

    with pytest.raises(RuntimeError, match="Unexpected te.native"):
        entry.ensure_native_registered(manager)


def test_entry_selects_vendor_registered_after_native_resolution(isolated_entry_state):
    manager = OpManager()

    def native(value):
        return f"native:{value}"

    def vendor(value):
        return f"vendor:{value}"

    entry.remember_natives({"triton_test_op": native})
    wrapped = entry.make_entry("triton_test_op", native)

    with patch.object(manager, "ensure_initialized"):
        with patch.object(entry, "get_default_manager", return_value=manager):
            assert wrapped(3) == "native:3"
            manager.registry.register_impl(
                OpImpl(
                    op_name="triton_test_op",
                    impl_id="vendor.test",
                    kind=BackendImplKind.VENDOR,
                    vendor="test",
                    fn=vendor,
                )
            )
            assert wrapped(3) == "vendor:3"


def test_vendor_failure_is_not_retried_with_native(isolated_entry_state):
    manager = OpManager()
    calls = []

    def native(value):
        calls.append(("native", value))
        return value

    def vendor(value):
        calls.append(("vendor", value))
        raise RuntimeError("vendor failed")

    entry.remember_natives({"triton_test_op": native})
    entry.ensure_native_registered(manager)
    manager.registry.register_impl(
        OpImpl(
            op_name="triton_test_op",
            impl_id="vendor.test",
            kind=BackendImplKind.VENDOR,
            vendor="test",
            fn=vendor,
        )
    )
    wrapped = entry.make_entry("triton_test_op", native)

    with patch.object(manager, "ensure_initialized"):
        with patch.object(entry, "get_default_manager", return_value=manager):
            with pytest.raises(RuntimeError, match="vendor failed"):
                wrapped(3)

    assert calls == [("vendor", 3)]


def test_registry_clear_restores_native_registration(isolated_entry_state):
    manager = OpManager()

    def native(value):
        return value

    entry.remember_natives({"triton_test_op": native})
    entry.ensure_native_registered(manager)
    manager.registry.clear()
    entry.ensure_native_registered(manager)

    assert manager.registry.get_implementation("triton_test_op", "te.native").fn is native


def test_reset_after_fork_replaces_registration_lock(isolated_entry_state):
    old_lock = entry._lock

    def native(value):
        return value

    # Populate both stores first: the fixture starts them empty, so asserting
    # they are empty afterwards would pass even if the reset dropped them.
    manager = OpManager()
    entry.remember_natives({"triton_test_op": native})
    with patch.object(manager, "ensure_initialized"):
        with patch.object(entry, "get_default_manager", return_value=manager):
            entry.make_entry("triton_test_op", native)(1)
    assert entry._fast_path and len(entry._registered_managers) == 1

    entry._reset_after_fork()

    assert entry._lock is not old_lock
    assert len(entry._registered_managers) == 0
    # Fork leaves the registry, the captured natives and the global policy
    # epoch untouched, so a surviving cache entry would validate and the child
    # would skip re-verifying registration on its own manager.
    assert entry._fast_path == {}


def test_second_call_takes_the_cached_fast_path(isolated_entry_state):
    """The fast path must serve the second call without re-resolving.

    Guards the hot path that keeps dispatch traceable by torch.compile: the
    slow path reads a WeakKeyDictionary keyed by the manager object, which
    Dynamo cannot trace, so it has to stay off the repeat path.
    """
    manager = OpManager()
    resolved = []

    def native(value):
        return value + 1

    entry.remember_natives({"triton_test_op": native})
    wrapped = entry.make_entry("triton_test_op", native)
    original_resolve = manager.resolve

    def counting_resolve(op_name):
        resolved.append(op_name)
        return original_resolve(op_name)

    with patch.object(manager, "ensure_initialized"):
        with patch.object(entry, "get_default_manager", return_value=manager):
            with patch.object(manager, "resolve", side_effect=counting_resolve):
                assert wrapped(3) == 4
                assert resolved == ["triton_test_op"]
                assert wrapped(3) == 4
                assert wrapped(3) == 4

    assert resolved == ["triton_test_op"]
