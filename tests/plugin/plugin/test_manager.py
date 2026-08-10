import os
import pytest
from unittest.mock import MagicMock, patch

from transformer_engine.plugin.core.types import BackendImplKind, OpImpl
from transformer_engine.plugin.core.policy import SelectionPolicy
from transformer_engine.plugin.core.registry import OpRegistry
from transformer_engine.plugin.core.manager import (
    OpManager,
    get_default_manager,
    reset_default_manager,
)


# ==============================================================================
# Fixtures & Mock Component Factories
# ==============================================================================


@pytest.fixture(autouse=True)
def clean_manager_singleton():
    """Ensure a freshly cleared manager instance before and after each test."""
    reset_default_manager()
    yield
    reset_default_manager()


def create_mock_impl(impl_id, kind, op_name="test_op", fn=None, priority=1, vendor=None):
    """
    Factory to generate fully structured OpImpl instances for control injection.
    Ensures that VENDOR kinds satisfy internal post-init constraint validations.
    """
    mock_fn = fn or MagicMock(return_value=f"res_{impl_id}")

    # Satisfy __post_init__ requirement: VENDOR kind must specify a vendor name
    if kind == BackendImplKind.VENDOR and not vendor:
        vendor = "nvidia"

    impl = OpImpl(
        op_name=op_name, impl_id=impl_id, kind=kind, fn=mock_fn, priority=priority, vendor=vendor
    )
    return impl


# ==============================================================================
# Part 1: Initialization, Fork Safety & Global Singleton Management
# ==============================================================================


def test_manager_singleton_lifecycle():
    """Verify singleton access, reset primitives, and Windows register_at_fork guards."""
    mgr1 = get_default_manager()
    mgr2 = get_default_manager()
    assert mgr1 is mgr2

    # Force error branch covering missing register_at_fork (e.g. Windows platforms)
    with patch("os.register_at_fork", side_effect=AttributeError):
        custom_mgr = OpManager()
        assert custom_mgr is not None


def test_lazy_initialization_flow():
    """Trigger ensure_initialized, checking registry synchronization and tracking logs."""
    mock_registry = OpRegistry()
    mgr = OpManager(registry=mock_registry)

    assert mgr.registry is mock_registry

    mgr.ensure_initialized()
    assert mgr._state.initialized is True
    assert mgr._state.init_pid == os.getpid()

    mgr.ensure_initialized()


def test_process_fork_invalidation_handling():
    """Force execute _reset_after_fork to clear transient states and step up policy epochs."""
    mgr = OpManager()
    mgr.ensure_initialized()

    mgr._dispatch_cache[("op", "fp", 0)] = lambda: None
    mgr._impl_cache["op"] = MagicMock()

    mgr._reset_after_fork()

    assert mgr._state.initialized is False
    assert mgr._state.init_pid == -1
    assert len(mgr._dispatch_cache) == 0
    assert len(mgr._impl_cache) == 0


# ==============================================================================
# Part 2: Vendor Whitelist/Blacklist Filtering Engine
# ==============================================================================


def test_vendor_policy_filter_matching():
    """Trigger _matches_vendor_filters evaluating valid, blocked, and non-vendor impls."""
    mgr = OpManager()

    non_vendor_impl = create_mock_impl("ref", BackendImplKind.REFERENCE)
    nvidia_impl = create_mock_impl("nv", BackendImplKind.VENDOR, vendor="nvidia")
    amd_impl = create_mock_impl("amd", BackendImplKind.VENDOR, vendor="amd")

    # Manually instantiate a VENDOR bypass to simulate missing vendor string if allowed by logic
    # Direct instantiation bypassed since it would hit __post_init__ error otherwise
    with patch.object(OpImpl, "__post_init__", return_value=None):
        vendor_no_name = OpImpl(
            op_name="test_op",
            impl_id="vend_none",
            kind=BackendImplKind.VENDOR,
            fn=MagicMock(),
            priority=1,
            vendor=None,
        )

    # Scenario 1: Deny List Filtering
    policy_deny = SelectionPolicy.from_dict(deny_vendors={"amd"})
    assert mgr._matches_vendor_filters(non_vendor_impl, policy_deny) is True
    assert mgr._matches_vendor_filters(vendor_no_name, policy_deny) is False
    assert mgr._matches_vendor_filters(nvidia_impl, policy_deny) is True
    assert mgr._matches_vendor_filters(amd_impl, policy_deny) is False

    # Scenario 2: Allow Whitelist Filtering
    policy_allow = SelectionPolicy.from_dict(allow_vendors={"nvidia"})
    assert mgr._matches_vendor_filters(nvidia_impl, policy_allow) is True
    assert mgr._matches_vendor_filters(amd_impl, policy_allow) is False


# ==============================================================================
# Part 3: Resolver Pipelines and Resolution Error Fallbacks
# ==============================================================================


def test_resolve_with_cache_and_priority():
    """Test operational resolve pathways, cache hits, priority sorting and empty states."""
    mock_registry = OpRegistry()
    mgr = OpManager(registry=mock_registry)

    impl_low = create_mock_impl(
        "v1", BackendImplKind.VENDOR, op_name="test_op", priority=1, vendor="nvidia"
    )
    impl_high = create_mock_impl(
        "v2", BackendImplKind.VENDOR, op_name="test_op", priority=10, vendor="nvidia"
    )

    mock_registry.register_impl(impl_low)
    mock_registry.register_impl(impl_high)

    # Safe patch of object method on frozen dataclasses to return True
    with patch.object(OpImpl, "is_available", return_value=True):
        selected_fn = mgr.resolve("test_op")
        assert selected_fn == impl_high.fn
        assert mgr.get_selected_impl_id("test_op") == "v2"
        assert mgr.resolve("test_op") == selected_fn


def test_resolution_failures_and_strict_modes():
    """Provoke exception blocks when operators are missing or filtered out."""
    mock_registry = OpRegistry()
    mgr = OpManager(registry=mock_registry)

    # nonexistent operator
    with pytest.raises(RuntimeError, match="No available implementation"):
        mgr.resolve("ghost_op")

    with pytest.raises(RuntimeError, match="No available implementation"):
        mgr.resolve_candidates("ghost_op")

    # availability check failure
    broken_impl = create_mock_impl(
        "broken",
        BackendImplKind.REFERENCE,
        op_name="broken_op",
    )
    mock_registry.register_impl(broken_impl)

    with patch.object(
        OpImpl,
        "is_available",
        side_effect=Exception("HW Missing"),
    ):
        with pytest.raises(RuntimeError, match="No available implementation"):
            mgr.resolve("broken_op")

    # vendor policy filters out all candidates
    amd_impl = create_mock_impl(
        "amd_impl",
        BackendImplKind.VENDOR,
        op_name="strict_op",
        vendor="amd",
    )

    mock_registry.register_impl(amd_impl)

    policy = SelectionPolicy.from_dict(
        allow_vendors={"nvidia"},
        strict=True,
    )

    with patch(
        "transformer_engine.plugin.core.manager.get_policy",
        return_value=policy,
    ):
        with patch.object(OpImpl, "is_available", return_value=True):
            with pytest.raises(
                RuntimeError,
                match="No available implementation",
            ):
                mgr.resolve("strict_op")


# ==============================================================================
# Part 4: High-Level Core Dispatch Invokers (call & fallback)
# ==============================================================================


def test_call_with_fallback_and_invalidation():
    """Route execution patterns through standard invoke, caching, errors, and fallbacks."""

    # ------------------------------------------------------------------
    # Case 1:
    # vendor implementation fails
    # reference implementation succeeds (fallback path)
    # ------------------------------------------------------------------
    registry = OpRegistry()
    mgr = OpManager(registry=registry)

    primary_impl = create_mock_impl(
        "v1",
        BackendImplKind.VENDOR,
        op_name="fallback_op",
        vendor="nvidia",
    )
    primary_impl.fn.side_effect = Exception("CUDA Out of Memory")

    backup_impl = create_mock_impl(
        "ref",
        BackendImplKind.REFERENCE,
        op_name="fallback_op",
    )

    registry.register_impl(primary_impl)
    registry.register_impl(backup_impl)

    with patch.object(OpImpl, "is_available", return_value=True):
        result = mgr.call("fallback_op", 10, x=5)

        assert result == "res_ref"

        backup_impl.fn.assert_called_once_with(
            10,
            x=5,
        )

        assert mgr._get_last_impl_id("fallback_op") == "ref"

    # ------------------------------------------------------------------
    # Case 2:
    # strict mode (TE_FL_STRICT=0)
    # fallback disabled
    # vendor implementation failure should propagate directly
    # ------------------------------------------------------------------
    strict_registry = OpRegistry()

    failing_impl = create_mock_impl(
        "strict_vendor",
        BackendImplKind.VENDOR,
        op_name="strict_op",
        vendor="nvidia",
    )

    failing_impl.fn.side_effect = Exception("CUDA Out of Memory")

    strict_registry.register_impl(failing_impl)

    strict_mgr = OpManager(registry=strict_registry)

    with patch("os.getenv", return_value="0"):
        with patch.object(OpImpl, "is_available", return_value=True):
            with pytest.raises(Exception, match="CUDA Out of Memory"):
                strict_mgr.call("strict_op")


# ==============================================================================
# Part 5: Cache Stability and Helper Primitives
# ==============================================================================


def test_cache_validation_and_epoch_bumps():
    """Cover _is_cache_valid, _update_cache and bump_policy_epoch."""
    mgr = OpManager()

    assert mgr._is_cache_valid("unknown_op") is False

    impl = create_mock_impl(
        "v1",
        BackendImplKind.VENDOR,
        op_name="validated_op",
        vendor="nvidia",
    )

    mgr._update_cache("validated_op", impl)

    assert mgr._is_cache_valid("validated_op") is True

    mgr.bump_policy_epoch()

    assert mgr._is_cache_valid("validated_op") is False

    assert mgr._get_last_impl_id("validated_op") == "v1"


def test_get_selected_impl_id():
    """Verify selected impl id lookup through resolve()."""

    registry = OpRegistry()

    impl = create_mock_impl(
        "v1",
        BackendImplKind.VENDOR,
        op_name="validated_op",
        vendor="nvidia",
    )

    registry.register_impl(impl)

    mgr = OpManager(registry=registry)

    with patch.object(mgr, "ensure_initialized"):
        with patch.object(OpImpl, "is_available", return_value=True):
            assert mgr.get_selected_impl_id("validated_op") == "v1"
