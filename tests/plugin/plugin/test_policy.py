import os
import pytest
import contextvars
from unittest.mock import patch

# Import all target classes and convenience functions
from transformer_engine.plugin.core.policy import (
    SelectionPolicy,
    PolicyManager,
    VALID_PREFER_VALUES,
    PREFER_DEFAULT,
    PREFER_VENDOR,
    PREFER_REFERENCE,
    get_policy_epoch,
    bump_policy_epoch,
    get_policy,
    set_global_policy,
    reset_global_policy,
    policy_from_env,
    policy_context,
    with_strict_mode,
    with_preference,
    with_allowed_vendors,
    with_denied_vendors,
)

# ==============================================================================
# Part 1: SelectionPolicy Core Logic & Edge-Case Interception
# ==============================================================================


def test_selection_policy_invalid_prefer():

    with pytest.raises(ValueError) as excinfo:
        SelectionPolicy(prefer="invalid_backend")
    assert "Invalid prefer value" in str(excinfo.value)


def test_selection_policy_from_dict_and_properties():

    per_op_order = {"te_gemm": ["vendor", "flagos"], "te_layernorm": ["reference"]}

    policy = SelectionPolicy.from_dict(
        prefer="VENDOR",  # Test case insensitivity via .lower()
        strict=True,
        per_op_order=per_op_order,
        deny_vendors={"amd", "intel"},
        allow_vendors={"nvidia"},
    )

    assert policy.prefer == "vendor"
    assert policy.strict is True
    # Target the per_op_order_dict property line
    assert policy.per_op_order_dict["te_gemm"] == ["vendor", "flagos"]

    # Target the loop hit and None fallback blocks within get_per_op_order
    assert policy.get_per_op_order("te_gemm") == ["vendor", "flagos"]
    assert policy.get_per_op_order("non_existent_op") is None


def test_selection_policy_default_orders():

    assert SelectionPolicy(prefer=PREFER_REFERENCE).get_default_order() == [
        "reference",
        "flagos",
        "vendor",
    ]
    assert SelectionPolicy(prefer=PREFER_VENDOR).get_default_order() == [
        "vendor",
        "flagos",
        "reference",
    ]
    assert SelectionPolicy(prefer=PREFER_DEFAULT).get_default_order() == [
        "flagos",
        "vendor",
        "reference",
    ]


def test_selection_policy_vendor_whitelist_blacklist():

    # 1. Blacklist interception
    policy_deny = SelectionPolicy.from_dict(deny_vendors={"bad_vendor"})
    assert policy_deny.is_vendor_allowed("bad_vendor") is False
    assert policy_deny.is_vendor_allowed("good_vendor") is True

    # 2. Whitelist miss interception
    policy_allow = SelectionPolicy.from_dict(allow_vendors={"nvidia"})
    assert policy_allow.is_vendor_allowed("nvidia") is True
    assert policy_allow.is_vendor_allowed("amd") is False


def test_selection_policy_fingerprint_and_hash():

    policy = SelectionPolicy.from_dict(
        prefer="flagos",
        strict=True,
        per_op_order={"op1": ["vendor"]},
        deny_vendors={"intel"},
        allow_vendors={"nvidia"},
    )
    fp = policy.fingerprint()
    assert "prefer=flagos" in fp
    assert "st=1" in fp
    assert "allow=nvidia" in fp
    assert "deny=intel" in fp
    assert "per=op1=vendor" in fp

    # Trigger __hash__
    assert isinstance(hash(policy), int)


# ==============================================================================
# Part 2: PolicyManager Singleton Pattern & Epoch State Control
# ==============================================================================


def test_policy_manager_singleton_and_epoch():

    mgr1 = PolicyManager.get_instance()
    mgr2 = PolicyManager.get_instance()
    assert mgr1 is mgr2

    # Target the duplicate initialization guard condition
    mgr1.__init__()

    # Test epoch manipulation convenience functions
    init_epoch = get_policy_epoch()
    new_epoch = bump_policy_epoch()
    assert new_epoch == init_epoch + 1
    assert get_policy_epoch() == new_epoch


# ==============================================================================
# Part 3: Static Environment Variable Parsers
# ==============================================================================


def test_parse_csv_set_edge_cases():

    mgr = PolicyManager.get_instance()
    assert mgr._parse_csv_set("") == set()
    assert mgr._parse_csv_set("  nvidia, , amd ,") == {"nvidia", "amd"}


def test_parse_per_op_edge_cases():

    mgr = PolicyManager.get_instance()
    assert mgr._parse_per_op("") == {}

    # Mixed input: contains malformed missing '=' string and empty elements
    bad_str = "invalid_format ; op1=vendor|flagos ; op2= ; =flagos"
    res = mgr._parse_per_op(bad_str)
    assert "op1" in res
    assert res["op1"] == ["vendor", "flagos"]


def test_policy_from_env_cascading():

    # Scenario 1: Highest priority environment variable 'TE_FL_PREFER'
    env_mock_1 = {
        "TE_FL_PREFER": "reference",
        "TE_FL_STRICT": "1",
        "TE_FL_DENY_VENDORS": "amd",
        "TE_FL_ALLOW_VENDORS": "nvidia",
        "TE_FL_PER_OP": "gemm=vendor",
    }
    with patch.dict(os.environ, env_mock_1):
        p = policy_from_env()
        assert p.prefer == "reference"
        assert p.strict is True
        assert "amd" in p.deny_vendors
        assert "nvidia" in p.allow_vendors

    # Scenario 2: Invalid 'TE_FL_PREFER' triggers [WARNING] printout and reverts to default
    with patch.dict(os.environ, {"TE_FL_PREFER": "corrupted_value"}):
        p = policy_from_env()
        assert p.prefer == "flagos"

    # Scenario 3: Fall back to legacy 'TE_FL_PREFER_VENDOR' evaluation logic (1=vendor, 0=flagos)
    with patch.dict(os.environ, {"TE_FL_PREFER": "", "TE_FL_PREFER_VENDOR": "1"}):
        assert policy_from_env().prefer == "vendor"

    with patch.dict(os.environ, {"TE_FL_PREFER": "", "TE_FL_PREFER_VENDOR": "0"}):
        assert policy_from_env().prefer == "flagos"


# ==============================================================================
# Part 4: Context Managers & Global Override Utilities
# ==============================================================================


def test_global_policy_lifecycle():

    init_policy = get_policy()
    new_policy = SelectionPolicy(prefer="vendor")

    old = set_global_policy(new_policy)
    assert get_policy().prefer == "vendor"

    reset_global_policy()
    # Restore original state
    set_global_policy(init_policy)


def test_policy_context_manager():

    base_policy = get_policy()
    override_policy = SelectionPolicy(prefer="reference")

    with policy_context(override_policy):
        assert get_policy().prefer == "reference"

    # Policy must revert back after exiting the context
    assert get_policy() == base_policy


def test_convenience_context_managers():

    # 1. Strict mode shortcut
    with with_strict_mode():
        assert get_policy().strict is True

    # 2. Preference shortcut
    with with_preference("vendor"):
        assert get_policy().prefer == "vendor"

    # 3. Whitelist/Blacklist vendor shortcuts
    with with_allowed_vendors("intel", "xpu"):
        assert get_policy().allow_vendors == frozenset({"intel", "xpu"})

    with with_denied_vendors("mock_gpu"):
        assert "mock_gpu" in get_policy().deny_vendors
