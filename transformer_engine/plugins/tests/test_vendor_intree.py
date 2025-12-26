# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Test and example for In-tree vendor plugin integration.

This demonstrates:
- How to register vendor implementations in-tree (open source contribution)
- Using OpImplKind.VENDOR with vendor name
- Policy-based selection with TE_FL_PREFER
- Per-op ordering and vendor filtering
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any

import torch


class TestInTreeVendorPlugin(unittest.TestCase):
    """
    Test case demonstrating in-tree vendor plugin integration.

    In-tree plugins are suitable for:
    - Open source vendor implementations
    - Following the main plugin repository release cycle
    - Direct integration into the codebase
    """

    def setUp(self):
        """Reset environment and policy before each test"""
        # Clear environment variables
        for key in list(os.environ.keys()):
            if key.startswith("TE_FL_"):
                del os.environ[key]

        # Reset global policy
        from transformer_engine.plugins.transformer_engine_fl.policy import reset_global_policy
        reset_global_policy()

    def test_vendor_plugin_registration(self):
        """Test registering a vendor implementation in-tree"""
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry

        # Create a mock vendor implementation
        def acme_rmsnorm_fwd(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5, **kwargs):
            """ACME vendor optimized RMSNorm implementation"""
            # Simulated vendor implementation
            variance = input.pow(2).mean(-1, keepdim=True)
            output = input * torch.rsqrt(variance + eps) * weight
            rsigma = torch.rsqrt(variance + eps)
            return output, rsigma

        def acme_is_available():
            """Check if ACME hardware is available"""
            # In real scenario, check device availability
            return True

        acme_rmsnorm_fwd._is_available = acme_is_available

        # Register vendor implementation
        registry = OpRegistry()
        impl = OpImpl(
            op_name="rmsnorm_fwd",
            impl_id="vendor.acme.rmsnorm_fwd",
            kind=BackendImplKind.VENDOR,
            vendor="acme",
            fn=acme_rmsnorm_fwd,
            priority=100,
        )
        registry.register_impl(impl)

        # Verify registration
        impls = registry.get_implementations("rmsnorm_fwd")
        self.assertEqual(len(impls), 1)
        self.assertEqual(impls[0].vendor, "acme")
        self.assertEqual(impls[0].kind, BackendImplKind.VENDOR)
        self.assertTrue(impls[0].is_available())

    def test_multiple_vendor_registration(self):
        """Test registering multiple vendor implementations"""
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry

        registry = OpRegistry()

        # Register multiple vendors for the same op
        for vendor_name, priority in [("acme", 100), ("nvidia", 90), ("amd", 80)]:
            def vendor_impl(input, weight, eps=1e-5, **kwargs):
                variance = input.pow(2).mean(-1, keepdim=True)
                output = input * torch.rsqrt(variance + eps) * weight
                rsigma = torch.rsqrt(variance + eps)
                return output, rsigma

            vendor_impl._is_available = lambda: True

            impl = OpImpl(
                op_name="rmsnorm_fwd",
                impl_id=f"vendor.{vendor_name}.rmsnorm_fwd",
                kind=BackendImplKind.VENDOR,
                vendor=vendor_name,
                fn=vendor_impl,
                priority=priority,
            )
            registry.register_impl(impl)

        # Verify all vendors registered
        impls = registry.get_implementations("rmsnorm_fwd")
        self.assertEqual(len(impls), 3)

        vendors = {impl.vendor for impl in impls}
        self.assertEqual(vendors, {"acme", "nvidia", "amd"})

    def test_vendor_selection_with_policy(self):
        """Test vendor selection using policy"""
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl, match_token
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy

        registry = OpRegistry()

        # Register default implementation
        def default_impl(input, weight, eps=1e-5, **kwargs):
            variance = input.pow(2).mean(-1, keepdim=True)
            return input * torch.rsqrt(variance + eps) * weight, torch.rsqrt(variance + eps)

        registry.register_impl(OpImpl(
            op_name="rmsnorm_fwd",
            impl_id="default.rmsnorm_fwd",
            kind=BackendImplKind.DEFAULT,
            fn=default_impl,
            priority=50,
        ))

        # Register vendor implementation
        def vendor_impl(input, weight, eps=1e-5, **kwargs):
            variance = input.pow(2).mean(-1, keepdim=True)
            return input * torch.rsqrt(variance + eps) * weight, torch.rsqrt(variance + eps)

        vendor_impl._is_available = lambda: True

        registry.register_impl(OpImpl(
            op_name="rmsnorm_fwd",
            impl_id="vendor.acme.rmsnorm_fwd",
            kind=BackendImplKind.VENDOR,
            vendor="acme",
            fn=vendor_impl,
            priority=100,
        ))

        # Test prefer="vendor"
        policy = SelectionPolicy(prefer="vendor")
        order = policy.get_default_order()
        self.assertEqual(order, ["vendor", "default", "reference"])

        # Test prefer="default"
        policy_default = SelectionPolicy(prefer="default")
        order_default = policy_default.get_default_order()
        self.assertEqual(order_default, ["default", "vendor", "reference"])

        # Verify vendor token matching
        impls = registry.get_implementations("rmsnorm_fwd")
        vendor_impls = [impl for impl in impls if match_token(impl, "vendor")]
        self.assertEqual(len(vendor_impls), 1)
        self.assertEqual(vendor_impls[0].vendor, "acme")

    def test_per_op_vendor_selection(self):
        """Test per-op vendor ordering"""
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy

        # Create policy with per-op ordering
        policy = SelectionPolicy.from_dict(
            prefer="vendor",
            per_op_order={
                "rmsnorm_fwd": ["vendor:acme", "vendor:nvidia", "default"],
                "rope_fwd": ["default", "vendor"],
            }
        )

        # Verify per-op orders
        rmsnorm_order = policy.get_per_op_order("rmsnorm_fwd")
        self.assertEqual(rmsnorm_order, ["vendor:acme", "vendor:nvidia", "default"])

        rope_order = policy.get_per_op_order("rope_fwd")
        self.assertEqual(rope_order, ["default", "vendor"])

        # Test default fallback
        unknown_order = policy.get_per_op_order("unknown_op")
        self.assertIsNone(unknown_order)

    def test_vendor_allow_deny_filters(self):
        """Test vendor allow/deny list filtering"""
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy

        # Test deny vendors
        policy_deny = SelectionPolicy.from_dict(
            deny_vendors={"bad_vendor", "another_bad"}
        )
        self.assertFalse(policy_deny.is_vendor_allowed("bad_vendor"))
        self.assertTrue(policy_deny.is_vendor_allowed("good_vendor"))

        # Test allow vendors (whitelist)
        policy_allow = SelectionPolicy.from_dict(
            allow_vendors={"acme", "nvidia"}
        )
        self.assertTrue(policy_allow.is_vendor_allowed("acme"))
        self.assertTrue(policy_allow.is_vendor_allowed("nvidia"))
        self.assertFalse(policy_allow.is_vendor_allowed("amd"))

    def test_env_var_configuration(self):
        """Test environment variable configuration"""
        from transformer_engine.plugins.transformer_engine_fl.policy import policy_from_env, reset_global_policy

        # Test TE_FL_PREFER (highest priority)
        os.environ["TE_FL_PREFER"] = "vendor"
        reset_global_policy()
        policy = policy_from_env()
        self.assertEqual(policy.prefer, "vendor")

        os.environ["TE_FL_PREFER"] = "reference"
        reset_global_policy()
        policy = policy_from_env()
        self.assertEqual(policy.prefer, "reference")

        os.environ["TE_FL_PREFER"] = "default"
        reset_global_policy()
        policy = policy_from_env()
        self.assertEqual(policy.prefer, "default")

        # Test TE_FL_PREFER_VENDOR (legacy, lower priority)
        del os.environ["TE_FL_PREFER"]
        os.environ["TE_FL_PREFER_VENDOR"] = "1"
        reset_global_policy()
        policy = policy_from_env()
        self.assertEqual(policy.prefer, "vendor")

        os.environ["TE_FL_PREFER_VENDOR"] = "0"
        reset_global_policy()
        policy = policy_from_env()
        self.assertEqual(policy.prefer, "default")

        # Test TE_FL_PREFER overrides TE_FL_PREFER_VENDOR
        os.environ["TE_FL_PREFER"] = "reference"
        os.environ["TE_FL_PREFER_VENDOR"] = "1"  # Should be ignored
        reset_global_policy()
        policy = policy_from_env()
        self.assertEqual(policy.prefer, "reference")  # TE_FL_PREFER wins

        del os.environ["TE_FL_PREFER"]
        del os.environ["TE_FL_PREFER_VENDOR"]

        # Test STRICT mode
        os.environ["TE_FL_STRICT"] = "1"
        policy = policy_from_env()
        self.assertTrue(policy.strict)

        # Test DENY_VENDORS
        os.environ["TE_FL_DENY_VENDORS"] = "bad1,bad2,bad3"
        policy = policy_from_env()
        self.assertEqual(policy.deny_vendors, frozenset({"bad1", "bad2", "bad3"}))

        # Test ALLOW_VENDORS
        os.environ["TE_FL_ALLOW_VENDORS"] = "acme,nvidia"
        policy = policy_from_env()
        self.assertEqual(policy.allow_vendors, frozenset({"acme", "nvidia"}))

        # Test PER_OP configuration
        os.environ["TE_FL_PER_OP"] = "rmsnorm_fwd=vendor:acme|default;rope_fwd=default|reference"
        policy = policy_from_env()
        rmsnorm_order = policy.get_per_op_order("rmsnorm_fwd")
        self.assertEqual(rmsnorm_order, ["vendor:acme", "default"])
        rope_order = policy.get_per_op_order("rope_fwd")
        self.assertEqual(rope_order, ["default", "reference"])

    def test_policy_context_override(self):
        """Test temporary policy override with context manager"""
        from transformer_engine.plugins.transformer_engine_fl.policy import (
            SelectionPolicy,
            get_policy,
            policy_context,
            set_global_policy,
        )

        # Set global policy
        global_policy = SelectionPolicy(prefer="vendor", strict=False)
        set_global_policy(global_policy)

        # Verify global policy
        self.assertEqual(get_policy().prefer, "vendor")
        self.assertFalse(get_policy().strict)

        # Override with context
        override_policy = SelectionPolicy(prefer="default", strict=True)
        with policy_context(override_policy):
            self.assertEqual(get_policy().prefer, "default")
            self.assertTrue(get_policy().strict)

        # Verify restored
        self.assertEqual(get_policy().prefer, "vendor")
        self.assertFalse(get_policy().strict)

    def test_in_tree_vendor_example_acme(self):
        """
        Complete example: ACME vendor in-tree plugin

        This shows how a vendor would contribute their implementation
        to the plugin repository as open source.
        """
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry

        # Simulate ACME vendor plugin registration function
        def acme_vendor_register(registry: OpRegistry) -> None:
            """
            ACME vendor registration function.

            This would be in: transformer_engine/plugins/vendor/acme/ops.py
            Called during plugin initialization.
            """

            # RMSNorm forward implementation
            def acme_rmsnorm_fwd(input: torch.Tensor, weight: torch.Tensor,
                                eps: float = 1e-5, **kwargs):
                """ACME optimized RMSNorm using custom kernels"""
                # In real implementation, this would call ACME's optimized kernels
                variance = input.pow(2).mean(-1, keepdim=True)
                output = input * torch.rsqrt(variance + eps) * weight
                rsigma = torch.rsqrt(variance + eps)
                return output, rsigma

            def acme_is_available():
                """Check ACME device availability"""
                # In real scenario: check ACME hardware/driver
                return torch.cuda.is_available()  # Simulated check

            acme_rmsnorm_fwd._is_available = acme_is_available

            # Register with vendor tag
            registry.register_impl(OpImpl(
                op_name="rmsnorm_fwd",
                impl_id="vendor.acme.rmsnorm_fwd.v1",
                kind=BackendImplKind.VENDOR,
                vendor="acme",
                fn=acme_rmsnorm_fwd,
                priority=100,
                supported_dtypes={"float16", "bfloat16", "float32"},
                min_arch="acme_arch_v2",
            ))

            # Could register more ops: rope, attention, etc.

        # Test the registration
        registry = OpRegistry()
        acme_vendor_register(registry)

        # Verify registration
        impls = registry.get_implementations("rmsnorm_fwd")
        self.assertEqual(len(impls), 1)

        acme_impl = impls[0]
        self.assertEqual(acme_impl.vendor, "acme")
        self.assertEqual(acme_impl.kind, BackendImplKind.VENDOR)
        self.assertEqual(acme_impl.priority, 100)
        self.assertIn("float16", acme_impl.supported_dtypes)

        # Test execution
        if acme_impl.is_available():
            input_tensor = torch.randn(2, 4, 8)
            weight_tensor = torch.ones(8)
            output, rsigma = acme_impl.fn(input_tensor, weight_tensor, eps=1e-5)
            self.assertEqual(output.shape, input_tensor.shape)


class TestVendorWithOpManager(unittest.TestCase):
    """
    Test vendor plugin using OpManager to see logging output.

    This demonstrates how to use the global OpManager to:
    1. Register vendor implementations
    2. Call operators through the manager (which prints logs)
    3. See which backend is being used for each op
    """

    def setUp(self):
        """Reset environment and managers before each test"""
        for key in list(os.environ.keys()):
            if key.startswith("TE_FL_"):
                del os.environ[key]

        from transformer_engine.plugins.transformer_engine_fl.policy import reset_global_policy
        from transformer_engine.plugins.transformer_engine_fl.manager import reset_default_manager
        reset_global_policy()
        reset_default_manager()

    def test_vendor_call_with_logging(self):
        """
        Test calling vendor implementation through OpManager.

        This will print logs like:
        [2025-xx-xx TE-FL manager.py:390 INFO] Op 'test_rmsnorm' using 'vendor.acme' (kind=vendor, vendor=acme)
        """
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl
        from transformer_engine.plugins.transformer_engine_fl.manager import OpManager
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy, set_global_policy

        print("\n" + "=" * 60)
        print("TEST: Vendor call with OpManager logging")
        print("=" * 60)

        # Create registry and manager
        registry = OpRegistry()
        manager = OpManager(registry)

        # Register DEFAULT implementation
        def default_rmsnorm(input, weight, eps=1e-5, **kwargs):
            print("  [CALLED] default_rmsnorm")
            variance = input.pow(2).mean(-1, keepdim=True)
            return input * torch.rsqrt(variance + eps) * weight, torch.rsqrt(variance + eps)

        registry.register_impl(OpImpl(
            op_name="test_rmsnorm",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=default_rmsnorm,
            priority=50,
        ))

        # Register VENDOR implementation (ACME)
        def acme_rmsnorm(input, weight, eps=1e-5, **kwargs):
            print("  [CALLED] acme_rmsnorm (vendor)")
            variance = input.pow(2).mean(-1, keepdim=True)
            return input * torch.rsqrt(variance + eps) * weight, torch.rsqrt(variance + eps)

        acme_rmsnorm._is_available = lambda: True

        registry.register_impl(OpImpl(
            op_name="test_rmsnorm",
            impl_id="vendor.acme",
            kind=BackendImplKind.VENDOR,
            vendor="acme",
            fn=acme_rmsnorm,
            priority=100,
        ))

        # Register REFERENCE implementation
        def reference_rmsnorm(input, weight, eps=1e-5, **kwargs):
            print("  [CALLED] reference_rmsnorm")
            variance = input.pow(2).mean(-1, keepdim=True)
            return input * torch.rsqrt(variance + eps) * weight, torch.rsqrt(variance + eps)

        registry.register_impl(OpImpl(
            op_name="test_rmsnorm",
            impl_id="reference.torch",
            kind=BackendImplKind.REFERENCE,
            fn=reference_rmsnorm,
            priority=10,
        ))

        # Print registered implementations
        snap = registry.snapshot()
        impl_ids = sorted(set(impl.impl_id for impls in snap.impls_by_op.values() for impl in impls))
        print(f"Registered impl_ids: {impl_ids}")

        # Test data
        input_tensor = torch.randn(2, 4, 8)
        weight_tensor = torch.ones(8)

        # Test 1: Default policy (prefer default)
        print("\n--- Test 1: prefer='default' ---")
        set_global_policy(SelectionPolicy(prefer="default"))
        result, _ = manager.call("test_rmsnorm", input_tensor, weight_tensor, eps=1e-5)
        self.assertEqual(result.shape, input_tensor.shape)

        # Test 2: Prefer vendor
        print("\n--- Test 2: prefer='vendor' ---")
        set_global_policy(SelectionPolicy(prefer="vendor"))
        result, _ = manager.call("test_rmsnorm", input_tensor, weight_tensor, eps=1e-5)
        self.assertEqual(result.shape, input_tensor.shape)

        # Test 3: Prefer reference
        print("\n--- Test 3: prefer='reference' ---")
        set_global_policy(SelectionPolicy(prefer="reference"))
        result, _ = manager.call("test_rmsnorm", input_tensor, weight_tensor, eps=1e-5)
        self.assertEqual(result.shape, input_tensor.shape)

        print("\n" + "=" * 60)
        print("TEST COMPLETED: Check logs above to see which backend was used")
        print("=" * 60 + "\n")


class TestVendorTokenMatching(unittest.TestCase):
    """Test vendor-specific token matching for policy selection"""

    def test_vendor_token_matching(self):
        """Test different vendor token patterns"""
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl, match_token

        def dummy_fn():
            pass

        # Create vendor implementations
        acme_impl = OpImpl(
            op_name="test_op",
            impl_id="vendor.acme.test",
            kind=BackendImplKind.VENDOR,
            vendor="acme",
            fn=dummy_fn,
        )

        nvidia_impl = OpImpl(
            op_name="test_op",
            impl_id="vendor.nvidia.test",
            kind=BackendImplKind.VENDOR,
            vendor="nvidia",
            fn=dummy_fn,
        )

        default_impl = OpImpl(
            op_name="test_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=dummy_fn,
        )

        # Test generic "vendor" token
        self.assertTrue(match_token(acme_impl, "vendor"))
        self.assertTrue(match_token(nvidia_impl, "vendor"))
        self.assertFalse(match_token(default_impl, "vendor"))

        # Test specific "vendor:name" tokens
        self.assertTrue(match_token(acme_impl, "vendor:acme"))
        self.assertFalse(match_token(acme_impl, "vendor:nvidia"))

        self.assertTrue(match_token(nvidia_impl, "vendor:nvidia"))
        self.assertFalse(match_token(nvidia_impl, "vendor:acme"))

        # Test "impl:id" tokens
        self.assertTrue(match_token(acme_impl, "impl:vendor.acme.test"))
        self.assertFalse(match_token(acme_impl, "impl:vendor.nvidia.test"))


def run_tests():
    """Run all in-tree vendor tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestInTreeVendorPlugin))
    suite.addTests(loader.loadTestsFromTestCase(TestVendorWithOpManager))
    suite.addTests(loader.loadTestsFromTestCase(TestVendorTokenMatching))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
