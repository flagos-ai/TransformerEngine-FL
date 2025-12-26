# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Test and example for Out-of-tree vendor plugin integration.

This demonstrates:
- Entry points-based plugin discovery (wheel packages)
- Environment variable module loading (development/testing)
- Closed-source vendor implementations
- Independent release cycles
- Zero modification to main repository
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import torch


class TestOutOfTreeEntryPoints(unittest.TestCase):
    """
    Test out-of-tree plugin discovery via entry points.

    This is the recommended approach for wheel-distributed plugins.
    """

    def setUp(self):
        """Reset discovery state"""
        from transformer_engine.plugins.transformer_engine_fl.discovery import clear_discovered_plugins
        clear_discovered_plugins()

        # Clear environment
        for key in list(os.environ.keys()):
            if key.startswith("TE_FL_"):
                del os.environ[key]

    def test_entry_point_plugin_structure(self):
        """
        Demonstrate the structure of an entry point plugin.

        A vendor would create a package like:

        acme_te_plugin/
        ├── setup.py (or pyproject.toml)
        └── acme_te_plugin/
            ├── __init__.py
            └── ops.py

        setup.py includes:
        ```python
        setup(
            name="acme-te-plugin",
            entry_points={
                "te_fl.plugins": [
                    "acme = acme_te_plugin:te_fl_register"
                ]
            }
        )
        ```
        """
        # This is a structural test - actual entry point registration
        # requires package installation, which we simulate in other tests
        from transformer_engine.plugins.transformer_engine_fl.discovery import PLUGIN_GROUP

        self.assertEqual(PLUGIN_GROUP, "te_fl.plugins")

    def test_plugin_register_function_signature(self):
        """
        Test that plugin register functions follow the correct signature.

        Plugins should provide either:
        1. def te_fl_register(registry: OpRegistry) -> None (recommended)
        2. def register(registry: OpRegistry) -> None (alternative)
        """
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl

        # Example plugin module
        class MockVendorPlugin:
            @staticmethod
            def te_fl_register(registry: OpRegistry) -> None:
                """Recommended register function name"""
                def vendor_op(*args, **kwargs):
                    pass
                vendor_op._is_available = lambda: True

                registry.register_impl(OpImpl(
                    op_name="test_op",
                    impl_id="vendor.mock.test_op",
                    kind=BackendImplKind.VENDOR,
                    vendor="mock",
                    fn=vendor_op,
                ))

        # Test registration
        registry = OpRegistry()
        plugin = MockVendorPlugin()
        plugin.te_fl_register(registry)

        impls = registry.get_implementations("test_op")
        self.assertEqual(len(impls), 1)
        self.assertEqual(impls[0].vendor, "mock")

    def test_entry_point_discovery_workflow(self):
        """
        Test the complete entry point discovery workflow.

        This simulates what happens when a user installs a plugin wheel.
        """
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry
        from transformer_engine.plugins.transformer_engine_fl.discovery import discover_from_entry_points

        # In a real scenario, entry points are discovered from installed packages
        # Here we test the discovery mechanism

        registry = OpRegistry()

        # Note: discover_from_entry_points() would find installed packages
        # In this test environment, we may not have any installed
        count = discover_from_entry_points(registry)

        # Count could be 0 if no plugins installed, which is fine for testing
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)


class TestOutOfTreeEnvModule(unittest.TestCase):
    """
    Test out-of-tree plugin discovery via environment variable.

    This approach is useful for:
    - Development and testing
    - Internal/proprietary plugins
    - Plugins not distributed as wheels
    """

    def setUp(self):
        """Reset environment and discovery state"""
        from transformer_engine.plugins.transformer_engine_fl.discovery import clear_discovered_plugins
        clear_discovered_plugins()

        for key in list(os.environ.keys()):
            if key.startswith("TE_FL_"):
                del os.environ[key]

    def test_env_module_loading(self):
        """Test loading plugins via TE_FL_PLUGIN_MODULES"""
        from transformer_engine.plugins.transformer_engine_fl.discovery import PLUGIN_MODULES_ENV

        self.assertEqual(PLUGIN_MODULES_ENV, "TE_FL_PLUGIN_MODULES")

        # Example usage:
        # export TE_FL_PLUGIN_MODULES=acme_plugin,internal_ops
        # This would import and call register() from both modules

    def test_create_mock_plugin_module(self):
        """Create a mock plugin module for testing"""
        # Create a temporary directory for our mock plugin
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = Path(tmpdir)
            plugin_file = plugin_path / "mock_vendor_plugin.py"

            # Write a mock plugin module
            plugin_code = textwrap.dedent("""
                import torch
                from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl

                def te_fl_register(registry):
                    '''Register mock vendor ops'''

                    def mock_rmsnorm_fwd(input, weight, eps=1e-5, **kwargs):
                        variance = input.pow(2).mean(-1, keepdim=True)
                        output = input * torch.rsqrt(variance + eps) * weight
                        rsigma = torch.rsqrt(variance + eps)
                        return output, rsigma

                    mock_rmsnorm_fwd._is_available = lambda: True

                    registry.register_impl(OpImpl(
                        op_name="rmsnorm_fwd",
                        impl_id="vendor.mock.rmsnorm_fwd",
                        kind=BackendImplKind.VENDOR,
                        vendor="mock_vendor",
                        fn=mock_rmsnorm_fwd,
                        priority=120,
                    ))

                    def mock_rope_fwd(input, cos, sin, **kwargs):
                        # Simplified mock implementation
                        return input * cos + input * sin

                    mock_rope_fwd._is_available = lambda: True

                    registry.register_impl(OpImpl(
                        op_name="rope_fwd",
                        impl_id="vendor.mock.rope_fwd",
                        kind=BackendImplKind.VENDOR,
                        vendor="mock_vendor",
                        fn=mock_rope_fwd,
                        priority=110,
                    ))
            """)

            plugin_file.write_text(plugin_code)

            # Add to Python path
            sys.path.insert(0, str(plugin_path))

            try:
                # Import and test
                import mock_vendor_plugin
                from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry

                registry = OpRegistry()
                mock_vendor_plugin.te_fl_register(registry)

                # Verify registration
                rmsnorm_impls = registry.get_implementations("rmsnorm_fwd")
                self.assertEqual(len(rmsnorm_impls), 1)
                self.assertEqual(rmsnorm_impls[0].vendor, "mock_vendor")
                self.assertEqual(rmsnorm_impls[0].priority, 120)

                rope_impls = registry.get_implementations("rope_fwd")
                self.assertEqual(len(rope_impls), 1)
                self.assertEqual(rope_impls[0].vendor, "mock_vendor")

            finally:
                sys.path.remove(str(plugin_path))
                # Clean up imported module
                if 'mock_vendor_plugin' in sys.modules:
                    del sys.modules['mock_vendor_plugin']

    def test_env_module_discovery(self):
        """Test discovering plugins from environment variable"""
        from transformer_engine.plugins.transformer_engine_fl.discovery import (
            discover_from_env_modules,
            PLUGIN_MODULES_ENV,
        )
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry

        # Create a temporary plugin module
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = Path(tmpdir)
            plugin_file = plugin_path / "test_env_plugin.py"

            plugin_code = textwrap.dedent("""
                from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl

                def te_fl_register(registry):
                    def dummy_op():
                        pass
                    dummy_op._is_available = lambda: True

                    registry.register_impl(OpImpl(
                        op_name="env_test_op",
                        impl_id="vendor.env_test.op",
                        kind=BackendImplKind.VENDOR,
                        vendor="env_test",
                        fn=dummy_op,
                    ))
            """)

            plugin_file.write_text(plugin_code)
            sys.path.insert(0, str(plugin_path))

            try:
                # Set environment variable
                os.environ[PLUGIN_MODULES_ENV] = "test_env_plugin"

                registry = OpRegistry()
                loaded_count = discover_from_env_modules(registry)

                # Verify discovery
                self.assertEqual(loaded_count, 1)

                impls = registry.get_implementations("env_test_op")
                self.assertEqual(len(impls), 1)
                self.assertEqual(impls[0].vendor, "env_test")

            finally:
                sys.path.remove(str(plugin_path))
                if 'test_env_plugin' in sys.modules:
                    del sys.modules['test_env_plugin']
                if PLUGIN_MODULES_ENV in os.environ:
                    del os.environ[PLUGIN_MODULES_ENV]

    def test_multiple_env_modules(self):
        """Test loading multiple plugins via environment variable"""
        from transformer_engine.plugins.transformer_engine_fl.discovery import (
            discover_from_env_modules,
            PLUGIN_MODULES_ENV,
        )
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = Path(tmpdir)

            # Create two plugin modules
            for i, vendor in enumerate(["vendor_a", "vendor_b"]):
                plugin_file = plugin_path / f"plugin_{vendor}.py"
                plugin_code = textwrap.dedent(f"""
                    from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl

                    def te_fl_register(registry):
                        def dummy_op():
                            pass
                        dummy_op._is_available = lambda: True

                        registry.register_impl(OpImpl(
                            op_name="multi_test_op",
                            impl_id="vendor.{vendor}.op",
                            kind=BackendImplKind.VENDOR,
                            vendor="{vendor}",
                            fn=dummy_op,
                            priority={100 - i * 10},
                        ))
                """)
                plugin_file.write_text(plugin_code)

            sys.path.insert(0, str(plugin_path))

            try:
                # Set environment variable with multiple modules
                os.environ[PLUGIN_MODULES_ENV] = "plugin_vendor_a,plugin_vendor_b"

                registry = OpRegistry()
                loaded_count = discover_from_env_modules(registry)

                # Verify both plugins loaded
                self.assertEqual(loaded_count, 2)

                impls = registry.get_implementations("multi_test_op")
                self.assertEqual(len(impls), 2)

                vendors = {impl.vendor for impl in impls}
                self.assertEqual(vendors, {"vendor_a", "vendor_b"})

            finally:
                sys.path.remove(str(plugin_path))
                for mod_name in ["plugin_vendor_a", "plugin_vendor_b"]:
                    if mod_name in sys.modules:
                        del sys.modules[mod_name]
                if PLUGIN_MODULES_ENV in os.environ:
                    del os.environ[PLUGIN_MODULES_ENV]


class TestCompleteOutOfTreeExample(unittest.TestCase):
    """
    Complete example demonstrating an out-of-tree vendor plugin.

    This shows what a vendor would ship as a closed-source wheel package.
    """

    def test_complete_vendor_plugin_example(self):
        """
        Complete example: ACME vendor out-of-tree plugin

        Package structure:

        acme_te_plugin/
        ├── pyproject.toml
        ├── acme_te_plugin/
        │   ├── __init__.py
        │   ├── ops.py
        │   └── kernels/  (compiled kernels, closed source)
        │       └── libacme_kernels.so

        pyproject.toml:
        ```toml
        [project]
        name = "acme-te-plugin"
        version = "1.0.0"

        [project.entry-points."te_fl.plugins"]
        acme = "acme_te_plugin:te_fl_register"
        ```
        """
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry

        # Simulate the vendor plugin module
        class ACMETEPlugin:
            """ACME TransformerEngine Plugin (closed source)"""

            @staticmethod
            def _check_acme_device():
                """Check if ACME hardware is available"""
                # In real implementation:
                # - Check for ACME driver
                # - Query device capabilities
                # - Validate kernel compatibility
                return torch.cuda.is_available()  # Simulated

            @staticmethod
            def te_fl_register(registry: OpRegistry) -> None:
                """
                Register ACME vendor implementations.

                This function is called by the plugin discovery system
                when the package is installed and TE-FL initializes.
                """

                # RMSNorm Forward
                def acme_rmsnorm_fwd(input: torch.Tensor, weight: torch.Tensor,
                                    eps: float = 1e-5, **kwargs):
                    """
                    ACME optimized RMSNorm forward pass.

                    In real implementation, this would call:
                    from .kernels import libacme_kernels
                    return libacme_kernels.rmsnorm_fwd(input, weight, eps)
                    """
                    # Simulated implementation
                    variance = input.pow(2).mean(-1, keepdim=True)
                    output = input * torch.rsqrt(variance + eps) * weight
                    rsigma = torch.rsqrt(variance + eps)
                    return output, rsigma

                acme_rmsnorm_fwd._is_available = ACMETEPlugin._check_acme_device

                registry.register_impl(OpImpl(
                    op_name="rmsnorm_fwd",
                    impl_id="vendor.acme.rmsnorm_fwd.v2",
                    kind=BackendImplKind.VENDOR,
                    vendor="acme",
                    fn=acme_rmsnorm_fwd,
                    priority=150,  # High priority for optimized impl
                    supported_dtypes={"float16", "bfloat16", "float32"},
                    min_arch="acme_v2",
                ))

                # RoPE Forward
                def acme_rope_fwd(input: torch.Tensor, cos: torch.Tensor,
                                 sin: torch.Tensor, **kwargs):
                    """ACME optimized RoPE forward pass"""
                    # Simulated implementation
                    # Real: libacme_kernels.rope_fwd(input, cos, sin)
                    x1, x2 = input.chunk(2, dim=-1)
                    output = torch.cat([
                        x1 * cos - x2 * sin,
                        x2 * cos + x1 * sin
                    ], dim=-1)
                    return output

                acme_rope_fwd._is_available = ACMETEPlugin._check_acme_device

                registry.register_impl(OpImpl(
                    op_name="rope_fwd",
                    impl_id="vendor.acme.rope_fwd.v1",
                    kind=BackendImplKind.VENDOR,
                    vendor="acme",
                    fn=acme_rope_fwd,
                    priority=140,
                    supported_dtypes={"float16", "bfloat16"},
                ))

                # Flash Attention (if available)
                def acme_flash_attention(q, k, v, dropout_p=0.0, **kwargs):
                    """ACME optimized flash attention"""
                    # Simulated implementation
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
                    attn = torch.softmax(scores, dim=-1)
                    output = torch.matmul(attn, v)
                    return output

                acme_flash_attention._is_available = ACMETEPlugin._check_acme_device

                registry.register_impl(OpImpl(
                    op_name="flash_attention",
                    impl_id="vendor.acme.flash_attn.v3",
                    kind=BackendImplKind.VENDOR,
                    vendor="acme",
                    fn=acme_flash_attention,
                    priority=160,
                ))

        # Test the plugin
        registry = OpRegistry()
        ACMETEPlugin.te_fl_register(registry)

        # Verify all ops registered
        rmsnorm_impls = registry.get_implementations("rmsnorm_fwd")
        self.assertGreater(len(rmsnorm_impls), 0)
        acme_rmsnorm = next(impl for impl in rmsnorm_impls if impl.vendor == "acme")
        self.assertEqual(acme_rmsnorm.priority, 150)

        rope_impls = registry.get_implementations("rope_fwd")
        acme_rope = next(impl for impl in rope_impls if impl.vendor == "acme")
        self.assertEqual(acme_rope.vendor, "acme")

        flash_impls = registry.get_implementations("flash_attention")
        acme_flash = next(impl for impl in flash_impls if impl.vendor == "acme")
        self.assertEqual(acme_flash.priority, 160)

    def test_vendor_plugin_with_policy(self):
        """Test using vendor plugin with different policies"""
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl
        from transformer_engine.plugins.transformer_engine_fl.registry import OpRegistry
        from transformer_engine.plugins.transformer_engine_fl.policy import (
            SelectionPolicy,
            policy_from_env,
        )

        # Register multiple implementations
        registry = OpRegistry()

        # Default implementation
        def default_impl(*args, **kwargs):
            return "default"
        registry.register_impl(OpImpl(
            op_name="test_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=default_impl,
            priority=50,
        ))

        # ACME vendor
        def acme_impl(*args, **kwargs):
            return "acme"
        acme_impl._is_available = lambda: True
        registry.register_impl(OpImpl(
            op_name="test_op",
            impl_id="vendor.acme.test",
            kind=BackendImplKind.VENDOR,
            vendor="acme",
            fn=acme_impl,
            priority=100,
        ))

        # Test different policy scenarios

        # 1. Prefer vendor
        policy1 = SelectionPolicy(prefer="vendor")
        self.assertEqual(policy1.get_default_order(), ["vendor", "default", "reference"])

        # 2. Prefer default using TE_FL_PREFER_VENDOR=0
        os.environ["TE_FL_PREFER_VENDOR"] = "0"
        policy2 = policy_from_env()
        self.assertEqual(policy2.prefer, "default")
        self.assertEqual(policy2.get_default_order(), ["default", "vendor", "reference"])

        # 3. TE_FL_PREFER overrides TE_FL_PREFER_VENDOR
        os.environ["TE_FL_PREFER"] = "reference"
        os.environ["TE_FL_PREFER_VENDOR"] = "1"  # Should be ignored
        policy2b = policy_from_env()
        self.assertEqual(policy2b.prefer, "reference")
        del os.environ["TE_FL_PREFER"]
        del os.environ["TE_FL_PREFER_VENDOR"]

        # 4. Block specific vendor
        os.environ["TE_FL_DENY_VENDORS"] = "acme"
        policy3 = policy_from_env()
        self.assertFalse(policy3.is_vendor_allowed("acme"))

        # 5. Allow only specific vendors
        os.environ["TE_FL_ALLOW_VENDORS"] = "nvidia,amd"
        del os.environ["TE_FL_DENY_VENDORS"]
        policy4 = policy_from_env()
        self.assertFalse(policy4.is_vendor_allowed("acme"))
        self.assertTrue(policy4.is_vendor_allowed("nvidia"))


def run_tests():
    """Run all out-of-tree vendor tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestOutOfTreeEntryPoints))
    suite.addTests(loader.loadTestsFromTestCase(TestOutOfTreeEnvModule))
    suite.addTests(loader.loadTestsFromTestCase(TestCompleteOutOfTreeExample))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
