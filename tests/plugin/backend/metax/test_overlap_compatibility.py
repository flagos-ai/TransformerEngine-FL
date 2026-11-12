# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""CPU contract regressions for MetaX legacy Userbuffer adaptation.

Run in the matching TE-FL environment without GPU devices.
Real pybind/GPU execution is covered separately by the recorded UB/numeric runs.
"""

import enum
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformer_engine.plugin.core.backends.vendor.metax import metax
from transformer_engine.plugin.core.backends.vendor.metax import register_ops


class FrontendComm(enum.IntEnum):
    AG = 0
    RS = 1


class NativeComm(enum.IntEnum):
    AG = 0
    RS = 1


class LegacyBulk:
    def __init__(self, *args):
        self.args = args

    def native_buffer(self):
        return self.args[0]


class LegacyP2P(LegacyBulk):
    def __init__(self, *args):
        if type(args[4]) is not NativeComm:
            raise TypeError("native constructor requires native enum identity")
        super().__init__(*args)


class ModernBulk(LegacyBulk):
    def with_cublasmp(self):
        return True


class ModernP2P(LegacyP2P):
    def with_cublasmp(self):
        return True


class RegistryCapture:
    def __init__(self):
        self.impls = []

    def register_many(self, impls):
        self.impls.extend(impls)


class MetaXOverlapCompatibility(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def multicast(device=-1):
            self.calls.append(("multicast", device))
            return device == 3

        def mpi():
            self.calls.append(("mpi",))
            return True

        def priority(device=-1):
            self.calls.append(("priority", device))
            return (0, -2)

        self.native = SimpleNamespace(
            CommOverlap=LegacyBulk,
            CommOverlapP2P=LegacyP2P,
            CommOverlapType=NativeComm,
            device_supports_multicast=multicast,
            ubuf_built_with_mpi=mpi,
            get_stream_priority_range=priority,
        )
        self.backend = metax.MetaxBackend()
        self.backend._get_tex = lambda: self.native
        self.shape = [8192, 5120]
        self.helper = object()

    def test_capability_queries_preserve_arguments_and_values(self):
        self.assertFalse(self.backend.device_supports_multicast())
        self.assertTrue(self.backend.device_supports_multicast(3))
        self.assertTrue(self.backend.ubuf_built_with_mpi())
        self.assertEqual(self.backend.get_stream_priority_range(), (0, -2))
        self.assertEqual(self.backend.get_stream_priority_range(3), (0, -2))
        self.assertEqual(
            self.calls,
            [
                ("multicast", -1),
                ("multicast", 3),
                ("mpi",),
                ("priority", -1),
                ("priority", 3),
            ],
        )

    def test_capability_vendor_registrations_are_callable(self):
        registry = RegistryCapture()
        with patch.object(metax.MetaxBackend, "is_available", return_value=True), patch.object(
            metax.MetaxBackend, "_get_tex", return_value=self.native
        ), patch.object(register_ops.torch.version, "maca", "3.8.1.0", create=True):
            register_ops.register_builtins(registry)
            wanted = {
                "device_supports_multicast",
                "ubuf_built_with_mpi",
                "get_stream_priority_range",
            }
            impls = [i for i in registry.impls if i.op_name in wanted]
            self.assertEqual({i.op_name for i in impls}, wanted)
            self.assertEqual(len(impls), 3)
            by_name = {i.op_name: i for i in impls}
            for impl in impls:
                self.assertEqual(impl.impl_id, "vendor.metax")
                self.assertEqual(impl.vendor, "METAX")
                self.assertTrue(impl.fn._is_available())
            self.assertTrue(by_name["device_supports_multicast"].fn(3))
            self.assertTrue(by_name["ubuf_built_with_mpi"].fn())
            self.assertEqual(by_name["get_stream_priority_range"].fn(3), (0, -2))

    def test_unavailable_backend_registers_nothing(self):
        registry = RegistryCapture()
        with patch.object(metax.MetaxBackend, "is_available", return_value=False):
            register_ops.register_builtins(registry)
        self.assertEqual(registry.impls, [])

    def test_legacy_bulk_retains_native_type_and_constructor_contract(self):
        result = self.backend.create_comm_overlap(
            self.shape,
            torch.bfloat16,
            self.helper,
            2,
            4,
            5,
            1,
            -1,
            -2,
            1,
            False,
            False,
            True,
            comm_type=FrontendComm.RS,
        )
        self.assertIsInstance(result, LegacyBulk)
        self.assertEqual(
            result.args,
            (
                self.shape,
                torch.bfloat16,
                self.helper,
                2,
                4,
                5,
                1,
                -1,
                -2,
                1,
                False,
                False,
                True,
            ),
        )
        self.assertIs(result.native_buffer(), self.shape)
        self.assertIs(result.with_cublasmp(), False)

    def test_p2p_accepts_frontend_enum_for_both_collectives(self):
        for comm in FrontendComm:
            with self.subTest(comm=comm):
                result = self.backend.create_comm_overlap_p2p(
                    self.shape,
                    torch.bfloat16,
                    self.helper,
                    2,
                    comm,
                    4,
                    1,
                    -1,
                    -2,
                    1,
                    False,
                    False,
                    True,
                    True,
                )
                self.assertIsInstance(result, LegacyP2P)
                self.assertIs(type(result.args[4]), NativeComm)
                self.assertEqual(int(result.args[4]), int(comm))
                self.assertEqual(result.args[:4], (self.shape, torch.bfloat16, self.helper, 2))
                self.assertEqual(result.args[5:], (4, 1, -1, -2, 1, False, False, True, True))
                self.assertIs(result.with_cublasmp(), False)

    def test_p2p_also_accepts_native_enum(self):
        result = self.backend.create_comm_overlap_p2p(
            self.shape, torch.bfloat16, self.helper, 2, NativeComm.RS
        )
        self.assertIs(result.args[4], NativeComm.RS)
        self.assertIs(result.with_cublasmp(), False)

    def test_existing_native_capability_is_preserved(self):
        self.native.CommOverlap = ModernBulk
        self.native.CommOverlapP2P = ModernP2P
        bulk = self.backend.create_comm_overlap(self.shape, torch.bfloat16, self.helper, 2)
        p2p = self.backend.create_comm_overlap_p2p(
            self.shape, torch.bfloat16, self.helper, 2, NativeComm.AG
        )
        self.assertIs(type(bulk), ModernBulk)
        self.assertIs(type(p2p), ModernP2P)
        self.assertTrue(bulk.with_cublasmp())
        self.assertTrue(p2p.with_cublasmp())

    def test_legacy_adapter_reuses_type_without_global_native_mutation(self):
        a = self.backend.create_comm_overlap(self.shape, torch.bfloat16, self.helper, 2)
        b = self.backend.create_comm_overlap(self.shape, torch.bfloat16, self.helper, 2)
        self.assertIs(type(a), type(b))
        self.assertFalse(hasattr(LegacyBulk, "with_cublasmp"))
        self.assertIs(a.with_cublasmp(), False)

    def test_unsupported_cublasmp_fails_before_native_allocation(self):
        def unexpected():
            raise AssertionError("native allocation must not be attempted")

        self.backend._get_tex = unexpected
        for name, extra in [
            ("create_comm_overlap", []),
            ("create_comm_overlap_p2p", [FrontendComm.AG]),
        ]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(NotImplementedError, "cuBLASMp"):
                    getattr(self.backend, name)(
                        self.shape, torch.bfloat16, self.helper, 2, *extra, use_cublasmp=True
                    )

    def test_queries_require_maca_runtime_identity(self):
        names = {"device_supports_multicast", "ubuf_built_with_mpi", "get_stream_priority_range"}
        versions = [
            (SimpleNamespace(), False),
            (SimpleNamespace(maca=None), False),
            (SimpleNamespace(maca=""), False),
            (SimpleNamespace(maca="3.8.1.0"), True),
        ]
        for version, expected in versions:
            with self.subTest(version=version):
                with patch.object(
                    register_ops, "torch", SimpleNamespace(version=version)
                ), patch.object(metax.MetaxBackend, "is_available", return_value=True):
                    registry = RegistryCapture()
                    register_ops.register_builtins(registry)
                    impls = [i for i in registry.impls if i.op_name in names]
                    self.assertEqual(len(impls), 3)
                    self.assertEqual([i.is_available() for i in impls], [expected] * 3)


if __name__ == "__main__":
    print("BACKEND_SOURCE", metax.__file__, flush=True)
    print("REGISTRATION_SOURCE", register_ops.__file__, flush=True)
    unittest.main(verbosity=2)
