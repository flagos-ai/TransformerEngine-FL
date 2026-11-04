# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Portable math/guard checks; device metadata is simulated in guard tests."""

import importlib.util
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest import mock

import torch

path = (
    Path(__file__).resolve().parents[4]
    / "transformer_engine/plugin/core/backends/vendor/hygon/expert_gemm.py"
)
spec = importlib.util.spec_from_file_location("hygon_expert_gemm_test", path)
gemm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gemm)


def call_args(layout, shape=(1024, 2048)):
    """Use meta tensors for production dimensions without allocating GPU memory."""
    n, k = shape
    splits = [0, 1, 3, 7] * 8
    weights = [torch.empty(shape, dtype=torch.bfloat16, device="meta") for _ in splits]
    x = [torch.empty((m, k), dtype=torch.bfloat16, device="meta") for m in splits]
    dy = [torch.empty((m, n), dtype=torch.bfloat16, device="meta") for m in splits]
    empty = [torch.empty(0)] * 32
    if layout == "NT":
        A, B, out = x, dy, weights
    else:
        A, B = weights, x if layout == "TN" else dy
        out = [
            torch.empty(
                (sum(splits), n if layout == "TN" else k),
                dtype=torch.bfloat16,
                device="meta",
            )
        ]
    return dict(
        A=A,
        transa=layout[0] == "T",
        B=B,
        transb=layout[1] == "T",
        out=out,
        m_splits=splits,
        bias=empty,
        single_output=layout != "NT",
        pre_gelu_out=empty,
        grad=layout != "TN",
        accumulate=False,
        use_split_accumulator=False,
    )


class TestExpertGuards(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(mock.patch.dict("os.environ", {"NVTE_HYGON_EXPERT_GEMM": "1"}))
        self.stack.enter_context(mock.patch.object(torch.version, "hip", "test"))
        self.stack.enter_context(
            mock.patch.object(torch.compiler, "is_compiling", return_value=False)
        )
        self.stack.enter_context(
            mock.patch.object(
                torch.Tensor, "is_cuda", new=property(lambda t: t.device.type == "meta")
            )
        )
        self.stack.enter_context(
            mock.patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext())
        )
        self.stack.enter_context(
            mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        self.device_check = self.stack.enter_context(
            mock.patch.object(gemm, "_is_bw1100", return_value=True)
        )

    def test_all_measured_layouts_and_weights(self):
        for shape in ((1024, 2048), (2048, 512)):
            for layout in ("TN", "NN", "NT"):
                for split_accumulator in (False, True):
                    with self.subTest(
                        shape=shape, layout=layout, split_accumulator=split_accumulator
                    ):
                        args = call_args(layout, shape)
                        args["use_split_accumulator"] = split_accumulator
                        self.assertTrue(gemm._supports_expert_gemm(**args))

    def test_flags_and_fused_operations_fall_back_before_execution(self):
        for field, value in (
            ("accumulate", True),
            ("grad", True),
            ("single_output", False),
            ("bias", [torch.ones(1)] * 32),
            ("pre_gelu_out", [torch.ones(1)] * 32),
        ):
            args = call_args("TN")
            args[field] = value
            with self.subTest(field=field), mock.patch.object(gemm, "_run_expert_gemm") as run:
                self.assertFalse(gemm.try_expert_gemm(**args))
                run.assert_not_called()

    def test_invalid_shapes_splits_dtype_and_layout_fall_back(self):
        mutations = (
            lambda a: a.update(m_splits=[-1] + a["m_splits"][1:]),
            lambda a: a.update(m_splits=a["m_splits"][:-1]),
            lambda a: a.update(A=a["A"][:-1]),
            lambda a: a["B"].__setitem__(
                1, torch.empty((2, 2048), dtype=torch.bfloat16, device="meta")
            ),
            lambda a: a["A"].__setitem__(0, a["A"][0].float()),
            lambda a: a["A"].__setitem__(
                0, torch.empty((2048, 1024), dtype=torch.bfloat16, device="meta").T
            ),
            lambda a: a.update(out=[torch.empty((1, 1024), dtype=torch.bfloat16, device="meta")]),
            lambda a: a["B"].__setitem__(0, torch.empty((0, 2048), dtype=torch.bfloat16)),
        )
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                args = call_args("TN")
                mutate(args)
                self.assertFalse(gemm._supports_expert_gemm(**args))
        self.assertFalse(gemm._supports_expert_gemm(**call_args("TN", (512, 1024))))

    def test_disable_device_compile_capture_and_autograd_fallback(self):
        args = call_args("TN")
        with mock.patch.dict("os.environ", {"NVTE_HYGON_EXPERT_GEMM": "0"}):
            self.assertFalse(gemm._supports_expert_gemm(**args))
        with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(gemm._supports_expert_gemm(**args))
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertFalse(gemm._supports_expert_gemm(**args))
        with torch.enable_grad():
            self.assertFalse(gemm._supports_expert_gemm(**args))
        self.device_check.return_value = False
        self.assertFalse(gemm._supports_expert_gemm(**args))

    def test_execution_errors_propagate(self):
        with mock.patch.object(
            gemm, "_run_expert_gemm", side_effect=RuntimeError("BLAS failure")
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "BLAS failure"):
                gemm.try_expert_gemm(**call_args("TN"))
            run.assert_called_once()


class TestExpertMath(unittest.TestCase):
    def test_forward_dgrad_wgrad_and_empty_experts(self):
        generator = torch.Generator().manual_seed(5)
        splits = [0, 1, 3, 7] * 8
        k, n = 8, 12
        weights = [torch.randn(n, k, generator=generator).bfloat16() for _ in splits]
        inputs = [torch.randn(m, k, generator=generator).bfloat16() for m in splits]
        grads = [torch.randn(m, n, generator=generator).bfloat16() for m in splits]
        with torch.no_grad():
            y = [torch.full((sum(splits), n), float("nan"), dtype=torch.bfloat16)]
            dx = [torch.full((sum(splits), k), float("nan"), dtype=torch.bfloat16)]
            dw = [torch.full_like(w, float("nan")) for w in weights]
            gemm._run_expert_gemm(weights, True, inputs, False, y, splits)
            gemm._run_expert_gemm(weights, False, grads, False, dx, splits)
            gemm._run_expert_gemm(inputs, False, grads, True, dw, splits)
            for w, x, dy, actual_y, actual_dx, actual_dw in zip(
                weights, inputs, grads, y[0].split(splits), dx[0].split(splits), dw
            ):
                torch.testing.assert_close(
                    actual_y.float(), (x.float() @ w.float().T).bfloat16().float()
                )
                torch.testing.assert_close(
                    actual_dx.float(), (dy.float() @ w.float()).bfloat16().float()
                )
                torch.testing.assert_close(
                    actual_dw.float(), (dy.float().T @ x.float()).bfloat16().float()
                )

    def test_bf16_wgrad_then_fp32_microbatch_accumulation(self):
        generator = torch.Generator().manual_seed(8)
        main_grad = [torch.zeros(4, 6, dtype=torch.float32) for _ in range(32)]
        reference = [torch.zeros_like(t) for t in main_grad]
        with torch.no_grad():
            for step in range(3):
                splits = [0 if step == 2 else (i + step) % 5 for i in range(32)]
                x = [torch.randn(m, 6, generator=generator).bfloat16() for m in splits]
                dy = [torch.randn(m, 4, generator=generator).bfloat16() for m in splits]
                dw = [torch.full((4, 6), float("nan"), dtype=torch.bfloat16) for _ in splits]
                gemm._run_expert_gemm(x, False, dy, True, dw, splits)
                for i in range(32):
                    main_grad[i].add_(dw[i])
                    reference[i].add_((dy[i].float().T @ x[i].float()).bfloat16().float())
            for actual, expected in zip(main_grad, reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
