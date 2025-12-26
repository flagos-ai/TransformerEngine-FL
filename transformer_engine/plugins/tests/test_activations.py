# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
import torch.nn.functional as F
import sys

from transformer_engine.plugins.test_utils import (
    get_available_backends,
    get_backend,
    TestCase,
    generate_random_tensor,
    generate_test_shapes,
)


class ActivationTests(TestCase):
    def __init__(self, device="cpu"):
        super().__init__(
            "Activation Functions",
            "Test correctness of all activation functions across backends"
        )
        self.backends = get_available_backends()
        self.reference_backend = "reference"
        self.device = device

    def _get_reference_gelu(self, x):
        return F.gelu(x, approximate='tanh')

    def _get_reference_geglu(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.gelu(a, approximate='tanh') * b

    def _get_reference_qgelu(self, x):
        return x * torch.sigmoid(1.702 * x)

    def _get_reference_relu(self, x):
        return F.relu(x)

    def _get_reference_silu(self, x):
        return F.silu(x)

    def _get_reference_swiglu(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

    def test_gelu_forward(self, shape=(4, 8)):
        print(f"\n  Testing GELU forward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_gelu(x)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.gelu(x, None)
                self.assert_close(
                    output, reference, rtol=1e-4, atol=1e-6,
                    msg=f"GELU forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_gelu_backward(self, shape=(4, 8)):
        print(f"\n  Testing GELU backward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)

        y = self._get_reference_gelu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                x_copy = x.detach()
                grad_input = backend.dgelu(grad_output, x_copy, None)
                self.assert_close(
                    grad_input, reference_grad, rtol=1e-4, atol=1e-6,
                    msg=f"GELU backward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_geglu_forward(self, shape=(4, 16)):
        print(f"\n  Testing GEGLU forward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_geglu(x)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.geglu(x, None)
                self.assert_close(
                    output, reference, rtol=1e-4, atol=1e-6,
                    msg=f"GEGLU forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_relu_forward(self, shape=(4, 8)):
        print(f"\n  Testing ReLU forward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_relu(x)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.relu(x, None)
                self.assert_close(
                    output, reference, rtol=1e-6, atol=1e-8,
                    msg=f"ReLU forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_silu_forward(self, shape=(4, 8)):
        print(f"\n  Testing SiLU forward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_silu(x)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.silu(x, None)
                self.assert_close(
                    output, reference, rtol=1e-5, atol=1e-7,
                    msg=f"SiLU forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_swiglu_forward(self, shape=(4, 16)):
        print(f"\n  Testing SwiGLU forward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_swiglu(x)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.swiglu(x, None)
                self.assert_close(
                    output, reference, rtol=1e-5, atol=1e-7,
                    msg=f"SwiGLU forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def run_all_tests(self):
        print("\n" + "="*60)
        print("Testing Activation Functions")
        print("="*60)
        print(f"Available backends: {', '.join(self.backends)}")

        shapes = [(4, 8), (8, 16), (2, 4, 8)]
        glu_shapes = [(4, 16), (8, 32), (2, 4, 16)]

        for shape in shapes:
            self.test_gelu_forward(shape)
            self.test_gelu_backward(shape)
            self.test_relu_forward(shape)
            self.test_silu_forward(shape)

        for shape in glu_shapes:
            self.test_geglu_forward(shape)
            self.test_swiglu_forward(shape)

        return self.report()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_suite = ActivationTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
