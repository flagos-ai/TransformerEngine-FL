# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
import torch.nn.functional as F

from transformer_engine.plugin.test_utils import (
    get_available_backends,
    get_backend,
    TestCase,
    generate_random_tensor,
)


class moeTests(TestCase):
    def __init__(self, device="cpu"):
        super().__init__(
            "Moe permute Operations",
            "Test correctness of all moe permute operations across backends",
        )
        self.backends = get_available_backends()
        self.device = device

    def test_moe_permute_fw_basic(self, num_tokens=8, num_cols=256, topK=4, num_out_tokens=-1):
        print(
            f"\nTesting moe_permute_fw_basic (tokens={num_tokens}, cols={num_cols},"
            f" topK={topK},num_out_tokens={num_out_tokens})"
        )
        import transformer_engine_torch_nv as te

        input_tensor = generate_random_tensor(
            (num_tokens, num_cols), dtype=torch.float16, device=self.device
        )
        indices = torch.randint(0, 8, (num_tokens, topK), dtype=torch.int32, device=self.device)

        if input_tensor.dtype == torch.float16:
            dtype = te.DType.kFloat16
        elif input_tensor.dtype == torch.float32:
            dtype = te.DType.kFloat32
        elif input_tensor.dtype == torch.bfloat16:
            dtype = te.DType.kBFloat16
        else:
            raise ValueError("Unsupported dtype")

        workspace = []
        max_expanded_token_num = num_tokens * topK + 10

        reference_permuted, reference_row_id_map, reference_workspace = te.moe_permute_fwd(
            input_tensor, dtype, indices, num_out_tokens, workspace, max_expanded_token_num
        )
        for backend_name in self.backends:
            backend = get_backend(backend_name)
            print("backend:", backend)
            try:
                permuted_output, row_id_map, workspace_out = backend.moe_permute_fwd(
                    input_tensor, dtype, indices, num_out_tokens, workspace, max_expanded_token_num
                )
                self.assert_close(
                    reference_permuted,
                    permuted_output,
                    rtol=1e-2,
                    atol=1e-3,
                    msg=f"moe_permute_fwd mismatch for {backend_name}",
                )
                self.assert_close(
                    row_id_map,
                    reference_row_id_map,
                    rtol=1e-2,
                    atol=1e-3,
                    msg=f"moe_permute_fwd mismatch for {backend_name}",
                )

                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"  ✗ Test failed: {e}")

    def test_moe_permute_fw_out_token(self, num_tokens=8, num_cols=256, topK=4, num_out_tokens=8):
        print(
            f"\nTesting moe_permute_fw_out_token (tokens={num_tokens}, cols={num_cols},"
            f" topK={topK},num_out_tokens={num_out_tokens})"
        )
        import transformer_engine_torch_nv as te

        input_tensor = generate_random_tensor(
            (num_tokens, num_cols), dtype=torch.float16, device=self.device
        )
        indices = torch.randint(0, 8, (num_tokens, topK), dtype=torch.int32, device=self.device)

        if input_tensor.dtype == torch.float16:
            dtype = te.DType.kFloat16
        elif input_tensor.dtype == torch.float32:
            dtype = te.DType.kFloat32
        elif input_tensor.dtype == torch.bfloat16:
            dtype = te.DType.kBFloat16
        else:
            raise ValueError("Unsupported dtype")

        workspace = []
        max_expanded_token_num = num_tokens * topK + 10

        reference_permuted, reference_row_id_map, reference_workspace = te.moe_permute_fwd(
            input_tensor, dtype, indices, num_out_tokens, workspace, max_expanded_token_num
        )
        for backend_name in self.backends:
            backend = get_backend(backend_name)
            print("backend:", backend)
            try:
                permuted_output, row_id_map, workspace_out = backend.moe_permute_fwd(
                    input_tensor, dtype, indices, num_out_tokens, workspace, max_expanded_token_num
                )
                self.assert_close(
                    reference_permuted,
                    permuted_output,
                    rtol=1e-2,
                    atol=1e-3,
                    msg=f"moe_permute_fwd_out_token mismatch for {backend_name}",
                )
                self.assert_close(
                    row_id_map,
                    reference_row_id_map,
                    rtol=1e-2,
                    atol=1e-3,
                    msg=f"moe_permute_fwd_out_token mismatch for {backend_name}",
                )

                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"  ✗ Test failed: {e}")

    def test_moe_unpermute_bw_basic(self, num_tokens=8, num_cols=256, topK=4, num_out_tokens=8):
        print(
            f"\nTesting moe_unpermute_bw_basic (tokens={num_tokens}, cols={num_cols}, topK={topK},"
            f" num_out_tokens={num_out_tokens})"
        )
        import transformer_engine_torch_nv as te

        input_fwd = generate_random_tensor(
            (num_tokens, num_cols), dtype=torch.float16, device=self.device
        )
        indices = torch.randint(0, 8, (num_tokens, topK), dtype=torch.int32, device=self.device)

        prob = torch.rand((num_tokens, topK), dtype=torch.float32, device=self.device)
        prob = prob / prob.sum(dim=1, keepdim=True)

        if input_fwd.dtype == torch.float16:
            dtype = te.DType.kFloat16
        elif input_fwd.dtype == torch.float32:
            dtype = te.DType.kFloat32
        elif input_fwd.dtype == torch.bfloat16:
            dtype = te.DType.kBFloat16
        else:
            raise ValueError("Unsupported dtype")

        workspace = []
        max_expanded_token_num = num_tokens * topK + 10

        _, row_id_map, _ = te.moe_permute_fwd(
            input_fwd, dtype, indices, num_out_tokens, workspace, max_expanded_token_num
        )

        input_bwd = generate_random_tensor(
            (num_out_tokens, num_cols), dtype=input_fwd.dtype, device=self.device
        )

        reference_act_grad, reference_prob_grad = te.moe_unpermute_bwd(
            input_bwd, input_fwd, dtype, row_id_map, prob
        )
        for backend_name in self.backends:
            backend = get_backend(backend_name)
            print("backend:", backend)
            try:
                act_grad, prob_grad = backend.moe_unpermute_bwd(
                    input_bwd, input_fwd, dtype, row_id_map, prob
                )
                self.assert_close(
                    act_grad,
                    reference_act_grad,
                    rtol=1e-2,
                    atol=1e-3,
                    msg=f"moe_unpermute_bw_basic mismatch for {backend_name}",
                )
                self.assert_close(
                    prob_grad,
                    reference_prob_grad,
                    rtol=1e-2,
                    atol=1e-3,
                    msg=f"moe_unpermute_bw_basic mismatch for {backend_name}",
                )

                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"  ✗ Test failed: {e}")

    def test_moe_unpermute_fw_basic(self, num_tokens=4, num_cols=8, topK=1, num_out_tokens=2):
        print(
            f"\nTesting moe_unpermute_fw_basic (tokens={num_tokens}, cols={num_cols},"
            f" topK={topK},num_out_tokens={num_out_tokens})"
        )
        import transformer_engine_torch_nv as te

        input_tensor = generate_random_tensor(
            (num_tokens, num_cols), dtype=torch.float16, device=self.device
        )
        probs = torch.rand((num_tokens, topK), device=self.device, dtype=torch.float32)
        probs = probs / probs.sum(dim=1, keepdim=True)
        # probs = torch.empty(0)

        indices = torch.randint(0, 8, (num_tokens, topK), dtype=torch.int32, device=self.device)

        if input_tensor.dtype == torch.float16:
            dtype = te.DType.kFloat16
        elif input_tensor.dtype == torch.float32:
            dtype = te.DType.kFloat32
        elif input_tensor.dtype == torch.bfloat16:
            dtype = te.DType.kBFloat16
        else:
            raise ValueError("Unsupported dtype")

        workspace = []
        max_expanded_token_num = num_tokens * topK + 10

        permuted, row_id_map, _ = te.moe_permute_fwd(
            input_tensor, dtype, indices, num_out_tokens, workspace, max_expanded_token_num
        )
        reference_output = te.moe_unpermute_fwd(
            permuted, dtype, row_id_map, probs, num_tokens, topK
        )
        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                moe_output = backend.moe_unpermute_fwd(
                    permuted, dtype, row_id_map, probs, num_tokens, topK
                )
                self.assert_close(
                    moe_output,
                    reference_output,
                    rtol=1e-2,
                    atol=1e-3,
                    msg=f"moe_unpermute_fw_basic mismatch for {backend_name}",
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"  ✗ Test failed: {e}")

    def test_moe_permute_bw_basic(self, num_tokens=8, num_cols=256, topK=4, num_out_tokens=8):
        print(
            f"\nTesting moe_permute_bw_basic (tokens={num_tokens}, cols={num_cols},"
            f" topK={topK},num_out_tokens={num_out_tokens})"
        )
        import transformer_engine_torch_nv as te

        input_tensor = generate_random_tensor(
            (num_tokens, num_cols), dtype=torch.float16, device=self.device
        )
        # cuda kernel :probs must float32
        probs = generate_random_tensor((num_tokens, topK), dtype=torch.float32, device=self.device)
        probs = probs / probs.sum(dim=1, keepdim=True)

        indices = torch.randint(0, 8, (num_tokens, topK), dtype=torch.int32, device=self.device)

        if input_tensor.dtype == torch.float16:
            dtype = te.DType.kFloat16
        elif input_tensor.dtype == torch.float32:
            dtype = te.DType.kFloat32
        elif input_tensor.dtype == torch.bfloat16:
            dtype = te.DType.kBFloat16
        else:
            raise ValueError("Unsupported dtype")

        workspace = []
        max_expanded_token_num = num_tokens * topK + 10

        permuted, row_id_map, _ = te.moe_permute_fwd(
            input_tensor, dtype, indices, num_out_tokens, workspace, max_expanded_token_num
        )
        reference_output = te.moe_permute_bwd(permuted, dtype, row_id_map, probs, num_tokens, topK)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                moe_output = backend.moe_permute_bwd(
                    permuted, dtype, row_id_map, probs, num_tokens, topK
                )
                self.assert_close(
                    moe_output,
                    reference_output,
                    rtol=1e-2,
                    atol=1e-3,
                    msg=f"moe_permute_bw_basic mismatch for {backend_name}",
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"  ✗ Test failed: {e}")

    def run_all_tests(self):
        print("\n" + "=" * 60)
        print("=" * 60)
        print(f"Available backends: {', '.join(self.backends)}")

        # moe permute forward tests
        # self.test_moe_permute_fw_basic()
        # self.test_moe_permute_fw_out_token()
        # self.test_moe_permute_bw_basic()
        # self.test_moe_unpermute_fw_basic()
        self.test_moe_unpermute_bw_basic()

        return self.report()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_suite = moeTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
