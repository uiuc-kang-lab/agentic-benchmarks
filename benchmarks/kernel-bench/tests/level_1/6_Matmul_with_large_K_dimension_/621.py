
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Non-contiguous input tensors
def test_noncontiguous_input():
    cuda_module = build_kernel()
    M, K, N = 128, 128, 128
    # Create contiguous inputs first
    A_contig = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_contig = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Make them noncontiguous. For example, take the transpose or narrow a view.
    A_noncontig = A_contig.t()  # Now shape is (K, M) and noncontiguous!
    B_noncontig = B_contig.t()  # Now shape is (N, K)
    # For the purpose of triggering the issue, we force a view that is noncontiguous
    # (Even though the kernel is written for [M,K] and [K,N], here we deliberately misuse it)
    with pytest.raises(Exception):
        # Since our kernel expects A.size(0)==M and A.size(1)==K, a noncontiguous or wrong shaped tensor
        # will produce an incorrect result. We therefore compare against the expected result.
        # In this test we expect a failure (an incorrect result) so we can use an assertion that forces a mismatch.
        C = cuda_module.forward(A_noncontig, B_noncontig)
        torch.cuda.synchronize()
        C_ref = torch.matmul(A_noncontig, B_noncontig)
        # The error is in the computation (or even a crash) because the indexing from pointer arithmetic is wrong.
        # Thus if the kernel were correct for noncontiguous data, C would equal C_ref.
        assert not torch.allclose(C, C_ref, atol=1e-4), (
            "Kernel unexpectedly produced correct result for noncontiguous inputs (it should fail)."
        )

# Issue 2: Lack of batched matrix multiplication support.
def test_batched_input_not_supported():
    cuda_module = build_kernel()
    batch, M, K, N = 4, 64, 64, 64
    # Create batched inputs.
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, N, device="cuda", dtype=torch.float32)
    # The kernel is designed for 2D matrices only.
    with pytest.raises(IndexError):
        # We expect that if we pass a 3D tensor to the kernel, the kernel interface (which takes A.size(0) as M)
        # will trigger an index error (or an incorrect shape error) because dimensions are mismatched.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Unsupported data type (half precision)
def test_half_precision_not_supported():
    cuda_module = build_kernel()
    M, K, N = 32, 32, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # AT_DISPATCH_FLOATING_TYPES does not include float16; so calling kernel.forward with FP16 should raise an error.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()
