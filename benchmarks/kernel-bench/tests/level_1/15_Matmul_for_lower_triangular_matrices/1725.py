
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Compile the CUDA extension; note that verbose output is enabled for troubleshooting.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_matrix_incorrect_threads():
    """
    Test case 1: Use a matrix with dimensions greater than the fixed threads per block (e.g. 100 > 64)
    Expected behavior: Because the kernel only launches 64 threads per row, the computed lower triangular part
    will be incomplete and differ from the reference torch.tril(torch.matmul(A, B)).
    """
    N = 100  # N > 64 so that columns >= 64 are never processed.
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A, B))
    # Check if any element in C (especially for column indices >= 64) is not matching the reference.
    max_diff = (C - C_ref).abs().max().item()
    assert not torch.allclose(C, C_ref, atol=1e-5), f"Kernel output unexpectedly matches reference! Max diff: {max_diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incomplete_zeroing():
    """
    Test case 2: Check that the upper triangular part is not completely zeroed out when N > blockDim.x.
    For a matrix with N > 64, the kernel is supposed to zero out all positions (i, j) with j > i.
    However, since only threads for j < 64 are launched, some elements may remain uninitialized.
    """
    N = 80  # Choose N such that there are columns beyond the fixed threadsPerBlock (64).
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # Now, the upper triangle (j > i) should be 0. We check for a sample row where i < 64 but also check rows i >= 64.
    for i in range(N):
        for j in range(N):
            if j > i:
                # For j < 64, kernel sets C[i,j]=0; for j>=64, they were never written.
                if j < 64:
                    assert torch.abs(C[i, j]) < 1e-5, f"Element C[{i}, {j}] expected to be zero, but got {C[i, j].item()}"
                else:
                    # Elements in columns j >= 64 are uninitialized; they likely don't equal zero.
                    assert torch.abs(C[i, j]) > 1e-5, f"Element C[{i}, {j}] unexpectedly zero, which may indicate accidental initialization."

