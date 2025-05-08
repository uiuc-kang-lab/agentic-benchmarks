
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: __ldg intrinsic with half precision (float16) might be problematic.
def test_half_precision():
    # Using float16 to trigger potential issues with __ldg usage.
    N, M, K, L = 4, 64, 128, 32
    A = torch.randn(N, M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, L, device='cuda', dtype=torch.float16)
    my_module = build_kernel()
    # Compute kernel output. The kernel may produce inaccurate results on half precision if __ldg isn't well supported.
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Reference result computed in higher precision then cast back to float16.
    C_ref = torch.matmul(A.float(), B.float()).to(torch.float16)
    max_diff = (C_kernel - C_ref).abs().max().item()
    assert max_diff < 1e-2, f"Kernel half precision result deviates too much from reference. Max diff: {max_diff}"

# Issue 2: Tensor A is assumed to be contiguous. Use a non-contiguous input to trigger a failure.
def test_non_contiguous():
    N, M, K, L = 4, 64, 128, 32
    A = torch.randn(N, M, K, device='cuda', dtype=torch.float32).permute(0, 2, 1)  # make non-contiguous
    B = torch.randn(K, L, device='cuda', dtype=torch.float32)
    my_module = build_kernel()

    with pytest.raises(RuntimeError):
        # The CHECK_CONTIGUOUS macro should trigger an error for a non-contiguous tensor.
        my_module.forward(A, B)

# Issue 3: Fixed tile dimensions may create problems when dimensions are not a multiple of TILE_DIM.
def test_non_multiple_tile_dim():
    # Choose dimensions that are not multiples of 32.
    N, M, K, L = 3, 50, 70, 45
    A = torch.randn(N, M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, L, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    max_diff = (C_kernel - C_ref).abs().max().item()
    assert max_diff < 1e-5, f"Kernel result for non-multiple tile dimensions is incorrect. Max diff: {max_diff}"

# Issue 4: Lack of dimension-checking for the inner dimensions (K mismatch).
def test_dimension_mismatch():
    # Create a mismatch: A.size(2) != B.size(0)
    N, M, K, L = 4, 64, 128, 32
    A = torch.randn(N, M, K, device='cuda', dtype=torch.float32)
    # Deliberately set mismatch in B's first dimension.
    B = torch.randn(K + 1, L, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting an error because the dimensions do not align for matrix multiplication.
        my_module.forward(A, B)
        
if __name__ == "__main__":
    pytest.main([__file__])
