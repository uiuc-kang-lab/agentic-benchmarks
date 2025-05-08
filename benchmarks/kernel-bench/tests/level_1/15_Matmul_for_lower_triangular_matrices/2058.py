
import torch
import pytest
from torch.utils.cpp_extension import load

# Function to build the CUDA module from kernel.cu
def build_kernel():
    return load(
        name="triangular_mm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Kernel assumes float32 input.
def test_dtype_issue():
    # Create double-precision lower triangular matrices.
    N = 128
    A = torch.tril(torch.randn(N, N, dtype=torch.float64, device='cuda'))
    B = torch.tril(torch.randn(N, N, dtype=torch.float64, device='cuda'))
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel is not set up for float64.
        C = module.forward(A, B)
        # Force synchronization to catch any errors.
        torch.cuda.synchronize()

# Issue 2: Kernel hard-codes lower-triangular behavior leading to incompatibility with general matrix multiplication.
def test_triangular_only_behavior():
    # Create full (non-triangular) matrices.
    N = 256
    A_full = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B_full = torch.randn(N, N, device="cuda", dtype=torch.float32)
    # Even if A and B are full matrices, the kernel will output only the lower triangular part.
    # Hence, the expected result is torch.tril(torch.matmul(A_full, B_full)).
    expected = torch.tril(torch.matmul(A_full, B_full))
    
    module = build_kernel()
    C = module.forward(torch.tril(A_full), torch.tril(B_full))
    torch.cuda.synchronize()
    
    # Use a tolerance for floating point comparisons.
    assert torch.allclose(C, expected, atol=1e-5), \
        f"Kernel output does not match reference lower triangular multiplication."

# Issue 3: Inefficient tile loading not optimized for triangular matrices.
# To trigger any potential mis-calculation in the boundary (when dimensions are not divisible by TILE_SIZE),
# we test with a matrix size that is not a multiple of TILE_SIZE.
def test_boundary_tile_loading():
    # Choose a size not divisible by typical TILE_SIZE (32)
    N = 70
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A_tri = torch.tril(A)
    B_tri = torch.tril(B)
    expected = torch.tril(torch.matmul(A_tri, B_tri))
    
    module = build_kernel()
    C = module.forward(A_tri, B_tri)
    torch.cuda.synchronize()
    
    assert torch.allclose(C, expected, atol=1e-5), \
        f"Kernel output differs at boundaries! Max diff: {(C - expected).abs().max().item()}"
