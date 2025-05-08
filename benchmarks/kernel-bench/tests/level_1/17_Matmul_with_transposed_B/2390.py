
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel(extra_cuda_cflags=None):
    if extra_cuda_cflags is None:
        extra_cuda_cflags = ["-O3", "--use_fast_math"]
    else:
        extra_cuda_cflags = ["-O3", "--use_fast_math"] + extra_cuda_cflags
    # Build and load the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def get_kernel_module():
    return build_kernel()

# Test case 1: Using an unsupported data type (e.g., float64) should trigger an error.
def test_float32_only():
    M, K, N = 128, 64, 32
    A = torch.randn(M, K, dtype=torch.double, device='cuda')
    B = torch.randn(N, K, dtype=torch.double, device='cuda')
    mod = get_kernel_module()
    with pytest.raises(RuntimeError):
        mod.forward(A, B)

# Test case 2: Using a K dimension that is not a multiple of BLOCK_SIZE or 4.
def test_non_multiple_K():
    # BLOCK_SIZE is defined as 16 and the inner loop is unrolled by 4.
    # Using K=67 (not divisible by 16 or 4) should trigger incorrect behavior (or wrong results).
    M, K, N = 128, 67, 32
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(N, K, dtype=torch.float32, device='cuda')
    mod = get_kernel_module()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B.t())
    # The outputs should be significantly different because of the issue.
    assert not torch.allclose(C, C_ref, atol=1e-3), \
        "Kernel output should be incorrect for non-multiple K due to loop unrolling issues."

# Test case 3: Using an N dimension that is not a multiple of BLOCK_SIZE*COARSENING.
def test_non_multiple_N():
    # BLOCK_SIZE*COARSENING = 16*2 = 32.
    # Use N=35 to force boundary conditions issues.
    M, K, N = 128, 64, 35
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(N, K, dtype=torch.float32, device='cuda')
    mod = get_kernel_module()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B.t())
    # With faulty boundary handling, the results are expected to differ.
    assert not torch.allclose(C, C_ref, atol=1e-3), \
        "Kernel output should be incorrect for non-multiple N due to boundary handling issues."

# Test case 4: Non-contiguous tensor input.
def test_non_contiguous():
    # The kernel checks for contiguous inputs and should raise an error.
    M, K, N = 128, 64, 32
    A = torch.randn(M, K, dtype=torch.float32, device='cuda').t()  # This makes it non-contiguous.
    B = torch.randn(N, K, dtype=torch.float32, device='cuda')
    mod = get_kernel_module()
    with pytest.raises(RuntimeError):
        mod.forward(A, B)

# Test case 5: Boundary conditions with small matrices.
def test_boundary_conditions():
    # Small matrices that do not align with the block tiling.
    M, K, N = 10, 10, 10
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(N, K, dtype=torch.float32, device='cuda')
    mod = get_kernel_module()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B.t())
    # In a correct kernel, these would be nearly equal.
    # Here we expect differences due to tiling and manual unrolling issues.
    assert not torch.allclose(C, C_ref, atol=1e-3), \
        "Kernel output should be incorrect for small matrix dimensions due to improper boundary handling."
