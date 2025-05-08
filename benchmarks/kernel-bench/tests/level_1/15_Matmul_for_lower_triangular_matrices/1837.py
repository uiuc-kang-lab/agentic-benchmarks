
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="triangular_mm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

def ref_triangular_mm(A, B):
    # Produces the same result as the PyTorch model: lower triangular matrix multiplication.
    C = torch.matmul(A, B)
    return torch.tril(C)

# Test case 1: Trigger the lack of type‐safety by passing double (float64) tensors.
def test_dtype_issue():
    my_module = build_kernel()
    N = 128
    # Create double tensors
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    A = torch.tril(A)
    B = torch.tril(B)
    
    # The kernel expects float32. Passing float64 will reinterpret the memory (without error).
    C_kernel = my_module.forward(A, B)
    
    # Compute reference using torch.double multiplication and conversion to lower triangular
    C_ref = ref_triangular_mm(A, B)
    
    # The results are expected to be different beyond any acceptable tolerance.
    max_diff = (C_kernel - C_ref).abs().max().item()
    assert max_diff > 1e-3, (
        f"Test failed: Kernel returned nearly correct results for float64 input! "
        f"max diff = {max_diff}"
    )

# Test case 2: Use a matrix with size not divisible by the launch block dimension (16)
def test_non_divisible_dimension():
    my_module = build_kernel()
    N = 17  # 17 is not divisible by 16, so boundary threads are exercised.
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, N, dtype=torch.float32, device='cuda')
    A = torch.tril(A)
    B = torch.tril(B)
    
    C_kernel = my_module.forward(A, B)
    C_ref = ref_triangular_mm(A, B)
    
    # The kernel should correctly handle boundaries. If not, this test will fail.
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel output differs from reference output for non-divisible matrix dimensions!"
    )
