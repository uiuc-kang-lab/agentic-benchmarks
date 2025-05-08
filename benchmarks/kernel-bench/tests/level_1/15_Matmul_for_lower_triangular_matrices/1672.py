
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Force rebuild to pick up local changes
    module = load(
        name="triangular_mm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
        is_python_module=False  # Explicitly not a python package module.
    )
    return module

@pytest.fixture(scope="module")
def cuda_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    mod = build_kernel()
    return mod

def test_float32_works(cuda_module):
    # This test uses the expected float32 type and 2D square matrices.
    N = 128
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    
    # Call the kernel function
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Verify lower-triangular property and correctness of multiplication for lower triangular matrices:
    C_ref = torch.matmul(A, B)
    # Only lower triangular part should be computed by our kernel.
    C_ref = torch.tril(C_ref)
    assert torch.allclose(C, C_ref, atol=1e-4), \
        f"Kernel output differs from reference output! Max diff: {(C - C_ref).abs().max()}"

def test_non_float32_dtype(cuda_module):
    # Pass double precision matrices.
    N = 64
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    
    with pytest.raises(RuntimeError) as excinfo:
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()
    # The error should complain about wrong input type.
    assert "must be a CUDA tensor" not in str(excinfo.value), "Unexpected error message"  # Checking that error is due to type incompatibility.

def test_non_2d_tensor(cuda_module):
    # Pass a 3D tensor to trigger the dimension check.
    N = 32
    A = torch.tril(torch.randn(4, N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(4, N, N, device="cuda", dtype=torch.float32))
    
    with pytest.raises(RuntimeError) as excinfo:
        # The kernel explicitly checks for 2D input.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()
    assert "must be a 2D tensor" in str(excinfo.value), "Expected 2D tensor check failure."

def test_non_square_tensor(cuda_module):
    # Pass non-square matrices.
    N = 64
    # Create a rectangular tensor by taking a slice.
    A_full = torch.tril(torch.randn(N+10, N+10, device="cuda", dtype=torch.float32))
    B_full = torch.tril(torch.randn(N+10, N+10, device="cuda", dtype=torch.float32))
    A = A_full[:N, :N+5]
    B = B_full[:N, :N+5]
    
    with pytest.raises(RuntimeError) as excinfo:
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()
    assert "must be square" in str(excinfo.value), "Expected square tensor check failure."

def test_grid_config_scalability(cuda_module):
    # Test with a matrix size that is not a multiple of 16 to trigger potential grid/block misconfiguration issues.
    N = 70  # 70 is not a multiple of 16.
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    
    # Even if the kernel computes correctly, the fixed block/grid shape may cause performance issues.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    
    C_ref = torch.tril(torch.matmul(A, B))
    assert torch.allclose(C, C_ref, atol=1e-3), \
        f"Kernel output for non-multiple-of-16 dimensions differs! Max diff: {(C - C_ref).abs().max()}"
