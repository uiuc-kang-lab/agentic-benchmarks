
import pytest
import torch
from torch.utils.cpp_extension import load
import time

# Utility to compile and load the CUDA extension from kernel.cu.
def build_kernel():
    # For testing purposes, we force rebuild.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.mark.timeout(5)
def test_deadlock_due_to_early_return():
    """
    Issue 1: Early termination in threads with row < col may cause deadlock.
    This test creates a lower triangular matrix where many threads in a block have row<col.
    If the kernel deadlocks, this test will timeout.
    """
    my_module = build_kernel()
    N = 64  # A moderate size where block divergence is likely.
    # Create lower triangular matrices.
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    # Launch kernel.
    start_time = time.time()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()  # This will hang if deadlock occurs.
    elapsed = time.time() - start_time
    # If we get here, the kernel completed within the timeout.
    assert elapsed < 5, f"Kernel execution took too long ({elapsed:.2f} s), likely deadlocked."

def test_incorrect_dtype():
    """
    Issue 2: Kernel only supports float32.
    This test uses double precision, which should trigger a check error or produce incorrect results.
    """
    my_module = build_kernel()
    N = 128
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    A = torch.tril(A)
    B = torch.tril(B)
    with pytest.raises(RuntimeError):
        # Expected to fail because the kernel passes float* to A.data_ptr<float>()
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

def test_non_divisible_by_tile_size():
    """
    Issue 3: Matrix dimensions not divisible by TILE_SIZE.
    Use a matrix size that is not a multiple of TILE_SIZE to test proper handling of boundary tiles.
    """
    my_module = build_kernel()
    N = 70  # 70 is not divisible by TILE_SIZE=32.
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute the reference lower triangular product.
    C_ref = torch.tril(torch.matmul(A, B))
    # The kernel might produce differences at the boundaries.
    assert torch.allclose(C, C_ref, atol=1e-3), "Kernel output incorrect for matrix sizes not divisible by TILE_SIZE."

def test_batched_input_error():
    """
    Issue 4: Kernel is designed for 2D tensors only.
    This test sends a batched input (3D tensor) and expects the kernel to raise an error.
    """
    my_module = build_kernel()
    N = 64
    # Create batched triangular matrices.
    A = torch.randn(4, N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(4, N, N, device="cuda", dtype=torch.float32)
    # Force the triangular property on the last two dimensions.
    A = torch.tril(A)
    B = torch.tril(B)
    with pytest.raises(RuntimeError):
        # The check in the kernel expects 2D inputs.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
