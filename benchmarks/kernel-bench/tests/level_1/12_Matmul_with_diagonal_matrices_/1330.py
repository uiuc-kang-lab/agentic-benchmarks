
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension from kernel.cu
def build_kernel():
    try:
        cuda_module = load(
            name="kernel_module",
            sources=["kernel.cu"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=True,
        )
    except Exception as e:
        pytest.fail(f"Failed to compile kernel.cu: {e}")
    return cuda_module

# Reference implementation using PyTorch operations
def reference_forward(A, B):
    return torch.diag(A) @ B

# Test 1: Vectorized grid configuration: Use a size that triggers the vectorized branch.
def test_vectorized_grid_configuration():
    # M must be divisible by 4 and large enough to trigger vectorization (>=512)
    N = 32
    M = 512  # divisible by 4 and >=512
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, M, device="cuda", dtype=torch.float32)
    ref = reference_forward(A, B)
    module = build_kernel()
    out = module.forward(A, B)
    torch.cuda.synchronize()
    # This test is expected to fail because the block configuration for the vectorized branch is wrong.
    assert not torch.allclose(out, ref, atol=1e-5), "Test must detect the grid configuration issue in vectorized branch."

# Test 2: Multiple kernel launches interfering: Using any input will trigger the experimental loop.
def test_multiple_kernel_launches():
    # Choose sizes that force the experimental loop and non-vectorized path.
    N = 64
    M = 300  # Not triggering vectorized branch
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, M, device="cuda", dtype=torch.float32)
    ref = reference_forward(A, B)
    module = build_kernel()
    out = module.forward(A, B)
    torch.cuda.synchronize()
    # Due to multiple launches writing into the same C, the end result may not match the reference.
    assert not torch.allclose(out, ref, atol=1e-5), "Test must detect interference from multiple kernel launches."

# Test 3: Miscalculation of thread count: Use a case where M is very small relative to block_size
def test_thread_count_calculation():
    # M is small so that M < any typical block size in the array
    N = 16
    M = 10  # small number, non-vectorized branch will be used
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, M, device="cuda", dtype=torch.float32)
    ref = reference_forward(A, B)
    module = build_kernel()
    out = module.forward(A, B)
    torch.cuda.synchronize()
    # With an improper thread count calculation, the computed multiplication might skip some columns.
    assert not torch.allclose(out, ref, atol=1e-5), "Test must detect issues caused by the thread count miscalculation."

# Test 4: Lack of error checking/synchronization: Passing an intentionally bad input
def test_error_without_sync():
    # Passing a tensor on CPU (or wrong type) should trigger a failure.
    N = 32
    M = 64
    # Wrong device: keep B on CPU while A is on CUDA
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, M, device="cpu", dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(A, B)
    # This test should trigger an error due to missing proper type/device checks in kernel launch.
