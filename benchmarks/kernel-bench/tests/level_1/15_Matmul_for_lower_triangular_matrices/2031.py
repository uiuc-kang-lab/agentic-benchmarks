
import torch
import pytest
from torch.utils.cpp_extension import load
import time

# Build/load the CUDA extension from kernel.cu
def build_kernel():
    module = load(
        name="triangular_mm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Utility reference function: lower triangular matrix multiplication using PyTorch
def triangular_mm_ref(A, B):
    # Product of two lower triangular matrices is lower triangular.
    return torch.tril(torch.matmul(A, B))

# Issue 1: Early exit causing possible deadlock.
# We design a test where the matrix size is not a multiple of the tile size.
# This will force some threads to exit early and potentially cause a deadlock.
@pytest.mark.timeout(10)
def test_deadlock_for_partial_blocks():
    my_module = build_kernel()
    # Choose N such that N is not a multiple of 32 (default block size).
    N = 70  
    # Create lower triangular matrices
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    # Launch the kernel with block_size 32.
    start = time.time()
    C = my_module.forward(A, B, 32)
    # Force synchronization; if a deadlock occurs, this will timeout.
    torch.cuda.synchronize()
    elapsed = time.time() - start
    # Additionally, compare with the CPU reference result.
    C_ref = triangular_mm_ref(A, B)
    assert torch.allclose(C, C_ref, atol=1e-4), "Kernel result does not match reference for deadlock test."

# Issue 2: Data type support.
# Using non float32 (e.g. float64) should fail or produce wrong results.
def test_type_handling():
    my_module = build_kernel()
    N = 128
    # Create lower triangular double matrices instead of float32.
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float64))
    B = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float64))
    with pytest.raises(RuntimeError):
        # Expect failure because the kernel uses float pointers internally.
        _ = my_module.forward(A, B)

# Issue 3: Non-contiguous tensor inputs.
# Providing non-contiguous tensors can lead to incorrect results.
def test_non_contiguous_inputs():
    my_module = build_kernel()
    N = 256
    A_full = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B_full = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    # Create non-contiguous versions by slicing with a step.
    A_nc = A_full[:, ::1].transpose(0, 1)  # Transpose makes it non-contiguous; then ensure triangular structure.
    A_nc = torch.tril(A_nc)
    B_nc = B_full[:, ::1].transpose(0, 1)
    B_nc = torch.tril(B_nc)
    # They are non-contiguous now.
    C = my_module.forward(A_nc, B_nc, 32)
    torch.cuda.synchronize()
    # Reference computation should be done on contiguous copies.
    C_ref = triangular_mm_ref(A_nc.contiguous(), B_nc.contiguous())
    assert torch.allclose(C, C_ref, atol=1e-4), "Kernel result does not match reference for non-contiguous inputs."

# Issue 4: Unsupported block_size values.
# When an unsupported block_size (e.g., 20) is given, the kernel silently falls back to block_size=32.
def test_invalid_block_size_fallback():
    my_module = build_kernel()
    N = 128
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    # Provide a block size that is not 8, 16, or 32. Expect it to use block size 32 fallback.
    C_invalid = my_module.forward(A, B, 20)
    torch.cuda.synchronize()
    # Compute reference result with default settings.
    C_ref = triangular_mm_ref(A, B)
    assert torch.allclose(C_invalid, C_ref, atol=1e-4), "Kernel result does not match reference when using an unsupported block size."
