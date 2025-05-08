
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="bmm_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue 2 – using unsupported data type (double) should raise an error.
def test_unsupported_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cuda_module = build_kernel()
    # Create double precision inputs.
    batch_size, m, k, n = 2, 32, 64, 48
    A = torch.randn(batch_size, m, k, dtype=torch.double, device='cuda')
    B = torch.randn(batch_size, k, n, dtype=torch.double, device='cuda')
    with pytest.raises(RuntimeError):
        # Expect failure because the kernel expects float (float32)
        _ = cuda_module.forward(A, B)

# Test 2: Trigger issue 1 and issue 3 – using input dimensions and block configuration assumptions.
# Here, we choose matrix dimensions that are not an exact multiple of TILE (16) and expect the kernel to either give
# incorrect output or need extra care. In a correct implementation, torch.bmm would be computed correctly.
def test_non_multiple_tile_dimensions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cuda_module = build_kernel()
    # Use dimensions that are not multiples of TILE
    batch_size, m, k, n = 3, 30, 50, 45
    A = torch.randn(batch_size, m, k, dtype=torch.float32, device='cuda')
    B = torch.randn(batch_size, k, n, dtype=torch.float32, device='cuda')

    # Run our custom kernel
    C_kernel = cuda_module.forward(A, B)
    torch.cuda.synchronize()

    # Compute reference using torch.bmm
    C_ref = torch.bmm(A, B)

    # The kernel's tiling assumptions might lead to numerical differences if misaligned;
    # here we check for closeness but if the kernel is incorrect the test may fail.
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), (
        f"Kernel output does not match reference output!\nMax diff: {(C_kernel - C_ref).abs().max().item()}"
    )

# Test 3: Trigger issue 4 – with mismatched inner dimensions.
# This validates that the kernel performs a check on the inner dimension sizes.
def test_mismatched_inner_dimensions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cuda_module = build_kernel()
    batch_size, m, k1, k2, n = 2, 32, 64, 63, 48
    A = torch.randn(batch_size, m, k1, dtype=torch.float32, device='cuda')
    B = torch.randn(batch_size, k2, n, dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(A, B)
        
if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
