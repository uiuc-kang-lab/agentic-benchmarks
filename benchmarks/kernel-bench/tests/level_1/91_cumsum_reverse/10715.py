
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA kernel module from kernel.cu
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_grid_size_issue():
    """
    This test triggers the incorrect grid size calculation.
    We create an input tensor with many rows (each of size equal to WARP_SIZE = 32)
    so that the number of rows exceeds the number of blocks launched by the incorrect grid computation.
    The kernel is expected to produce an incorrect result because many rows will be unprocessed.
    """
    module = build_kernel()
    # Create a tensor with 100 rows and row_size = 32 (which is equal to the warp size).
    # (rows = 100 > 32, but our grid launch incorrectly computes grid = (100 + 31) // 32 = 4 blocks)
    input_tensor = torch.randn(100, 32, device="cuda", dtype=torch.float32)
    # Use dim=1 to force the optimized kernel to run.
    output = module.forward(input_tensor, 1)
    # Compute the expected result using the reference implementation (flip, cumsum, flip)
    expected = input_tensor.flip(1).cumsum(1).flip(1)
    # Since many rows are unprocessed by the kernel, the output should NOT match the expected result.
    assert not torch.allclose(output, expected, atol=1e-5), (
        "Test failed: Kernel output unexpectedly matches the reference result. "
        "This indicates that the grid launch issue might have been fixed."
    )

def test_negative_dim_fallback():
    """
    This test checks that when a negative dimension is provided, the fallback is activated.
    The optimized kernel is only used when dim equals (x.dim()-1). Since negative dims for the last axis
    are allowed in PyTorch but do not equal (x.dim()-1), the fallback will run and produce a correct result.
    """
    module = build_kernel()
    # Create a tensor with two dimensions where the last dimension has size 32.
    input_tensor = torch.randn(50, 32, device="cuda", dtype=torch.float32)
    # Provide a negative dimension (-1) so that although -1 refers to the last dim, the kernel
    # condition (dim == x.dim()-1) fails (since -1 != 1), triggering the fallback.
    output = module.forward(input_tensor, -1)
    expected = input_tensor.flip(-1).cumsum(-1).flip(-1)
    assert torch.allclose(output, expected, atol=1e-5), (
        "Test failed: Fallback implementation for negative dimension did not produce the expected result."
    )
