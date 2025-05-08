
import torch
import pytest
from torch.nn import GroupNorm
from torch.utils.cpp_extension import load

# Helper function to dynamically compile and load the CUDA kernel
def build_kernel():
    cuda_module = load(
        name="group_norm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference implementation using PyTorch's GroupNorm module:
def ref_group_norm(x, num_groups, eps, weight, bias):
    gn = GroupNorm(num_groups=num_groups, num_channels=x.size(1), eps=eps)
    gn.weight.data = weight.clone().detach()
    gn.bias.data = bias.clone().detach()
    return gn(x)

# Test Case 1:
# This test passes double precision (torch.float64) input, which should trigger:
#   - Insufficient shared memory allocation due to using sizeof(float) in the kernel.
#   - Incorrect usage of rsqrt which is not defined for doubles.
def test_double_precision_issue():
    device = "cuda"
    # Use a moderately sized tensor.
    N, C, H, W = 4, 8, 32, 32
    num_groups = 2
    eps = 1e-5
    # Create double precision input and parameters.
    x = torch.randn(N, C, H, W, dtype=torch.float64, device=device)
    weight = torch.randn(C, dtype=torch.float64, device=device)
    bias = torch.randn(C, dtype=torch.float64, device=device)
    
    # Build the CUDA kernel module.
    cuda_mod = build_kernel()
    # Attempt to run the custom kernel.
    with pytest.raises(Exception):
        # We expect the kernel execution to fail or produce an error due to the issues.
        y = cuda_mod.forward(x, weight, bias, num_groups, eps)
        torch.cuda.synchronize()
    
# Test Case 2:
# This test uses input dimensions that are not multiples of the block size (16x16) to expose any potential issues
# with boundary condition handling in the reduction within compute_stats_kernel.
def test_irregular_spatial_dimensions():
    device = "cuda"
    # Intentionally choose H and W not divisible by 16.
    N, C, H, W = 2, 8, 37, 41
    num_groups = 2
    eps = 1e-5
    # Use standard float32 inputs (to avoid the double precision rsqrt issue).
    x = torch.randn(N, C, H, W, dtype=torch.float32, device=device)
    weight = torch.randn(C, dtype=torch.float32, device=device)
    bias = torch.randn(C, dtype=torch.float32, device=device)
    
    # Build the CUDA kernel module.
    cuda_mod = build_kernel()
    # Run the custom kernel forward.
    y_kernel = cuda_mod.forward(x, weight, bias, num_groups, eps)
    torch.cuda.synchronize()
    # Compute reference result.
    y_ref = ref_group_norm(x, num_groups, eps, weight, bias)
    # The potential issue might corrupt reduction results, so the outputs are not close.
    # We assert that the maximum absolute difference is significant.
    diff = (y_kernel - y_ref).abs().max().item()
    assert diff > 1e-3, f"Test for irregular dimensions did not trigger a noticeable difference, diff: {diff}"
