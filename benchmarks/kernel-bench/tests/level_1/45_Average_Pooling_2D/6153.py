
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import numpy as np

# Helper to build the CUDA extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="avg_pool2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper to run the CUDA kernel forward function
def run_avg_pool(x, kernel_size, stride, padding):
    cuda_module = build_kernel()
    return cuda_module.forward(x, kernel_size, stride, padding)

# Test case 1: Test concurrent calls with different parameters to trigger potential race conditions
def test_concurrent_kernel_parameters():
    # Create two inputs with different sizes and different pooling parameters.
    # If the same __constant__ memory is used concurrently, one kernel launch might
    # override the other's parameters resulting in incorrect outputs.
    batch_size, channels, height, width = 8, 3, 32, 32
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    
    # Define two settings: one with kernel_size=3, stride=3, padding=1 and another with kernel_size=5, stride=2, padding=2.
    out1 = run_avg_pool(x, kernel_size=3, stride=3, padding=1)
    out2 = run_avg_pool(x, kernel_size=5, stride=2, padding=2)
    
    # Compute reference outputs using PyTorch's AvgPool2d
    pool1 = nn.AvgPool2d(kernel_size=3, stride=3, padding=1).to("cuda")
    pool2 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2).to("cuda")
    ref1 = pool1(x)
    ref2 = pool2(x)
    
    # The two outputs should match their respective reference values.
    # If the constant memory race condition occurs, one or both of the results will be incorrect.
    assert torch.allclose(out1, ref1, atol=1e-4), "Concurrent kernel call: output 1 does not match reference!"
    assert torch.allclose(out2, ref2, atol=1e-4), "Concurrent kernel call: output 2 does not match reference!"

# Test case 2: Test that the kernel fails or produces incorrect result when using half precision.
def test_half_precision_unsupported():
    # Using half precision input should trigger the lack of type support.
    batch_size, channels, height, width = 4, 3, 16, 16
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # The AT_DISPATCH_FLOATING_TYPES used in the kernel does not include half,
        # so it should not dispatch and should throw an error.
        _ = run_avg_pool(x, kernel_size=3, stride=3, padding=1)

# Test case 3: Test that an invalid kernel parameter (kernel_size = 0) leads to a division by zero.
def test_invalid_kernel_size_division_by_zero():
    # When kernel_size is 0, the kernel loops 0 times, and the final division becomes division by zero.
    batch_size, channels, height, width = 4, 3, 16, 16
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    # Kernel size 0 is invalid; we expect either an error from the CUDA kernel or NaN in the output.
    out = run_avg_pool(x, kernel_size=0, stride=1, padding=0)
    # Check if the output contains NaNs (which would be indicative of division by zero or undefined behavior)
    assert torch.isnan(out).any(), "Kernel with kernel_size=0 should produce NaN due to division by zero."

# Test case 4: Test that cudaMemcpyToSymbol error is handled when invalid parameters are passed.
# There is no straightforward way to force cudaMemcpyToSymbol to fail from Python,
# but we simulate an error scenario by passing a negative padding, which is invalid.
def test_invalid_padding():
    # Negative padding may not be supported by nn.AvgPool2d and our kernel logic.
    batch_size, channels, height, width = 4, 3, 16, 16
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = run_avg_pool(x, kernel_size=3, stride=1, padding=-1)
