
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension module
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    # Ensure the build happens only once per module
    return build_kernel()

# Test 1: groups != 1 is not supported.
def test_groups_not_supported(kernel_module):
    batch = 2
    in_channels = 4
    out_channels = 8
    height, width = 16, 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 2  # Not supported

    # Create contiguous float32 tensors
    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    
    # Bias is optional; provide one
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError, match="Only groups==1 is supported"):
        # Note: the kernel_module.forward signature is:
        # forward(x, weight, optional bias, stride, padding, dilation, groups)
        kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Test 2: Non float32 input tensors (e.g., using float64) should fail the CHECK_CUDA macros.
def test_incorrect_dtype(kernel_module):
    batch = 2
    in_channels = 3
    out_channels = 6
    height, width = 16, 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create double precision tensor on CUDA (which will fail since the kernel expects float32)
    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)
    
    with pytest.raises(RuntimeError):
        kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Test 3: Non-contiguous input tensor should trigger a contiguous check error.
def test_non_contiguous_input(kernel_module):
    batch = 2
    in_channels = 3
    out_channels = 6
    height, width = 16, 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    
    # Create a contiguous tensor and then make it non-contiguous by transposition
    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32).transpose(2, 3)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError, match="must be contiguous"):
        kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Test 4: The thread block configuration may become invalid if out_width is too large.
# By choosing an input width such that computed out_width exceeds the maximum allowed threads per block
# (typically 1024), a CUDA launch error should occur.
def test_large_out_width(kernel_module):
    batch = 1
    in_channels = 3
    out_channels = 3
    # Choose width such that out_width = (in_width + 2*padding - (kernel_size-1) - 1)/stride +1 > 1024.
    # For simplicity, set padding=0, stride=1, kernel_size=3: Then out_width = in_width - 2.
    # With in_width = 1030, out_width = 1028, which exceeds typical max threads (1024).
    in_width = 1030
    in_height = 16
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    x = torch.randn(batch, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Expect a CUDA launch configuration error due to too many threads per block.
    with pytest.raises(RuntimeError):
        kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)
