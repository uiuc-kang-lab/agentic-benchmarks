
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension module from kernel.cu
    cuda_module = load(
        name="test_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def conv2d_reference(x, weight, bias, stride, padding, dilation, groups):
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

# Each test below is designed to trigger a different issue in the CUDA kernel implementation.
# Note: The tests assume that the CUDA kernel is used via the 'forward' function exposed
#       in the extension module (from kernel.cu) and that it will produce incorrect results.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_size_mismatch():
    # Issue 1: hard-coded kernel size of 3 versus user provided kernel_size 5.
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 8
    kernel_size = 5  # user intended kernel size
    height, width = 32, 32
    stride = 1
    padding = 2  # padding to maintain dimensions

    # Create random input and weight (with kernel_size=5)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Using the extension kernel: note that the kernel always uses KERNEL_SIZE=3.
    output_cuda = cuda_module.forward(x, weight, bias, stride, padding, 1, 1)
    
    # Compute the correct output using PyTorch reference
    output_ref = conv2d_reference(x, weight, bias, stride, padding, dilation=1, groups=1)
    
    # The outputs must differ because the launched kernel is incorrect.
    assert not torch.allclose(output_cuda, output_ref, atol=1e-4), (
        "Test failed: CUDA kernel output matches reference despite kernel size mismatch."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dilation_ignored():
    # Issue 2: dilation parameter is ignored.
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 8
    kernel_size = 3  # same as kernel macro but dilation != 1
    height, width = 32, 32
    stride = 1
    padding = 2  # chosen for dilation=2 to maintain output size approximately
    dilation = 2

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    output_cuda = cuda_module.forward(x, weight, bias, stride, padding, dilation, 1)
    output_ref = conv2d_reference(x, weight, bias, stride, padding, dilation, groups=1)
    
    assert not torch.allclose(output_cuda, output_ref, atol=1e-4), (
        "Test failed: CUDA kernel output matches reference even though dilation is ignored."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_groups_not_supported():
    # Issue 3: groups parameter is not handled.
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    height, width = 32, 32
    stride = 1
    padding = 1
    groups = 2  # So that each group has 2 channels
    
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Weight shape for grouped convolution: out_channels, in_channels/groups, k, k
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    output_cuda = cuda_module.forward(x, weight, bias, stride, padding, 1, groups)
    output_ref = conv2d_reference(x, weight, bias, stride, padding, dilation=1, groups=groups)
    
    assert not torch.allclose(output_cuda, output_ref, atol=1e-4), (
        "Test failed: CUDA kernel output matches reference even though groups are not supported."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_stride_not_applied():
    # Issue 4: stride parameter is not applied in reading input.
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    height, width = 32, 32
    stride = 2  # using a stride different from 1
    padding = 1
    
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    output_cuda = cuda_module.forward(x, weight, bias, stride, padding, 1, 1)
    output_ref = conv2d_reference(x, weight, bias, stride, padding, dilation=1, groups=1)
    
    assert not torch.allclose(output_cuda, output_ref, atol=1e-4), (
        "Test failed: CUDA kernel output matches reference even though stride is not applied."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_thread_block_tile_mismatch():
    # Issue 5: The block configuration is inconsistent with the tiling logic.
    # This test uses an input size that forces full usage of all threads in the block.
    # Due to the extra threads, the CUDA kernel may write incorrect output for some positions.
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    # Choose an input size so that output size is near the block dimensions.
    height, width = 40, 40
    stride = 1
    padding = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    output_cuda = cuda_module.forward(x, weight, bias, stride, padding, 1, 1)
    output_ref = conv2d_reference(x, weight, bias, stride, padding, dilation=1, groups=1)
    
    assert not torch.allclose(output_cuda, output_ref, atol=1e-4), (
        "Test failed: CUDA kernel output matches reference even though thread block tiling is misconfigured."
    )
