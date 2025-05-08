
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper function to build/load the kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Weight tile fixed-size issue.
# Using a kernel whose width exceeds 8 (the allocated shared memory size for weights)
def test_weight_tile_issue():
    device = "cuda"
    batch_size = 2
    in_channels = 3
    out_channels = 4
    # Kernel height is small but kernel width is too large (>8)
    kernel_size = (3, 9)
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    # Create input and weight tensors.
    x = torch.randn(batch_size, in_channels, 32, 32, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], 
                         device=device, dtype=torch.float32)
    # Use no bias.
    bias = None

    my_module = build_kernel()
    out_cuda = my_module.forward(x, weight, bias, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    # Because the kernel shared memory for weights is too small, the result should be significantly different.
    # We require that the maximum absolute difference is above a small threshold.
    diff = (out_cuda - out_ref).abs().max().item()
    assert diff > 1e-3, f"Expected a difference due to weight tile size limitation, but got diff={diff}"

# Test 2: Input tile shared memory size issue.
# Using a stride > 1 forcing the kernel to load a larger input tile than allocated.
def test_input_tile_issue():
    device = "cuda"
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3)
    stride = 2  # Stride 2 forces a larger tile: TILE_HEIGHT*2 + kernel_height - 1
    padding = 0
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, 40, 40, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], 
                         device=device, dtype=torch.float32)
    bias = None

    my_module = build_kernel()
    out_cuda = my_module.forward(x, weight, bias, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    diff = (out_cuda - out_ref).abs().max().item()
    assert diff > 1e-3, f"Expected a difference due to insufficient input tile size, but got diff={diff}"

# Test 3: Non-contiguous input to trigger the CHECK_CONTIGUOUS macro.
def test_non_contiguous_input():
    device = "cuda"
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create a contiguous input and then make it non-contiguous via a transpose.
    x = torch.randn(batch_size, in_channels, 32, 32, device=device, dtype=torch.float32).transpose(1, 2)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], 
                         device=device, dtype=torch.float32)
    bias = None

    my_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        my_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Test 4: Non–CUDA tensor input to trigger the CHECK_CUDA macro.
def test_non_cuda_input():
    device = "cpu"  # intentionally use CPU
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, 32, 32, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], 
                         device=device, dtype=torch.float32)
    bias = None

    my_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        my_module.forward(x, weight, bias, stride, padding, dilation, groups)
