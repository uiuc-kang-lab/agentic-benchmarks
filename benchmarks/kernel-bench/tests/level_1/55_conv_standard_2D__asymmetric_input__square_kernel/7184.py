
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA kernel module from kernel.cu
def build_kernel():
    return load(
        name="conv2d_shared_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Kernel only supports float32, so passing double should trigger an error.
def test_dtype_issue():
    device = "cuda"
    batch = 1
    in_channels = 3
    out_channels = 8
    in_height = 32
    in_width = 32
    kernel_size = 3
    # Create tensors of type float64 (double)
    x = torch.randn(batch, in_channels, in_height, in_width, dtype=torch.float64, device=device)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float64, device=device)
    bias = torch.randn(out_channels, dtype=torch.float64, device=device)
    mod = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        mod.forward(x, weight, bias, 1, 0, 1, 1)
    assert "must be a CUDA tensor" not in str(excinfo.value), "Expected dtype error for double tensors"

# Issue 2: Kernel does not support non-contiguous tensors.
def test_non_contiguous_input():
    device = "cuda"
    batch = 1
    in_channels = 3
    out_channels = 8
    in_height = 32
    in_width = 32
    kernel_size = 3
    # Create a contiguous tensor and then make a non-contiguous view (e.g., transpose)
    x = torch.randn(batch, in_channels, in_height, in_width, device=device).transpose(1, 2)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device)
    bias = torch.randn(out_channels, device=device)
    mod = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        mod.forward(x, weight, bias, 1, 0, 1, 1)
    assert "must be contiguous" in str(excinfo.value)

# Issue 3: Kernel only supports groups == 1.
def test_group_convolution():
    device = "cuda"
    batch = 1
    in_channels = 4
    out_channels = 8
    in_height = 32
    in_width = 32
    kernel_size = 3
    x = torch.randn(batch, in_channels, in_height, in_width, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device, dtype=torch.float32)
    bias = torch.randn(out_channels, device=device, dtype=torch.float32)
    mod = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        # Pass groups != 1 (e.g., groups=2) to trigger the check.
        mod.forward(x, weight, bias, 1, 0, 1, 2)
    assert "Only groups==1 is supported" in str(excinfo.value)

# Issue 4: Fixed TILE_WIDTH and tiling assumptions.
# This test creates an input where the output dimensions are not multiples of TILE_WIDTH.
def test_non_multiple_tile_dimensions():
    device = "cuda"
    batch = 1
    in_channels = 3
    out_channels = 8
    # choose an input shape so that output dimensions are not multiples of TILE_WIDTH (16)
    in_height = 45  # arbitrary dimension that will lead to non-multiple output rows
    in_width = 45   # same for width
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    # Using float tensor and contiguous memory
    x = torch.randn(batch, in_channels, in_height, in_width, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device, dtype=torch.float32)
    bias = torch.randn(out_channels, device=device, dtype=torch.float32)
    mod = build_kernel()
    # The kernel should run without out-of-bound accesses.
    # We are not checking numerical accuracy here since the implementation may be suboptimal
    # for non-multiple tile dimensions, but we expect it to not crash.
    output = mod.forward(x, weight, bias, stride, padding, dilation, 1)
    assert output.shape[2] == (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    assert output.shape[3] == (in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
