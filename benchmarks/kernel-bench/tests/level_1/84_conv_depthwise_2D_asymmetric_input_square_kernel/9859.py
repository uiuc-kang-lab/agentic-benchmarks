
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to build/load the CUDA kernel module.
def build_kernel():
    cuda_module = load(
        name="cuda_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test for float32-only assumption.
def test_dtype_not_float32():
    cuda_module = build_kernel()
    # Create double (float64) tensors.
    batch_size, in_channels, h, w = 4, 3, 32, 32
    input = torch.randn(batch_size, in_channels, h, w, dtype=torch.float64, device="cuda")
    # For depthwise conv, weight shape is [in_channels, 1, kernel_size, kernel_size].
    kernel_size = 3
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float64, device="cuda")
    bias = torch.randn(in_channels, dtype=torch.float64, device="cuda")
    with pytest.raises(Exception):
        cuda_module.forward(input, weight, bias, 1, 0)

# Issue 2: Test for non-contiguous tensor input.
def test_non_contiguous():
    cuda_module = build_kernel()
    batch_size, in_channels, h, w = 4, 3, 32, 32
    input = torch.randn(batch_size, in_channels, h, w, device="cuda", dtype=torch.float32)
    # Make input non-contiguous by transposing a couple of dimensions.
    input_noncontig = input.transpose(1, 2)
    kernel_size = 3
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception):
        cuda_module.forward(input_noncontig, weight, bias, 1, 0)

# Issue 3: Test with a kernel_size other than 3 to trigger fallback code.
def test_kernel_size_not_3():
    cuda_module = build_kernel()
    batch_size, in_channels, h, w = 2, 3, 32, 32
    # Choose kernel_size != 3; for example, 5.
    kernel_size = 5
    input = torch.randn(batch_size, in_channels, h, w, device="cuda", dtype=torch.float32)
    # For depthwise conv, weight shape is [in_channels, 1, kernel_size, kernel_size]
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    # Although the kernel has fallback branch, the hard-coded unrolled version won't trigger.
    # We assume the fallback branch may be less optimized or have hidden issues.
    output = cuda_module.forward(input, weight, bias, 1, 0)
    # Simple sanity check: output shape.
    output_h = (h + 0 - kernel_size) // 1 + 1
    output_w = (w + 0 - kernel_size) // 1 + 1
    assert output.shape == (batch_size, in_channels, output_h, output_w), \
        f"Unexpected output shape: {output.shape}"

# Issue 4: Test potential load imbalance by forcing a tile configuration that is not naturally 2D.
def test_block_dimension_inefficiency():
    cuda_module = build_kernel()
    # Create an input where the output dimensions are not multiples of TILE_WIDTH/TILE_HEIGHT.
    batch_size, in_channels = 1, 3
    h, w = 45, 47  # dimensions chosen to produce incomplete tiles
    kernel_size = 3
    input = torch.randn(batch_size, in_channels, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    output = cuda_module.forward(input, weight, bias, 1, 1)
    # Check output size consistency.
    output_h = (h + 2 * 1 - kernel_size) // 1 + 1
    output_w = (w + 2 * 1 - kernel_size) // 1 + 1
    assert output.shape == (batch_size, in_channels, output_h, output_w), \
        f"Unexpected output shape: {output.shape}"

# Issue 5: Test for excessive shared memory usage.
def test_excessive_shared_memory():
    cuda_module = build_kernel()
    # Choose a very large kernel_size to force high shared memory usage.
    batch_size, in_channels = 1, 3
    h, w = 100, 100
    kernel_size = 129  # This extremely high kernel size likely exceeds available shared memory.
    input = torch.randn(batch_size, in_channels, h, w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception):
        cuda_module.forward(input, weight, bias, 1, 0)
