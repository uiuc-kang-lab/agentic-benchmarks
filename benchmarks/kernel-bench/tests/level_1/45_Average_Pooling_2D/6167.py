
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility: compile and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="avg_pool2d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that feeding a non–float32 tensor (e.g. torch.double)
# to the kernel leads to incorrect results (or a failure). The kernel is
# not handling double inputs.
def test_input_type_mismatch():
    kernel_size = 3
    stride = 1
    padding = 1
    # Create an input tensor of type double.
    x = torch.randn(1, 1, 8, 8, device="cuda", dtype=torch.double)
    # Build the extension.
    ext = build_kernel()
    # Try calling the forward function. Either this should error out or produce
    # incorrect result because the kernel uses data_ptr<float>() unconditionally.
    with pytest.raises(RuntimeError):
        # Expecting a runtime error due to wrong memory interpretation.
        out = ext.forward(x, kernel_size, stride, padding)
        torch.cuda.synchronize()

# Issue 2: Test that an extreme padding setting leads to an output location where
# the pooling window has 0 valid input elements. Division by 0 should occur.
def test_division_by_zero():
    # Create an input tensor that is small relative to the padding.
    # For instance, an input of shape 1x1x5x5, kernel_size=3, stride=3, padding=3.
    # For some output positions, the kernel window will not overlap any valid input.
    x = torch.randn(1, 1, 5, 5, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 3
    padding = 3
    ext = build_kernel()
    out = ext.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    # Check if any element in out is NaN (resulting from a division by 0)
    assert torch.isnan(out).any(), "Expected NaN values due to division by zero, but none were found."

# Issue 3: Test that the kernel’s averaging method (dividing by the number of valid elements)
# does not match PyTorch's default behavior when count_include_pad is True.
def test_semantic_mismatch_with_pytorch():
    # Use a scenario where the pooling window is partially outside the valid region.
    # PyTorch's AvgPool2d (default settings with count_include_pad=True) always divides
    # by kernel_size * kernel_size, but our kernel divides only by the number of valid inputs.
    kernel_size = 3
    stride = 1
    padding = 1
    # Create an input tensor with one channel.
    torch.manual_seed(42)
    x = torch.randn(1, 1, 8, 8, device="cuda", dtype=torch.float32)
    ext = build_kernel()
    out_cuda = ext.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()

    # Use PyTorch's native avg_pool2d, which divides by (kernel_size*kernel_size)
    # even if part of the kernel window is padded.
    pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    out_torch = pool(x)

    # The results should differ because of the different averaging strategy.
    # We expect that at least one element is not approximately equal.
    differences = (out_cuda - out_torch).abs()
    assert differences.max() > 1e-5, f"Kernel output unexpectedly matched PyTorch's AvgPool2d!"

