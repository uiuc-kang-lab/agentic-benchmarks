
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Issue 1: Data type is fixed to float32. Passing double tensors should cause an error or produce wrong results.
def test_dtype_incompatibility():
    my_module = build_kernel()
    batch_size, in_channels, length = 2, 3, 10
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    # Create double precision tensors on CUDA.
    x = torch.randn(batch_size, in_channels, length, dtype=torch.float64, device="cuda")
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float64, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda")

    with pytest.raises(RuntimeError):
        # The kernel expects float32 and will misuse the memory layout,
        # so we expect a runtime error (or, possibly, a crash that is caught as a Python RuntimeError)
        my_module.forward(x, weight, bias, stride, padding, dilation)

# Issue 2: Invalid convolution parameters can produce a negative output length.
def test_negative_output_length():
    my_module = build_kernel()
    # Choose parameters so that computed output length becomes negative.
    # Using input_length = 1, padding = 2, kernel_size = 3, dilation = 1, stride = 1:
    # output_length = (1-1)*1 - 2*2 + 1*(3-1) + 1 = 0 - 4 + 2 + 1 = -1.
    batch_size, in_channels, length = 2, 3, 1
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 2
    dilation = 1

    x = torch.randn(batch_size, in_channels, length, dtype=torch.float32, device="cuda")
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError):
        # The allocation of an output tensor with negative dimensions should cause an error.
        my_module.forward(x, weight, bias, stride, padding, dilation)

# Issue 3: A stride value of 0 leads to division-by-zero in the kernel.
def test_stride_zero_division_by_zero():
    my_module = build_kernel()
    batch_size, in_channels, length = 2, 3, 10
    out_channels = 4
    kernel_size = 3
    stride = 0  # Invalid: will lead to division-by-zero in the kernel.
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, length, dtype=torch.float32, device="cuda")
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError):
        # The kernel should trigger a runtime error due to division by zero.
        my_module.forward(x, weight, bias, stride, padding, dilation)
    # It is also a good practice to call torch.cuda.synchronize() to force error reporting.
    torch.cuda.synchronize()
