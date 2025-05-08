
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="my_cuda_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to construct a transposed convolution scenario.
def run_transposed_conv(input, weight, bias, stride, padding, dilation):
    module = build_kernel()
    # The kernel function is exposed as "forward" in the extension.
    out = module.forward(input, weight, bias, stride, padding, dilation)
    torch.cuda.synchronize()
    return out

def get_default_input_params(batch_size=1, in_channels=1, height=10, width=10):
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    return x

# Issue 1: Kernel buffer overflow when kernel_size > MAX_KERNEL_SIZE.
def test_kernel_size_overflow():
    # We use a kernel_size larger than MAX_KERNEL_SIZE (16)
    # To force maximum number of valid indices, we use stride 1, padding such that
    # every kernel index is valid.
    batch_size = 1
    in_channels = 1
    out_channels = 1
    kernel_size = 17  # > MAX_KERNEL_SIZE, should trigger potential overflow.
    stride = 1
    padding = kernel_size // 2  # center padding so that each output position will use nearly all kernel elements.
    dilation = 1

    # Input and output dimensions for a simple case.
    height_in = 10
    width_in = 10

    x = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    # Weight shape for conv_transpose2d in PyTorch is (in_channels, out_channels, kernel_size, kernel_size)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Although a correct implementation would guard against kernel_size > MAX_KERNEL_SIZE,
    # our implementation does not. This test is expected to either crash or produce incorrect results.
    with pytest.raises(RuntimeError):
        # Depending on the system, the overflow might produce a CUDA error.
        run_transposed_conv(x, weight, bias, stride, padding, dilation)

# Issue 2: Incorrect bias shape.
def test_bias_shape_mismatch():
    # Use a valid kernel_size (<= MAX_KERNEL_SIZE) to avoid triggering issue 1.
    batch_size = 1
    in_channels = 1
    out_channels = 2  # set intended out_channels to 2
    kernel_size = 3  # within MAX_KERNEL_SIZE limit
    stride = 2
    padding = 1
    dilation = 1

    height_in = 8
    width_in = 8

    x = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    # Construct a bias with an incorrect shape, e.g. with only one channel instead of 2.
    bias = torch.randn(1, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError):
        run_transposed_conv(x, weight, bias, stride, padding, dilation)

# Issue 3: No error checking after cudaMemcpyToSymbol or kernel launch.
# While this is a host–side matter, we can force an error by passing an input on the wrong device.
def test_wrong_device_input():
    batch_size = 1
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    height_in = 8
    width_in = 8

    # Create an input tensor on the CPU instead of CUDA.
    x = torch.randn(batch_size, in_channels, height_in, width_in, device="cpu", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError):
        run_transposed_conv(x, weight, bias, stride, padding, dilation)
