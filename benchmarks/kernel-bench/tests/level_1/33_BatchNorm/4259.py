
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to build and load the CUDA kernel module.
def build_kernel():
    cuda_module = load(
        name="batch_norm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# A helper module that mimics the BatchNorm forward API.
class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var, training, momentum, eps, kernel):
        # Call the external CUDA kernel forward function.
        return kernel.forward(input, weight, bias, running_mean, running_var, training, momentum, eps)

# Test case 1: Passing non-float32 (e.g., float64) input to trigger dtype issue.
def test_non_float32_dtype():
    kernel = build_kernel()
    batch_size = 16
    channels = 64
    H, W = 32, 32
    # Create input and parameter tensors of type float64 instead of float32.
    x = torch.randn(batch_size, channels, H, W, dtype=torch.float64, device="cuda")
    weight = torch.randn(channels, dtype=torch.float64, device="cuda")
    bias = torch.randn(channels, dtype=torch.float64, device="cuda")
    running_mean = torch.zeros(channels, dtype=torch.float64, device="cuda")
    running_var = torch.ones(channels, dtype=torch.float64, device="cuda")
    momentum = 0.1
    eps = 1e-5
    with pytest.raises(RuntimeError):
        # This should fail because the kernel expects float (float32) tensors.
        out = BatchNormFunction.forward(None, x, weight, bias, running_mean, running_var, True, momentum, eps, kernel)
        torch.cuda.synchronize()

# Test case 2: Passing non-contiguous input tensor.
def test_non_contiguous_input():
    kernel = build_kernel()
    batch_size = 16
    channels = 64
    H, W = 32, 32
    # Create a contiguous tensor and then make it non-contiguous by a transpose operation.
    x = torch.randn(batch_size, channels, H, W, device="cuda", dtype=torch.float32).transpose(1, 2)
    # Make weight, bias, running_mean, and running_var always contiguous.
    weight = torch.randn(channels, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    running_mean = torch.zeros(channels, device="cuda", dtype=torch.float32)
    running_var = torch.ones(channels, device="cuda", dtype=torch.float32)
    momentum = 0.1
    eps = 1e-5
    with pytest.raises(RuntimeError):
        # The CHECK_CONTIGUOUS macros should trigger an error.
        out = BatchNormFunction.forward(None, x, weight, bias, running_mean, running_var, True, momentum, eps, kernel)
        torch.cuda.synchronize()

# Test case 3: Passing an input with wrong dimensions (not 4D).
def test_incorrect_input_dimensions():
    kernel = build_kernel()
    # Create a 3D tensor instead of a 4D tensor.
    batch_size = 16
    channels = 64
    H = 32
    x = torch.randn(batch_size, channels, H, device="cuda", dtype=torch.float32)
    weight = torch.randn(channels, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    running_mean = torch.zeros(channels, device="cuda", dtype=torch.float32)
    running_var = torch.ones(channels, device="cuda", dtype=torch.float32)
    momentum = 0.1
    eps = 1e-5
    with pytest.raises(IndexError):
        # The memory indexing will be invalid when the input is not 4D.
        out = BatchNormFunction.forward(None, x, weight, bias, running_mean, running_var, True, momentum, eps, kernel)
        torch.cuda.synchronize()

# Test case 4: Passing a very large number of channels to trigger grid dimension limit issues.
def test_excessively_large_channels():
    kernel = build_kernel()
    # Use a channel count that is unusually large. Note: This test might require a GPU that supports such a launch.
    batch_size = 1
    # Choose a channel number that exceeds typical gridDim.x limits (e.g., > 65535)
    channels = 70000  
    H, W = 8, 8
    x = torch.randn(batch_size, channels, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(channels, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    running_mean = torch.zeros(channels, device="cuda", dtype=torch.float32)
    running_var = torch.ones(channels, device="cuda", dtype=torch.float32)
    momentum = 0.1
    eps = 1e-5
    with pytest.raises(RuntimeError):
        # The grid launch might fail if the number of blocks (channels) exceeds the device limit.
        out = BatchNormFunction.forward(None, x, weight, bias, running_mean, running_var, True, momentum, eps, kernel)
        torch.cuda.synchronize()
