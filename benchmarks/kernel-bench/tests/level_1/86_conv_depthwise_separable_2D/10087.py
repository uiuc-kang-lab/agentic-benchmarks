
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    # Assume kernel.cu is in the same directory as this test file.
    module = load(
        name="custom_cuda_conv",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Helper: call the custom forward function from the extension.
def custom_forward(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation):
    mod = build_kernel()
    return mod.forward(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)

# Test case 1: Trigger grid dimension overflow.
# We trigger this by artificially constructing a tensor with a very large batch and number of channels,
# such that current_chunk * channels (or out_channels) becomes larger than CUDA's max gridDim.z.
# Most CUDA devices support gridDim.z up to 65535. We intentionally set the parameters so that the grid's z dimension exceeds that.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_grid_dimension_overflow():
    # set up dimensions so that (sub-batch * channels) exceeds 65535.
    # Here, pick a sub-batch that equals the full batch (num_streams=1) and set in_channels high.
    # For example, if batch=1024 and in_channels=128, then grid depth = 1024 * 128 = 131072.
    batch = 1024
    in_channels = 128
    out_channels = 256  # arbitrary
    kernel_size = 3
    height = 64
    width = 64
    stride = 1
    padding = 1
    dilation = 1

    # Create input tensor and weights on CUDA.
    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Depthwise convolution weight shape: [in_channels, 1, k, k]
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    # Pointwise convolution weight shape: [out_channels, in_channels, 1, 1] but our kernel expects [out_channels, in_channels]
    # (flattened 1x1 kernel).
    pointwise_weight = torch.randn(out_channels, in_channels, device="cuda", dtype=torch.float32)
    # Biases: provide empty tensor if bias is not used.
    depthwise_bias = torch.tensor([], device="cuda", dtype=torch.float32)
    pointwise_bias = torch.tensor([], device="cuda", dtype=torch.float32)

    # The expected behavior is that the kernel launch will fail (or produce undefined results).
    # We run the custom forward and catch a RuntimeError, which PyTorch throws when a CUDA kernel launch fails.
    with pytest.raises(RuntimeError):
        out = custom_forward(x, depthwise_weight, pointwise_weight, depthwise_bias,
                             pointwise_bias, stride, padding, dilation)
        # Synchronize to trigger any asynchronous error.
        torch.cuda.synchronize()

# Test case 2: Passing a non-CUDA tensor should trigger a TORCH_CHECK failure.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_cuda_input():
    batch = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    height = 32
    width = 32
    stride = 1
    padding = 0
    dilation = 1

    # Create CPU tensors
    x = torch.randn(batch, in_channels, height, width, device="cpu", dtype=torch.float32)
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cpu", dtype=torch.float32)
    pointwise_weight = torch.randn(out_channels, in_channels, device="cpu", dtype=torch.float32)
    depthwise_bias = torch.tensor([], device="cpu", dtype=torch.float32)
    pointwise_bias = torch.tensor([], device="cpu", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        out = custom_forward(x, depthwise_weight, pointwise_weight, depthwise_bias,
                             pointwise_bias, stride, padding, dilation)
