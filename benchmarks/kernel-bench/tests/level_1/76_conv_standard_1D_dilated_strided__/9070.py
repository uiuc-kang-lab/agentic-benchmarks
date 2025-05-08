
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Helper to (re)build the CUDA extension module.
def build_kernel():
    cuda_module = load(
        name="optimized_conv1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# A helper function which mimics the expected convolution using PyTorch's Conv1d.
def pytorch_conv1d(x, weight, bias, stride, dilation):
    # Create a Conv1d with the same parameters as the input weight shape.
    # Note: weight is assumed to be in (out_channels, in_channels, kernel_size) format.
    in_channels = x.size(1)
    out_channels = weight.size(0)
    kernel_size = weight.size(2)
    conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=(bias is not None))
    # Overwrite the weight and bias with our tensors.
    conv1d.weight.data.copy_(weight)
    if bias is not None:
        conv1d.bias.data.copy_(bias)
    conv1d = conv1d.cuda()
    return conv1d(x)

# Issue 1: Kernel only supports float32.
def test_dtype_issue():
    # Build the kernel module.
    mod = build_kernel()
    # Create double precision input, weight and bias (if any).
    B = 4
    in_channels = 3
    out_channels = 8
    L = 32
    kernel_size = 3
    stride = 1
    dilation = 1

    x = torch.randn(B, in_channels, L, dtype=torch.float64, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.float64, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda")

    # Despite the fact that the input is double,
    # our kernel will use data_ptr<float>() and misinterpret the memory.
    # Thus the CUDA kernel result is expected to differ substantially from the PyTorch result.
    with pytest.raises(AssertionError):
        # We compare against the correct float64 convolution.
        expected = pytorch_conv1d(x, weight, bias, stride, dilation)
        output = mod.forward(x, weight, bias, stride, dilation)
        torch.cuda.synchronize()
        # The outputs will differ due to misinterpretation of the data.
        assert torch.allclose(output, expected, atol=1e-5)

# Issue 2: Kernel assumes contiguous tensors.
def test_non_contiguous_input():
    # Build the kernel module.
    mod = build_kernel()
    B = 4
    in_channels = 3
    out_channels = 8
    L = 64
    kernel_size = 3
    stride = 2
    dilation = 1

    # Create a contiguous tensor and then make a non-contiguous version via slicing.
    x = torch.randn(B, in_channels, L, device="cuda", dtype=torch.float32)
    x_noncontig = x[:, :, ::2]  # This slice is non-contiguous.
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # The forward function checks for contiguity.
    with pytest.raises(RuntimeError) as excinfo:
        mod.forward(x_noncontig, weight, bias, stride, dilation)
    assert "contiguous" in str(excinfo.value)

# Issue 3: Insufficient asynchronous error checking.
def test_async_error_detection():
    mod = build_kernel()
    B = 4
    in_channels = 3
    out_channels = 8
    L = 16   # Small length to more easily trigger issues.
    kernel_size = 3
    # Set stride to zero to force a division-by-zero (or invalid configuration) when computing output size.
    stride = 0  
    dilation = 1

    x = torch.randn(B, in_channels, L, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Expect the launch to fail (or the forward function to catch an issue) because stride=0 leads to undefined behavior.
    with pytest.raises(Exception):
        mod.forward(x, weight, bias, stride, dilation)
