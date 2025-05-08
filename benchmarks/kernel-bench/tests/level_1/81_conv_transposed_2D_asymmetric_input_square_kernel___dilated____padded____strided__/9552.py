
import torch
import pytest
from torch import nn
from torch.utils.cpp_extension import load

# Utility to build and load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue (overflow in fixed-size arrays) by using a kernel_size > MAX_KERNEL_SIZE.
def test_kernel_size_overflow():
    # Set kernel_size to 17 which is > MAX_KERNEL_SIZE (16)
    in_channels = 4
    out_channels = 4
    kernel_size = 17  # Greater than fixed limit 16
    stride = 2
    padding = 1
    dilation = 1
    batch_size = 1
    in_height = 10
    in_width = 10

    # Create input and weight using float32 data type
    input_tensor = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)

    # Manually create weight tensor of shape [in_channels, out_channels, kernel_size, kernel_size]
    weight_tensor = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)

    # Create a bias tensor
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Load the cuda module and run the kernel
    cuda_mod = build_kernel()
    # Invoke the forward function from our module
    output = cuda_mod.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation)

    # Compute the reference output using PyTorch's native nn.ConvTranspose2d
    conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=True)
    # Manually copy the weight and bias to the PyTorch module for a fair comparison.
    with torch.no_grad():
        conv_trans.weight.copy_(weight_tensor)
        conv_trans.bias.copy_(bias_tensor)
    ref_output = conv_trans(input_tensor)

    # The outputs may not match if the kernel overruns its temporary buffers.
    # We expect a large discrepancy.
    diff = (output - ref_output).abs().max().item()
    assert diff > 1e-3, f"Output unexpectedly matches reference output. Difference: {diff}"

# Test 2: Trigger issue by providing non-float32 (e.g., float64) tensors.
def test_dtype_not_float32():
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    batch_size = 1
    in_height = 10
    in_width = 10

    # Create double precision input tensor
    input_tensor = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float64)
    # Create weight tensor with float64
    weight_tensor = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float64)
    # Create bias tensor with float64
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    cuda_mod = build_kernel()

    with pytest.raises(RuntimeError):
        # The custom CUDA kernel is compiled only for float (float32). Using float64 should result in an error.
        _ = cuda_mod.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation)
