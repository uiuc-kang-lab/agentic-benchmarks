
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to build and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Check groups != 1 trigger error
def test_groups_not_supported():
    my_module = build_kernel()

    # Create a 4D input tensor (batch, channels, h, w)
    batch_size, in_channels, height, width = 4, 4, 16, 16
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    
    # Create a weight tensor with shape (out_channels, in_channels/groups, k, k)
    out_channels, kernel_size = 8, 3
    # For groups != 1, the weight expected shape is incompatible with our kernel
    weight = torch.randn(out_channels, in_channels // 2, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    
    # Bias is optional; just use None.
    bias = None

    # groups != 1 should trigger the TORCH_CHECK and raise an exception.
    with pytest.raises(RuntimeError, match="groups != 1 not supported"):
        my_module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=2)

# Test 2: Check non-float tensor (e.g. double) leads to wrong behavior
def test_non_float_dtype():
    my_module = build_kernel()

    batch_size, in_channels, height, width = 2, 3, 32, 32
    kernel_size = 3
    out_channels = 5

    # Create double data type inputs, which our kernel does not support.
    x_double = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float64)
    weight_double = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float64)
    bias_double = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    # In our kernel, the tensors are treated as float32. Running the kernel with double data
    # will lead to wrong results (or a runtime error depending on memory layout).
    # We compare the custom output with the reference convolution from nn.Conv2d,
    # after casting the reference result to float32 to force a mismatch.
    # No exception may be raised at call time, but the numerical output will be very inaccurate.
    out_custom = my_module.forward(x_double, weight_double, bias_double, stride=1, padding=1, dilation=1, groups=1)
    
    # Create a reference convolution for comparison (forcing double precision)
    conv_ref = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True).to(device="cuda", dtype=torch.float64)
    # Copy our weight and bias into the reference conv to get an identical operation
    with torch.no_grad():
        conv_ref.weight.copy_(weight_double)
        conv_ref.bias.copy_(bias_double)
    out_ref = conv_ref(x_double)
    
    # Now, cast our custom output to double for fair comparison.
    out_custom_double = out_custom.to(torch.float64)
    
    # The outputs should differ significantly due to the wrong interpretation of the tensor data.
    # We expect that the maximum absolute difference is large.
    diff = (out_custom_double - out_ref).abs().max().item()
    assert diff > 1e-3, f"Expected significant difference when using double tensors; got diff={diff}"

# Test 3: Check input dimension validation issue.
def test_invalid_input_dimensions():
    my_module = build_kernel()
    
    # Create an input tensor that is not 4D (e.g., 3D tensor) to simulate a shape error.
    x_invalid = torch.randn(3, 32, 32, device="cuda", dtype=torch.float32)  # Missing batch/channel dims
    weight = torch.randn(8, 3, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(8, device="cuda", dtype=torch.float32)
    
    # Since our kernel extracts dimensions assuming a 4D tensor, this should trigger an error
    with pytest.raises(IndexError):
        my_module.forward(x_invalid, weight, bias, stride=1, padding=1, dilation=1, groups=1)
