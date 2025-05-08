
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel does not implement a custom kernel, only a wrapper.
def test_custom_kernel_wrapper():
    # Create valid input tensors on CUDA.
    # Even though the kernel is only a wrapper, the expected behavior is to match torch.nn.functional.conv_transpose2d.
    x = torch.randn(2, 4, 10, 10, device="cuda")
    weight = torch.randn(4, 4, 3, 3, device="cuda")  # For groups=1, out_channels can equal in_channels.
    # Here we explicitly pass no bias.
    stride = [1, 1]
    padding = [0, 0]
    output_padding = [0, 0]
    dilation = [1, 1]
    groups = 1

    cuda_module = build_kernel()
    out_kernel = cuda_module.forward(x, weight, torch.Tensor(), stride, padding, output_padding, dilation, groups)
    
    # Get reference output using PyTorch's own conv_transpose2d.
    out_ref = torch.nn.functional.conv_transpose2d(x, weight, bias=None, stride=stride,
                                                   padding=padding, output_padding=output_padding,
                                                   groups=groups, dilation=dilation)
    # This test verifies that the wrapper produces the same result.
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), "Wrapped kernel output does not match reference output!"

# Issue 2: Bias handling can introduce device mismatches.
def test_bias_device_mismatch():
    # Create valid input and weight tensors on CUDA.
    x = torch.randn(2, 4, 10, 10, device="cuda")
    weight = torch.randn(4, 4, 3, 3, device="cuda")
    # Create a bias tensor on CPU instead of CUDA.
    bias_cpu = torch.randn(4)
    
    stride = [1, 1]
    padding = [0, 0]
    output_padding = [0, 0]
    dilation = [1, 1]
    groups = 1
    
    cuda_module = build_kernel()
    
    # We expect a device mismatch error, so we use pytest.raises.
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x, weight, bias_cpu, stride, padding, output_padding, dilation, groups)

# Issue 3: Kernel does not check that input tensors are on CUDA.
def test_input_not_cuda():
    # Create input tensor on CPU.
    x_cpu = torch.randn(2, 4, 10, 10, device="cpu")
    weight_cpu = torch.randn(4, 4, 3, 3, device="cpu")
    stride = [1, 1]
    padding = [0, 0]
    output_padding = [0, 0]
    dilation = [1, 1]
    groups = 1
    
    cuda_module = build_kernel()
    
    # Expect the kernel to raise an error because the inputs are not CUDA tensors.
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x_cpu, weight_cpu, torch.Tensor(), stride, padding, output_padding, dilation, groups)

# Issue 4: No validation on input tensor dimensions.
def test_invalid_dimensions():
    # Create an input tensor with invalid dimensions (e.g., 3D instead of 4D).
    x_invalid = torch.randn(4, 10, 10, device="cuda")  # Missing batch dimension.
    weight = torch.randn(4, 4, 3, 3, device="cuda")
    stride = [1, 1]
    padding = [0, 0]
    output_padding = [0, 0]
    dilation = [1, 1]
    groups = 1
    
    cuda_module = build_kernel()
    
    # The underlying at::conv_transpose2d should reject the wrong dimensions.
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x_invalid, weight, torch.Tensor(), stride, padding, output_padding, dilation, groups)
