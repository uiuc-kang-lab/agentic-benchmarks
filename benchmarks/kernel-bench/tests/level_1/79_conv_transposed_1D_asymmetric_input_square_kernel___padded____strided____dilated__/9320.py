
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that a tensor of a different type (e.g., double) triggers an error.
def test_dtype_validation():
    cuda_module = build_kernel()
    batch_size, in_channels, out_channels, kernel_size, length = 2, 3, 4, 3, 8
    stride, padding, dilation = 1, 0, 1

    # Create input and weight tensors in double precision.
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.double)
    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.double)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.double)
    
    with pytest.raises(RuntimeError):
        # The wrapper expects float32; passing double should lead to unexpected behavior 
        # (or a TORCH_CHECK failure if added later).
        cuda_module.forward(x, weight, bias, stride, padding, dilation)

# Issue 2: Test that when an output_padding is needed, the kernel (which does not support it)
# produces an output with an unexpected shape compared to PyTorch's built-in function.
def test_output_padding_support():
    cuda_module = build_kernel()
    batch_size, in_channels, out_channels, kernel_size, length = 2, 3, 4, 3, 10
    # Use stride > 1 so that output_padding is normally used.
    stride, padding, dilation = 2, 1, 1
    output_padding = 1  # Intended extra padding (not supported by our CUDA kernel)

    # Create tensors in float32.
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    # Run our custom CUDA kernel.
    custom_out = cuda_module.forward(x, weight, bias, stride, padding, dilation)
    
    # Compute reference output using PyTorch's built-in functional conv_transpose1d with output_padding
    # Note: We have to manually specify the weight permutation if needed.
    ref_out = torch.nn.functional.conv_transpose1d(
        x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding
    )
    
    # The shapes should differ because our CUDA kernel doesn't support output_padding.
    assert custom_out.shape != ref_out.shape, \
        f"Expected different output shape due to missing output_padding support, but got same shape: {custom_out.shape}"

# Issue 3: Test that passing a tensor on the wrong device (i.e. CPU instead of CUDA),
# which should trigger a TORCH_CHECK failure, ideally catching launch errors.
def test_kernel_launch_error():
    cuda_module = build_kernel()
    batch_size, in_channels, out_channels, kernel_size, length = 2, 3, 4, 3, 8
    stride, padding, dilation = 1, 0, 1

    # Create input tensor on CPU (wrong device)
    x = torch.randn(batch_size, in_channels, length, dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float32)
    bias = torch.randn(out_channels, dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        # This should fail because the kernel expects CUDA tensors.
        cuda_module.forward(x, weight, bias, stride, padding, dilation)
