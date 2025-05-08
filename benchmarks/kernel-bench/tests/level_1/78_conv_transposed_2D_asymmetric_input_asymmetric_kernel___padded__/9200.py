
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the extension from the kernel.cu file.
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function: call the kernel extension forward function.
def conv_transpose2d_forward(module, input, weight, bias, stride, padding):
    return module.forward(input, weight, bias, stride, padding)

# Test 1: Triggering the issue with incorrect tensor data type
def test_dtype_issue():
    module = build_kernel()
    # Create double precision inputs; our CUDA kernel expects float32.
    batch_size = 2
    in_channels = 3
    out_channels = 4
    height = 8
    width = 8
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    
    # Tensors in double precision instead of float.
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device='cuda', dtype=torch.float64)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float64)
    
    with pytest.raises(RuntimeError):
        # We expect the kernel extension to malfunction (e.g. crash or raise an error)
        # due to the erroneous reinterpretation of double data as float.
        y = conv_transpose2d_forward(module, x, weight, bias, stride, padding)
        torch.cuda.synchronize()

# Test 2: Triggering the issue with non-contiguous tensor assumptions
def test_non_contiguous_issue():
    module = build_kernel()
    # Create contiguous tensors first.
    batch_size = 2
    in_channels = 3
    out_channels = 4
    height = 8
    width = 8
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    # Make input and weight non-contiguous by transposing dimensions.
    x_non_contig = x.transpose(2, 3)  # Now shape is (batch, channels, width, height) and non-contiguous.
    weight_non_contig = weight.transpose(2, 3)  # non-contiguous
    # The kernel expects contiguous memory in a specific layout.
    
    with pytest.raises(Exception):
        # We expect that the kernel will produce incorrect results or crash when given non-contiguous tensors.
        y = conv_transpose2d_forward(module, x_non_contig, weight_non_contig, bias, stride, padding)
        torch.cuda.synchronize()

# Test 3: Testing with a non-compile-time-constant kernel size
# (This test aims at revealing potential performance/optimization issues due to improper loop unrolling.)
def test_dynamic_kernel_size_issue():
    module = build_kernel()
    # Use a kernel size that is irregular (i.e., not typical small constant) to potentially defeat compile-time unrolling.
    # The kernel's computation is still expected to be correct if unrolling worked well.
    batch_size = 1
    in_channels = 2
    out_channels = 2
    height = 16
    width = 16
    kernel_size = (3, 7)  # The width = 7 may not be unrolled optimally.
    stride = (1, 1)
    padding = (1, 3)
    
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    # Compute output using both our kernel and PyTorch's conv_transpose2d for correctness.
    y_kernel = conv_transpose2d_forward(module, x, weight, bias, stride, padding)
    torch.cuda.synchronize()
    
    # Compute reference output with PyTorch's built-in function.
    ref_conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
    # Copy the same weight and bias.
    with torch.no_grad():
        ref_conv.weight.copy_(weight)
        ref_conv.bias.copy_(bias)
    y_ref = ref_conv(x)
    
    # Allow for a loose tolerance in case of slight numerical differences 
    # (the test aims to reveal potential issues rather than perfect performance)
    assert torch.allclose(y_kernel, y_ref, atol=1e-4), \
        f"Kernel output does not match CPU reference output. Max difference: {(y_kernel - y_ref).abs().max()}"

if __name__ == "__main__":
    pytest.main([__file__])
