
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports groups == 1.
def test_groups_not_supported():
    # Create a convolution where groups != 1.
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3, 3)
    stride = 1
    padding = 1
    dilation = 1
    groups = 2  # non supported groups
    device = "cuda"
    
    # Create dummy input and weight; bias is None.
    x = torch.randn(batch_size, in_channels, 10, 10, 10, device=device, dtype=torch.float32)
    # Weight shape: (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device=device, dtype=torch.float32)
    
    cuda_module = build_kernel()
    with pytest.raises(Exception):
        # This should raise an error because the kernel checks groups == 1.
        cuda_module.forward(x, weight, None, stride, padding, dilation, groups)

# Issue 2: Kernel assumes scalar stride, padding, and dilation.
def test_non_scalar_parameters():
    # Here we try to pass non-scalar (tuple) parameters.
    batch_size = 2
    in_channels = 3
    out_channels = 8
    kernel_size = (3, 5, 7)  # Asymmetric kernel
    stride = (1, 2, 3)       # Non-scalar stride
    padding = (0, 1, 2)      # Non-scalar padding
    dilation = (1, 1, 1)     # In this case dilation is uniform, but stride and padding are non-scalar.
    groups = 1
    device = "cuda"
    
    x = torch.randn(batch_size, in_channels, 32, 32, 32, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device=device, dtype=torch.float32)
    
    cuda_module = build_kernel()
    with pytest.raises(TypeError):
        # This call should fail because the kernel forward expects int for stride/padding/dilation.
        cuda_module.forward(x, weight, None, stride, padding, dilation, groups)

# Issue 3: Kernel only accepts float32 tensors.
def test_input_dtype():
    batch_size = 2
    in_channels = 3
    out_channels = 8
    kernel_size = (3, 3, 3)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    device = "cuda"
    
    # Create tensors in float64.
    x = torch.randn(batch_size, in_channels, 20, 20, 20, device=device, dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device=device, dtype=torch.float64)
    bias = torch.randn(out_channels, device=device, dtype=torch.float64)
    
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel is written for float pointers (float32). So using float64 should cause an error.
        cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Issue 4: Lack of error checking after kernel launch can hide kernel execution errors.
def test_incorrect_dimensions_trigger_kernel_error():
    # Create an input that makes computed output dimensions negative.
    # For instance, if the input volume is too small compared to the kernel size.
    batch_size = 1
    in_channels = 3
    out_channels = 4
    kernel_size = (5, 5, 5)
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    device = "cuda"
    
    # Input dimensions are smaller than kernel dimensions.
    x = torch.randn(batch_size, in_channels, 3, 3, 3, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device=device, dtype=torch.float32)
    
    cuda_module = build_kernel()
    # If the kernel launch leads to an error (e.g., out-of-bound accesses due to negative output dims)
    # then cudaDeviceSynchronize() should report it. We expect an exception.
    with pytest.raises(Exception):
        cuda_module.forward(x, weight, None, stride, padding, dilation, groups)
