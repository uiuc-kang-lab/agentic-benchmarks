
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension module from kernel.cu.
    cuda_module = load(
        name="transposed_conv_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Wrong data type (non-float tensor)
def test_wrong_dtype():
    cuda_module = build_kernel()
    batch = 2
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    height = 8
    width = 8
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    # Create input and weight as double (float64) instead of float32.
    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    with pytest.raises(RuntimeError):
        # Expect the kernel to fail because it is written assuming float.
        cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Issue 2: Grid dimension (z) too large (batch * in_channels exceeds CUDA limit)
def test_large_grid_dimension():
    cuda_module = build_kernel()
    # Use extremely large batch and channel numbers so that batch*in_channels > 65535.
    # Note: These sizes are chosen to trigger a launch failure due to the CUDA grid dimension limits.
    batch = 70
    in_channels = 1000  # 70*1000 = 70000 > 65535
    out_channels = 1000
    kernel_size = (3, 3)
    height = 8
    width = 8
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    x = torch.randn(batch, in_channels, height, width, device="cuda")
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device="cuda")
    bias = torch.randn(out_channels, device="cuda")

    with pytest.raises(RuntimeError):
        # Expect a CUDA launch failure because the grid z-dimension exceeds hardware limits.
        cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()

# Issue 3: Lack of error-checking after initialize_output_kernel launch leading to out-of-bound access
# Here we simulate the error by providing a bias of incorrect shape.
def test_incorrect_bias_shape():
    cuda_module = build_kernel()
    batch = 2
    in_channels = 4
    out_channels = 6   # Expected out_channels is computed from weight: groups * weight.size(1)
    kernel_size = (3, 3)
    height = 8
    width = 8
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    x = torch.randn(batch, in_channels, height, width, device="cuda")
    # Here, weight shape dictates out_channels = weight.size(1) * groups.
    # We'll create a weight tensor that expects out_channels = 8 but we provide bias with wrong size.
    weight = torch.randn(in_channels, 8, kernel_size[0], kernel_size[1], device="cuda")
    # Provide a bias tensor with incorrect shape (should be 8, but we supply 6 elements).
    bias = torch.randn(out_channels, device="cuda")  

    with pytest.raises(RuntimeError):
        # The initialize_output_kernel should carry out an out-of-bound write/read causing an error (or later synchronization will catch it).
        cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()
