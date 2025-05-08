
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu (assumes kernel.cu is in the same directory as this file)
def build_kernel():
    module = load(
        name="custom_conv_transpose3d",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# A wrapper function for the kernel forward call
def conv_transpose3d_forward(input, weight, bias, stride, padding, output_padding, groups):
    mod = build_kernel()
    if bias is not None:
        bias_arg = bias
    else:
        bias_arg = torch.tensor([], device=input.device, dtype=input.dtype)
    return mod.forward(input, weight, bias_arg, 
                       [int(s) for s in stride],
                       [int(p) for p in padding],
                       [int(op) for op in output_padding],
                       int(groups))

# Test case 1: Non-float tensor types (e.g., double).
def test_non_float_tensor():
    # Create double precision input and weight (which is not supported by the kernel)
    N, C_in, D, H, W = 1, 4, 8, 8, 8
    out_channels_per_group = 2
    groups = 1
    # Weight layout: [C_in, out_channels_per_group, kD, kH, kW]
    kD, kH, kW = 3, 3, 3
    x = torch.randn(N, C_in, D, H, W, device="cuda", dtype=torch.double)
    weight = torch.randn(C_in, out_channels_per_group, kD, kH, kW, device="cuda", dtype=torch.double)
    bias = torch.randn(C_in * out_channels_per_group, device="cuda", dtype=torch.double)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel only supports float32.
        _ = conv_transpose3d_forward(x, weight, bias, stride, padding, output_padding, groups)

# Test case 2: Zero stride value causing division-by-zero.
def test_zero_stride():
    # Create a simple valid input (float) but with a zero in stride which is illegal.
    N, C_in, D, H, W = 1, 4, 8, 8, 8
    out_channels_per_group = 2
    groups = 1
    kD, kH, kW = 3, 3, 3
    x = torch.randn(N, C_in, D, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, out_channels_per_group, kD, kH, kW, device="cuda", dtype=torch.float32)
    bias = torch.randn(C_in * out_channels_per_group, device="cuda", dtype=torch.float32)
    # Set stride_d to 0 to trigger division-by-zero in the kernel.
    stride = (0, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    with pytest.raises(RuntimeError):
        _ = conv_transpose3d_forward(x, weight, bias, stride, padding, output_padding, groups)

# Test case 3: Groups parameter that does not evenly divide channel dimensions.
def test_invalid_groups():
    # Here, C_in is not divisible by groups.
    N, C_in, D, H, W = 1, 5, 8, 8, 8  # 5 channels, but we'll set groups=2.
    out_channels_per_group = 2
    groups = 2  # 5 is not divisible by 2.
    kD, kH, kW = 3, 3, 3
    x = torch.randn(N, C_in, D, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, out_channels_per_group, kD, kH, kW, device="cuda", dtype=torch.float32)
    bias = torch.randn((C_in // groups) * out_channels_per_group * groups, device="cuda", dtype=torch.float32)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    with pytest.raises(RuntimeError):
        # The kernel indexing for groups will be incorrect if channels cannot be evenly divided.
        _ = conv_transpose3d_forward(x, weight, bias, stride, padding, output_padding, groups)
