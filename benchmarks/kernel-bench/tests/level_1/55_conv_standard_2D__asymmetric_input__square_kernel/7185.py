
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test to trigger the unsupported groups issue.
def test_groups_not_supported():
    my_kernel = build_kernel()
    # Create a dummy input and weight that could be used in a grouped convolution.
    batch = 1
    in_channels = 4
    out_channels = 4
    height, width = 10, 10
    kernel_size = 3

    # Using groups=2 (should fail because only groups==1 are supported)
    groups = 2
    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Weight shape for a grouped convolution would normally be:
    # (out_channels, in_channels/groups, kernel_size, kernel_size)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Only groups==1 is supported"):
        # This should trigger the TORCH_CHECK in the kernel function.
        my_kernel.forward(x, weight, bias, stride=1, padding=1, dilation=1, groups=groups)

# Test to trigger the assumption on float32 type.
def test_dtype_not_float32():
    my_kernel = build_kernel()
    # Prepare input, weight and bias in double precision (float64)
    batch = 1
    in_channels = 3
    out_channels = 64
    height, width = 10, 10
    kernel_size = 3

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    # Run the custom kernel.
    output_custom = my_kernel.forward(x, weight, bias, stride=1, padding=1, dilation=1, groups=1)
    # Run the standard PyTorch convolution for reference using a model that accepts double dtype.
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1, bias=True).cuda().double()
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(bias)
    output_ref = conv(x)

    # The custom kernel misinterprets double as float32.
    # Thus, we expect the outputs to differ.
    assert not torch.allclose(output_custom.float(), output_ref.float(), atol=1e-5), (
        "The custom kernel incorrectly handled double inputs."
    )
