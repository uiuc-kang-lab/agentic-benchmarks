
import torch
import pytest
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Utility to build the CUDA kernel extension
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger an error when a non-float32 tensor is passed in.
def test_non_float_input():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 8
    out_channels = 2
    kernel_size = 3
    length = 32
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    # Create double precision inputs (instead of float32)
    input_tensor = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float64)
    # Create a weight tensor with matching dtype (the kernel expects float32, though)
    weight_tensor = torch.randn(in_channels, out_channels // groups, kernel_size, device='cuda', dtype=torch.float64)
    bias_tensor = None

    with pytest.raises(RuntimeError):
        # The extension uses data_ptr<float>(), so the double tensor will lead to wrong interpretation.
        cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)

# Test case 2: Detect issues with reduction if the block reduction were modified.
# (The current kernel always launches with 32 threads so that it works, but the issue is latent if the block size is not exactly one warp.)
def test_reduction_incorrectness():
    cuda_module = build_kernel()
    
    # Configuration chosen so that the looping in compute_contribution will have many iterations.
    batch_size = 2
    in_channels = 64
    out_channels = 8   # 8 output channels; groups=1 so out_channels_per_group = 8.
    kernel_size = 7
    length = 50
    stride = 2
    padding = 1
    output_padding = 1
    groups = 1

    # Create input tensors with float32 type.
    input_tensor = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, out_channels // groups, kernel_size, device='cuda', dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # Use the custom CUDA kernel extension.
    out_custom = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)

    # Build an equivalent PyTorch convTranspose1d module
    conv_transpose = torch.nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=True
    ).to(device="cuda", dtype=torch.float32)

    # Manually set the parameters of conv_transpose to match our tensors.
    with torch.no_grad():
        conv_transpose.weight.copy_(weight_tensor)
        conv_transpose.bias.copy_(bias_tensor)

    out_torch = conv_transpose(input_tensor)
    
    # This test is meant to reveal if the reduction in the custom kernel is done incorrectly.
    # (If a future change causes the blockDim to be set to >32, out_custom will deviate from out_torch.)
    assert torch.allclose(out_custom, out_torch, atol=1e-5), (
        f"Custom kernel output does not match PyTorch ConvTranspose1d output!\n"
        f"Max difference: {(out_custom - out_torch).abs().max().item()}"
    )
