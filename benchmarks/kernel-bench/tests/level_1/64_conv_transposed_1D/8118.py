
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="conv_transposed1d_atomic",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper function to call the kernel's forward with given parameters.
# The forward function signature is:
# forward(input, weight, bias(optional), stride, padding, output_padding, groups)
def conv_transposed1d_forward(module, input, weight, bias, stride, padding, output_padding, groups):
    return module.forward(input, weight, bias, stride, padding, output_padding, groups)

# Issue 1: dtype enforcement.
def test_invalid_dtype():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    length = 10
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    # Create tensors with dtype double instead of float32.
    input_tensor = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.double)
    # Weight is expected to be of shape [in_channels, out_channels/groups, kernel_size]
    weight_tensor = torch.randn(in_channels, out_channels // groups, kernel_size, device='cuda', dtype=torch.double)
    bias_tensor = torch.randn(out_channels, device='cuda', dtype=torch.double)

    with pytest.raises(RuntimeError):
        # Calling forward with wrong dtype should trigger an error.
        _ = conv_transposed1d_forward(cuda_module, input_tensor, weight_tensor, bias_tensor,
                                      stride, padding, output_padding, groups)

# Issue 2: Bias shape verification.
def test_incorrect_bias_shape():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 6  # expected bias shape: (6,)
    kernel_size = 3
    length = 10
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    input_tensor = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, out_channels // groups, kernel_size, device='cuda', dtype=torch.float32)
    # Create a bias tensor with incorrect shape, for example, too few elements.
    bias_tensor = torch.randn(out_channels - 1, device='cuda', dtype=torch.float32)

    with pytest.raises(RuntimeError):
        _ = conv_transposed1d_forward(cuda_module, input_tensor, weight_tensor, bias_tensor,
                                      stride, padding, output_padding, groups)

# Issue 3: Groups divisibility.
def test_groups_divisibility():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 5    # not divisible by groups=2
    out_channels = 4   # out_channels should be in_channels_per_group * groups, so this is misconfigured.
    kernel_size = 3
    length = 10
    stride = 1
    padding = 0
    output_padding = 0
    groups = 2

    input_tensor = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    # Weight shape is supposed to be [in_channels, out_channels/groups, kernel_size].
    # With in_channels=5 and groups=2, in_channels_per_group would be 2.5 which is invalid.
    # We simulate an incorrect configuration by setting weight shape that doesn’t align.
    weight_tensor = torch.randn(in_channels, out_channels // groups, kernel_size, device='cuda', dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    with pytest.raises(RuntimeError):
        _ = conv_transposed1d_forward(cuda_module, input_tensor, weight_tensor, bias_tensor,
                                      stride, padding, output_padding, groups)
