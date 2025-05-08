
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="transposed_conv3d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to get a ConvTranspose3d reference output from PyTorch
def torch_transposed_conv3d(x, weight, bias, stride, padding, output_padding, groups):
    # Create a ConvTranspose3d layer with the given parameters. The weight shape is
    # assumed to be [in_channels, out_channels_per_group, k_d, k_h, k_w]
    in_channels = x.size(1)
    out_channels = bias.size(0) if bias is not None else weight.size(0) * groups
    k_d, k_h, k_w = weight.shape[2], weight.shape[3], weight.shape[4]
    conv = torch.nn.ConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size=(k_d, k_h, k_w),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=(bias is not None)
    ).to(x.device, x.dtype)
    # Overwrite the weight and bias with the provided ones.
    # Note: For grouped convolutions, PyTorch expects weight shape:
    # (in_channels, out_channels_per_group, k_d, k_h, k_w)
    conv.weight.data.copy_(weight)
    if bias is not None:
        conv.bias.data.copy_(bias)
    return conv(x)

def random_tensor(shape, dtype=torch.float32, contiguous=True):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    if not contiguous:
        # Make a non-contiguous tensor by transposing dimensions temporarily
        x = x.transpose(1, 2)
    return x

# Issue 1: Output_padding is ignored by the kernel.
# This test creates a scenario where output_padding is non-zero.
def test_output_padding_ignored():
    cuda_mod = build_kernel()
    # Parameters
    batch = 2
    in_channels = 4
    groups = 1
    k_d, k_h, k_w = 3, 3, 3
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)  # non-zero output padding
    channels_per_group_out = 5  # arbitrary
    out_channels = channels_per_group_out * groups

    # Create input, weight, and bias tensors
    x = torch.randn(batch, in_channels, 8, 8, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, channels_per_group_out, k_d, k_h, k_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Using our custom kernel
    result_kernel = cuda_mod.forward(x, weight, bias, list(stride), list(padding), list(output_padding), groups)
    # Using PyTorch reference layer
    result_ref = torch_transposed_conv3d(x, weight, bias, stride, padding, output_padding, groups)
    # Because the kernel ignores output_padding, the results will differ.
    with pytest.raises(AssertionError):
        assert torch.allclose(result_kernel, result_ref, atol=1e-5), (
            "Test failed: Kernel should produce different results when output_padding is non-zero."
        )

# Issue 2: Kernel only supports float32.
def test_dtype_not_float32():
    cuda_mod = build_kernel()
    batch = 2
    in_channels = 4
    groups = 1
    k_d, k_h, k_w = 3, 3, 3
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    channels_per_group_out = 4

    # Create double precision tensors
    x = torch.randn(batch, in_channels, 8, 8, 8, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, channels_per_group_out, k_d, k_h, k_w, device="cuda", dtype=torch.float64)
    bias = torch.randn(channels_per_group_out * groups, device="cuda", dtype=torch.float64)

    # Expect the kernel to fail (or produce incorrect results) because of dtype mismatch.
    with pytest.raises(RuntimeError):
        # This call is expected to throw an error because the kernel only supports float (float32).
        cuda_mod.forward(x, weight, bias, list(stride), list(padding), list(output_padding), groups)

# Issue 3: Non-contiguous tensors are not handled.
def test_non_contiguous_tensor():
    cuda_mod = build_kernel()
    batch = 2
    in_channels = 4
    groups = 1
    k_d, k_h, k_w = 3, 3, 3
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    channels_per_group_out = 4

    # Create non-contiguous input tensor
    x = random_tensor((batch, in_channels, 8, 8, 8), dtype=torch.float32, contiguous=False)
    weight = torch.randn(in_channels, channels_per_group_out, k_d, k_h, k_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels_per_group_out * groups, device="cuda", dtype=torch.float32)

    # Expect the macro CHECK_INPUT to throw an error for non-contiguous tensor.
    with pytest.raises(RuntimeError):
        cuda_mod.forward(x, weight, bias, list(stride), list(padding), list(output_padding), groups)

# Issue 4: in_channels not divisible by groups.
def test_group_not_divisible():
    cuda_mod = build_kernel()
    batch = 2
    in_channels = 5  # Not divisible by groups=2
    groups = 2
    k_d, k_h, k_w = 3, 3, 3
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    channels_per_group_out = 4

    # Create input, weight, and bias tensors with mismatched group division.
    x = torch.randn(batch, in_channels, 8, 8, 8, device="cuda", dtype=torch.float32)
    # Weight shape is expected to be [in_channels, out_channels_per_group, k_d, k_h, k_w]
    weight = torch.randn(in_channels, channels_per_group_out, k_d, k_h, k_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels_per_group_out * groups, device="cuda", dtype=torch.float32)

    # Here, even though CHECK_INPUT passes, the kernel will compute an incorrect channels_per_group_in.
    # Thus, the output will not match the PyTorch reference.
    result_kernel = cuda_mod.forward(x, weight, bias, list(stride), list(padding), list(output_padding), groups)
    result_ref = torch_transposed_conv3d(x, weight, bias, stride, padding, output_padding, groups)

    with pytest.raises(AssertionError):
        assert torch.allclose(result_kernel, result_ref, atol=1e-5), (
            "Test failed: Kernel output should differ when in_channels is not divisible by groups."
        )
