
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn as nn

# Helper function to compile the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case for Issue 1: misuse of pragma unroll in loops with runtime-dependent trip counts.
def test_unroll_issue():
    # Use a kernel size that does not match the fixed unroll factor assumptions.
    # For example, use an asymmetric kernel with dimensions different from 4.
    batch_size = 4
    in_channels = 3
    out_channels = 6
    # Choose kernel size deliberately not matching 4 (e.g. 5 and 7)
    kernel_size = (5, 7)
    stride = 2
    padding = 1
    output_padding = 1
    groups = 1
    use_bias = True

    # Create input tensor and reference PyTorch module
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    convT = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, bias=use_bias).cuda()
    ref_output = convT(x)

    # Build the custom CUDA kernel module
    ext = build_kernel()

    # Get weight and bias from our conv module
    weight = convT.weight  # shape: [in_channels, out_channels/groups, kh, kw]
    bias_tensor = convT.bias if use_bias else None

    # Forward pass using our kernel
    out = ext.forward(
        x.contiguous(), weight.contiguous(),
        bias_tensor if use_bias else None,
        stride, padding, output_padding, groups, 1)
    torch.cuda.synchronize()

    # Because of the unroll issue, the result is expected to differ from the correct output.
    assert not torch.allclose(out, ref_output, atol=1e-3), \
        "The test for unroll issue did not trigger the expected difference in output!"

# Test case for Issue 2: kernel assumption that tensors are contiguous.
def test_noncontiguous_input_issue():
    batch_size = 4
    in_channels = 3
    out_channels = 6
    kernel_size = (3, 3)
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    use_bias = True

    # Create a contiguous input tensor then make it noncontiguous.
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    # Make x noncontiguous by a permutation (and then a transpose of two dims)
    x_noncontig = x.permute(0, 2, 3, 1).transpose(1, 3)
    # Note: x_noncontig is now logically the same but not in the standard contiguous layout.

    convT = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, bias=use_bias).cuda()
    ref_output = convT(x_noncontig)

    ext = build_kernel()
    weight = convT.weight
    bias_tensor = convT.bias if use_bias else None

    # Forward pass using our kernel with noncontiguous input.
    out = ext.forward(
        x_noncontig, weight, bias_tensor if use_bias else None,
        stride, padding, output_padding, groups, 1)
    torch.cuda.synchronize()

    # Because the kernel assumes contiguous memory, the incorrect indexing should produce a result that differs.
    assert not torch.allclose(out, ref_output, atol=1e-3), \
        "The test for noncontiguous input issue did not trigger the expected difference in output!"
