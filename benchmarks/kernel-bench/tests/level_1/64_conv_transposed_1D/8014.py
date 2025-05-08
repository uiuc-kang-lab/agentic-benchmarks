
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import math

# Helper function to build and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Utility: a reference using PyTorch's ConvTranspose1d module.
def reference_conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups):
    # Construct a ConvTranspose1d with the same parameters.
    in_channels = x.size(1)
    out_channels = weight.size(0) if groups == 1 else groups * weight.size(1)
    # Note: weight expected shape in kernel is [in_channels, out_channels/groups, kernel_size]
    kernel_size = weight.size(2)
    # The PyTorch ConvTranspose1d weight shape is [in_channels, out_channels/groups, kernel_size].
    conv_transpose = nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias is not None,
    ).cuda()
    # Set weights and bias from our provided tensors:
    with torch.no_grad():
        conv_transpose.weight.copy_(weight)
        if bias is not None:
            conv_transpose.bias.copy_(bias)
    return conv_transpose(x)

# Test case 1:
# Trigger the alignment assumption issue:
# We create a tensor that is “mis‐aligned” by slicing off the first element along a flattened dimension.
def test_misaligned_tensors():
    # We create a tensor and then slice it to force a misalignment.
    # Note: Although the tensor remains contiguous in its new shape, its underlying data_ptr may no longer be 128-bit aligned.
    batch_size = 4
    in_channels = 8
    out_channels = 6
    kernel_size = 3
    length = 32
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1

    # Allocate a larger tensor and slice it to induce misalignment.
    x_big = torch.randn(batch_size, in_channels, length + 1, device="cuda", dtype=torch.float32)
    # Slicing along last dimension to get a tensor that is still contiguous but possibly misaligned.
    x = x_big.narrow(2, 1, length)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Run our custom kernel
    cuda_mod = build_kernel()
    y_kernel = cuda_mod.forward(x, weight, bias, stride, padding, output_padding, groups)

    # Compare with PyTorch reference; even if the kernel works numerically,
    # misalignment might cause differences. We check if the error is unexpectedly high.
    y_ref = reference_conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups)
    diff = (y_kernel - y_ref).abs().max().item()
    # We expect that a misaligned access might lead to numerical discrepancies.
    # (This threshold is heuristic—the key is that the outputs should agree; a failure indicates the issue.)
    assert diff < 1e-4, f"Numerical difference too high with misaligned tensors: {diff}"

# Test case 2:
# Trigger the dtype issue by passing double (float64) tensors.
def test_wrong_dtype():
    batch_size = 4
    in_channels = 8
    out_channels = 6
    kernel_size = 3
    length = 32
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)
    cuda_mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel expects float32. This call should trigger an error or lead to a crash.
        cuda_mod.forward(x, weight, bias, stride, padding, output_padding, groups)

# Test case 3:
# Trigger the inefficiency (and potential correctness issues) by using an input size where many iterations
# in the inner loop perform unnecessary work.
def test_large_input_width():
    batch_size = 2
    in_channels = 8
    out_channels = 6
    kernel_size = 3
    # Create a large spatial dimension to stress the loop over j.
    length = 1024  
    stride = 2
    padding = 1
    output_padding = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    cuda_mod = build_kernel()
    y_kernel = cuda_mod.forward(x, weight, bias, stride, padding, output_padding, groups)
    y_ref = reference_conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups)
    # Even if the kernel is inefficient, its numerical result should be (nearly) correct.
    assert torch.allclose(y_kernel, y_ref, atol=1e-4), "Kernel output differs from reference in large input width case"

# Test case 4:
# Trigger the issue when the number of input channels is not divisible by groups.
def test_invalid_group_division():
    batch_size = 4
    in_channels = 7  # purposely not divisible by groups below
    groups = 2
    # For a valid weight shape, in_channels should be divisible by groups.
    # Here we bypass that requirement by constructing a weight tensor with shape based on in_channels
    # even though the group division is invalid.
    out_channels = 8
    kernel_size = 3
    length = 32
    stride = 1
    padding = 1
    output_padding = 0

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    # Construct weight with shape: [in_channels, out_channels/groups, kernel_size]
    # When in_channels is not divisible by groups, the kernel's computed group_in_channels = in_channels/groups
    # will be rounded down (integer division), causing some channels to be ignored.
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    cuda_mod = build_kernel()
    y_kernel = cuda_mod.forward(x, weight, bias, stride, padding, output_padding, groups)
    y_ref = reference_conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups)
    # Because the custom kernel does not check that in_channels is divisible by groups,
    # the kernel output is likely to differ from the reference.
    diff = (y_kernel - y_ref).abs().max().item()
    assert diff > 1e-3, "Kernel unexpectedly produced the same result even when in_channels is not divisible by groups"

