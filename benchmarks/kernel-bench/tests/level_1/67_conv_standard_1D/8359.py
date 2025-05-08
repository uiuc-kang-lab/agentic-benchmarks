
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    # Build the extension module from kernel.cu
    cuda_module = load(
        name="conv1d_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: run the custom CUDA kernel conv1d forward and return tensor result.
def run_custom_conv1d(x, weight, bias, stride, padding, dilation, groups):
    mod = build_kernel()
    # Call our custom forward; we assume the binding as defined in PYBIND11_MODULE.
    result = mod.forward(x, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()
    return result

# Test Case 1: Incorrect grid dimension handling (batch dimension not processed correctly).
# When N > 1, the expected behavior is not reached due to the kernel ignoring gridDim.z.
def test_batch_dimension_misindexing():
    # Use a batch size > 1 to reveal batch indexing issues.
    batch_size = 8  # >1
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    length = 32
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    # Create a standard Conv1d for reference.
    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True
    ).cuda()
    # Make sure our custom weight and bias match the nn.Conv1d parameters.
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    weight = conv.weight.detach()
    bias = conv.bias.detach() if conv.bias is not None else None
    
    with torch.no_grad():
        expected = conv(x)
        custom = run_custom_conv1d(x, weight, bias, stride, padding, dilation, groups)
    # The outputs should be close; if gridDim.z is ignored then for N>1 the batch elements will be computed incorrectly.
    assert not torch.allclose(expected, custom, atol=1e-5), \
        f"Test Failed: The custom kernel output matches expected output, but batch dimension misindexing issue was expected."

# Test Case 2: Unsafe vectorized load when kernel size is not a multiple of 4.
# For kernel_size=3, the conversion to float4 is unsafe and may read out-of-bounds,
# causing an incorrect result.
def test_vectorized_load_issue():
    batch_size = 4
    in_channels = 2
    out_channels = 2
    kernel_size = 3  # Not a multiple of 4.
    length = 25
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True
    ).cuda()
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    weight = conv.weight.detach()
    bias = conv.bias.detach() if conv.bias is not None else None

    with torch.no_grad():
        expected = conv(x)
        custom = run_custom_conv1d(x, weight, bias, stride, padding, dilation, groups)
    # Expecting a mismatch due to potential misaligned/illegal vectorized loads.
    assert not torch.allclose(expected, custom, atol=1e-4), \
        f"Test Failed: The custom kernel output unexpectedly matches expected output, vectorized load issue may not have been triggered."

# Test Case 3: Hard-coded loop unrolling assumption failure.
# The pragma unroll 4 assumes the kernel size is a multiple of 4.
# For kernel_size=3 the accumulation loop may not cover all kernel taps correctly.
def test_loop_unroll_issue():
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3  # Not a multiple of 4, unrolling assumption breaks.
    length = 40
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=False
    ).cuda()
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    weight = conv.weight.detach()

    with torch.no_grad():
        expected = conv(x)
        custom = run_custom_conv1d(x, weight, None, stride, padding, dilation, groups)
    # Since unrolling may drop some contributions, the results will differ.
    assert not torch.allclose(expected, custom, atol=1e-4), \
        f"Test Failed: The custom kernel output unexpectedly matches expected output, loop unroll issue was expected to occur."

