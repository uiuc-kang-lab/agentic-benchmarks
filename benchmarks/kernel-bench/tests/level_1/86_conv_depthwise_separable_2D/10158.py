
import pytest
import torch
from torch.nn import Conv2d
from torch.utils.cpp_extension import load

# Utility to build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="depthwise_pointwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: a simple forward function calling the extension's forward wrapper.
def run_forward(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation):
    mod = build_kernel()
    result = mod.forward(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)
    torch.cuda.synchronize()
    return result

# Issue 1: Trigger problems with loop unrolling when the kernel size is not known at compile time.
# (Here we use a kernel size of 5 instead of the hardcoded expectation, which may expose problems with unrolling.)
def test_runtime_kernel_size_unroll():
    batch = 2
    in_channels = 3
    out_channels = 8
    k = 5  # non-standard kernel size, forcing runtime-bound loop iterations
    height, width = 32, 32
    stride = 1
    padding = 2  # to keep output same size
    dilation = 1

    # Create a depthwise conv weight with shape (in_channels, 1, k, k)
    depthwise_weight = torch.randn(in_channels, 1, k, k, device="cuda", dtype=torch.float32).contiguous()
    # Create a pointwise conv weight with shape (out_channels, in_channels, 1, 1)
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device="cuda", dtype=torch.float32).contiguous()
    # No bias for simplicity
    depthwise_bias = torch.empty(0, device="cuda", dtype=torch.float32)
    pointwise_bias = torch.empty(0, device="cuda", dtype=torch.float32)

    # Input tensor
    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32).contiguous()

    # Call our kernel; if the unroll pragma causes any problems with non-constant k,
    # the output might be wrong (or an error might be raised).
    output = run_forward(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)

    # Reference using PyTorch's built-in depthwise and pointwise operations
    # Setting groups for depthwise convolution.
    depthwise_ref = torch.nn.functional.conv2d(
        x, depthwise_weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=in_channels
    )
    pointwise_ref = torch.nn.functional.conv2d(depthwise_ref, pointwise_weight, bias=None, stride=1, padding=0)
    # Compare shapes and values
    assert output.shape == pointwise_ref.shape, f"Output shape {output.shape} differs from reference {pointwise_ref.shape}"
    # Allow some tolerance as differences may come from reduced optimization
    assert torch.allclose(output, pointwise_ref, atol=1e-4), "Output values differ from reference (possible unroll issue)"

# Issue 2: Test noncontiguous input.
def test_noncontiguous_input():
    batch = 2
    in_channels = 3
    out_channels = 8
    k = 3
    height, width = 32, 32
    stride = 1
    padding = 1  # to have same spatial size
    dilation = 1

    # Create contiguous weights and biases.
    depthwise_weight = torch.randn(in_channels, 1, k, k, device="cuda", dtype=torch.float32).contiguous()
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device="cuda", dtype=torch.float32).contiguous()
    depthwise_bias = torch.empty(0, device="cuda", dtype=torch.float32)
    pointwise_bias = torch.empty(0, device="cuda", dtype=torch.float32)

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Make x noncontiguous by transposing two spatial dimensions then transposing back.
    noncontig_x = x.transpose(2, 3)
    noncontig_x = noncontig_x.transpose(2, 3)
    # Check that the tensor is indeed noncontiguous.
    assert not noncontig_x.is_contiguous(), "Test input is unexpectedly contiguous."

    # Run our kernel forward.
    output = run_forward(noncontig_x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)

    # Compute reference with built-in conv2d (which supports noncontiguous inputs).
    depthwise_ref = torch.nn.functional.conv2d(
        noncontig_x, depthwise_weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=in_channels
    )
    pointwise_ref = torch.nn.functional.conv2d(depthwise_ref, pointwise_weight, bias=None, stride=1, padding=0)
    assert torch.allclose(output, pointwise_ref, atol=1e-4), "Kernel output does not match reference for noncontiguous input"

# Issue 3: Test that using a floating point type not handled by AT_DISPATCH_FLOATING_TYPES (e.g. float16) raises an error.
def test_unsupported_dtype():
    batch = 2
    in_channels = 3
    out_channels = 8
    k = 3
    height, width = 32, 32
    stride = 1
    padding = 1
    dilation = 1

    depthwise_weight = torch.randn(in_channels, 1, k, k, device="cuda", dtype=torch.float16).contiguous()
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device="cuda", dtype=torch.float16).contiguous()
    depthwise_bias = torch.empty(0, device="cuda", dtype=torch.float16)
    pointwise_bias = torch.empty(0, device="cuda", dtype=torch.float16)

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float16).contiguous()

    with pytest.raises(RuntimeError):
        # This should raise because our AT_DISPATCH_FLOATING_TYPES kernel does not cover float16.
        run_forward(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)
