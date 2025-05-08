
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Utility to build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="depthwise_conv_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that the "groups" parameter is ignored.
# We create a situation where groups != in_channels.
# The built-in nn.Conv2d in PyTorch uses the "groups" param.
# Our kernel always performs a depthwise convolution, so if groups != in_channels,
# the output will differ from a correctly configured convolution.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_groups_ignored():
    batch_size = 2
    in_channels = 4
    kernel_size = 3
    stride = 1
    padding = 1
    groups_mismatch = 2  # deliberately not equal to in_channels (should be 4 for depthwise)

    # Create non-depthwise weight for built-in convolution (groups mismatch)
    # and also create a weight that would be interpreted as depthwise in our kernel.
    # For our kernel the expected weight shape is (in_channels, 1, kernel_size, kernel_size).
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    # Input tensor is contiguous.
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)

    # Create our custom kernel output with groups=groups_mismatch.
    depthwise_conv_module = build_kernel()
    out_kernel = depthwise_conv_module.forward(x, weight, bias, stride, padding, groups_mismatch)

    # Create a PyTorch depthwise convolution module.
    # PyTorch requires groups==in_channels for depthwise convolution.
    # Here we force the built-in conv2d to use groups=in_channels,
    # then we compare to our kernel output.
    conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                     groups=in_channels, bias=True).cuda()

    # Manually set the weights and bias.
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    out_ref = conv(x)

    # Since our kernel ignores groups and always does depthwise, the case when groups != in_channels
    # is an issue. We expect the outputs to differ from what a convolution configured with groups=groups_mismatch
    # would yield. Here, we check that our kernel output matches the depthwise conv output,
    # implying the groups parameter is being ignored.
    assert torch.allclose(out_kernel, out_ref, atol=1e-4), \
        f"Kernel did not ignore groups parameter as expected. max diff: {(out_kernel - out_ref).abs().max().item()}"


# Issue 2: Test non-contiguous inputs.
# We create non -contiguous tensors for input and weight, which the kernel assumes to be contiguous.
# This may cause mis-indexing or wrong results.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_tensors():
    batch_size = 2
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1

    # Create contiguous tensors and then make them non-contiguous by transposing some dims
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32).permute(0, 2, 3, 1)
    # Force non-contiguity for weight: original shape (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32).transpose(2,3)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Our kernel does not check for contiguity so we expect its output to be incorrect.
    depthwise_conv_module = build_kernel()
    try:
        out_kernel = depthwise_conv_module.forward(x, weight, bias, stride, padding, in_channels)
    except Exception as e:
        pytest.skip(f"Kernel raised an exception on non-contiguous tensors: {e}")

    # For reference, force contiguity and compute the correct output.
    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                     groups=in_channels, bias=True).cuda()
    with torch.no_grad():
        conv.weight.copy_(weight_contig)
        conv.bias.copy_(bias)
    out_ref = conv(x_contig)

    # The outputs should differ because our kernel assumes contiguous memory.
    # We assert that they are not all-close to highlight the issue.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-3), \
        "Kernel produced correct result on non-contiguous tensors, but it was expected to fail due to lack of checks."


# Issue 3: Test unsupported data types.
# Using an integer tensor (or half precision) should trigger an error because AT_DISPATCH_FLOATING_TYPES
# only supports float32 and float64.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unsupported_dtype():
    batch_size = 2
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1

    x = torch.randint(0, 10, (batch_size, in_channels, 16, 16), device="cuda", dtype=torch.int32)
    weight = torch.randint(0, 10, (in_channels, 1, kernel_size, kernel_size), device="cuda", dtype=torch.int32)
    bias = torch.randint(0, 10, (in_channels,), device="cuda", dtype=torch.int32)

    depthwise_conv_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting a runtime error because int32 is not handled by AT_DISPATCH_FLOATING_TYPES.
        _ = depthwise_conv_module.forward(x, weight, bias, stride, padding, in_channels)


# Issue 4: Test non-square kernel.
# The kernel is hard-coded for a square kernel: both dimensions are taken from weight.size(2).
# If the weight provided is non-square this will lead to incorrect indexing.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_square_kernel():
    batch_size = 2
    in_channels = 3
    # Define a non-square kernel: kernel height 3, kernel width 5.
    kernel_h = 3
    kernel_w = 5
    stride = 1
    padding = 1

    # Our kernel implementation expects weight shape (in_channels, 1, kernel_size, kernel_size)
    # where kernel_size is a single scalar. We deliberately construct a weight tensor with different sizes.
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)

    depthwise_conv_module = build_kernel()
    out_kernel = depthwise_conv_module.forward(x, weight, bias, stride, padding, in_channels)

    # For reference, construct a convolution module that supports non-square kernels.
    # PyTorch’s Conv2d accepts non-square kernels.
    conv = nn.Conv2d(in_channels, in_channels, (kernel_h, kernel_w),
                     stride=stride, padding=padding, groups=in_channels, bias=True).cuda()
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    out_ref = conv(x)

    # Because our kernel uses kernel_h (via weight.size(2)) for both dimensions, the result will be wrong.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-3), \
        "Kernel output is close to reference output despite non-square kernel, but it was expected to fail."

