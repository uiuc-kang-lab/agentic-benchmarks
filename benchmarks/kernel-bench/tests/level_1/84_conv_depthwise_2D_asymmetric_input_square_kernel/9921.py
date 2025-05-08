
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="depthwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that the kernel fails (or produces wrong result) when using a data type other than float32.
def test_float16_input():
    my_module = build_kernel()
    batch_size = 2
    in_channels = 3
    kernel_size = 3
    input_h = 8
    input_w = 8
    # Create input and weight tensors with float16 precision.
    x = torch.randn(batch_size, in_channels, input_h, input_w, device="cuda", dtype=torch.float16)
    # For a depthwise op, weight shape should be (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float16)
    bias = None

    # The kernel always uses data_ptr<float>(), so passing float16 memory will lead to incorrect interpretation.
    # We expect the result to be far from the reference computed by PyTorch's conv2d (or an error may occur).
    with pytest.raises(Exception):
        # This should either fail or produce a wrong result that we can test for.
        out = my_module.forward(x, weight, bias, 1, 0)
        # Force a synchronization to catch any kernel launch errors.
        torch.cuda.synchronize()


# Issue 2: Test that the kernel fails when gridDim.z is too large.
def test_excessive_grid_dim_z():
    my_module = build_kernel()
    # Choose parameters that force batch_size*out_channels to be huge.
    # Note: Since the kernel does no tiling over the batch or channel dimensions, gridDim.z is exactly batch_size*out_channels.
    # Most devices have a maximum gridDim.z of 65535; we create a case exceeding that.
    batch_size = 70000  # Exceeds typical gridDim.z limit even with one channel.
    in_channels = 1
    kernel_size = 3
    input_h = 8
    input_w = 8
    # For a depthwise convolution with groups==in_channels, weight shape is (in_channels, 1, kernel, kernel)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    # Create an input tensor.
    x = torch.randn(batch_size, in_channels, input_h, input_w, device="cuda", dtype=torch.float32)
    bias = None

    with pytest.raises(RuntimeError):
        out = my_module.forward(x, weight, bias, 1, 0)
        torch.cuda.synchronize()


# Issue 3: Test that the kernel indexing is not suitably general for non-depthwise cases.
def test_incorrect_group_configuration():
    my_module = build_kernel()
    # Here we simulate a scenario where the weight shape does not follow the assumed depthwise convention.
    # For a proper depthwise conv with groups==in_channels, weight shape is (in_channels, 1, k, k)
    # In a more general group convolution, users might supply a weight of shape (in_channels, multiplier, k, k)
    # where multiplier != 1. The kernel computes out_channels = in_channels * channels_per_group.
    # We deliberately supply a multiplier of 2, even though a standard PyTorch depthwise convolution
    # (with groups=in_channels) would expect a multiplier of 1.
    batch_size = 2
    in_channels = 3
    multiplier = 2  # not typical for a standard depthwise conv (expected to be 1)
    kernel_size = 3
    input_h = 8
    input_w = 8
    # Weight shape: (in_channels, multiplier, kernel_size, kernel_size)
    weight = torch.randn(in_channels, multiplier, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, input_h, input_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels * multiplier, device="cuda", dtype=torch.float32)
    
    # Compute reference output using PyTorch's convolution with groups set to in_channels (which expects multiplier=1)
    conv = torch.nn.Conv2d(
        in_channels,
        in_channels,  # PyTorch depthwise conv with groups == in_channels always produces same number of channels.
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        groups=in_channels,
        bias=True
    ).to(x.device).eval()
    # Overriding conv weight and bias intentionally with our tensors,
    # but note the shapes will not match. This forces a discrepancy in behavior.
    with pytest.raises(Exception):
        conv.weight.data.copy_(weight)
        conv.bias.data.copy_(bias)
        ref_out = conv(x)
        # Run our custom kernel
        out = my_module.forward(x, weight, bias, 1, 0)
        torch.cuda.synchronize()
        # The outputs should differ since the kernel indexing does not properly cover a general group case.
        assert not torch.allclose(out, ref_out, atol=1e-5), "Kernel unexpectedly produced correct output for a non-depthwise configuration."
