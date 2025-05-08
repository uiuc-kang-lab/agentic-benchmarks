
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper: Build and load the CUDA extension from kernel.cu
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )

# Issue 1: Data type limitation (only float32 is supported)
def test_dtype_handling():
    cuda_mod = build_kernel()
    # Create inputs in double precision.
    batch_size = 2
    in_channels = 3
    length = 10
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.double)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.double)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.double)

    with pytest.raises(Exception):
        # This call should fail because the kernel uses data_ptr<float>() on double tensors.
        _ = cuda_mod.forward(x, weight, bias, stride, padding, dilation)

# Issue 2: Missing stride validation (stride==0 leads to division/modulo by zero)
def test_stride_zero():
    cuda_mod = build_kernel()
    batch_size = 2
    in_channels = 3
    length = 10
    out_channels = 4
    kernel_size = 3
    stride = 0  # Invalid stride: should trigger an error.
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    with pytest.raises(Exception):
        _ = cuda_mod.forward(x, weight, bias, stride, padding, dilation)

# Issue 3: Lack of output_padding support.
# The extension does not accept an output_padding parameter so the output shape
# will differ from that of PyTorch's nn.ConvTranspose1d when output_padding is nonzero.
def test_output_padding_mismatch():
    # Set up parameters that require an output_padding for the full generality.
    batch_size = 2
    in_channels = 3
    length = 8
    out_channels = 4
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    output_padding = 1  # PyTorch's ConvTranspose1d would use this to adjust output size

    # Create identical input, weight and bias.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Use our custom CUDA kernel.
    cuda_mod = build_kernel()
    output_cuda = cuda_mod.forward(x, weight, bias, stride, padding, dilation)

    # Compute expected output using nn.ConvTranspose1d with output_padding.
    conv_tp = torch.nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        bias=True
    )
    # Copy over same weights/bias
    with torch.no_grad():
        conv_tp.weight.copy_(weight)
        conv_tp.bias.copy_(bias)
    output_ref = conv_tp(x)

    # The output shapes will differ because our CUDA kernel does not support output_padding.
    assert output_cuda.shape != output_ref.shape, (
        "Expected output shapes to differ because the kernel does not implement output_padding."
    )

# Issue 4: Fixed block size may not cover all cases optimally.
# We test with an input size that gives an output with a total element count not divisible by 64.
def test_non_divisible_thread_count():
    cuda_mod = build_kernel()
    batch_size = 2
    in_channels = 3
    length = 9  # Choose length so that (batch*out_channels*output_length) is not a multiple of 64.
    out_channels = 4
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create inputs.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Run our custom CUDA kernel.
    output_cuda = cuda_mod.forward(x, weight, bias, stride, padding, dilation)

    # Run PyTorch's native ConvTranspose1d (using default output_padding=0) as reference.
    conv_tp = torch.nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True
    )
    with torch.no_grad():
        conv_tp.weight.copy_(weight)
        conv_tp.bias.copy_(bias)
    output_ref = conv_tp(x)

    # Even if the block size is fixed, the kernel should compute the correct result.
    assert torch.allclose(output_cuda, output_ref, atol=1e-4), (
        "Output differs from expected result when using non-divisible thread count."
    )
