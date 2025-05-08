
import torch
import pytest
from torch.nn.functional import conv_transpose1d
from torch.utils.cpp_extension import load

# Helper to build the kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only works with float32. If the input dtype is not float32
# then interpreting the memory as floats will lead to incorrect results.
def test_dtype_float64():
    cuda_module = build_kernel()

    # Create input and weight of type float64.
    # We use conv_transpose1d built-in to generate expected results, and then we force dtypes.
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    length = 20
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, length, dtype=torch.float64, device="cuda")
    # The weight shape for ConvTranspose1d (groups=1) is [in_channels, out_channels, kernel_size]
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float64, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda")

    # Expected using PyTorch built-in conv_transpose1d with converted dtype.
    expected = conv_transpose1d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)

    # Call the custom CUDA kernel.
    # NOTE: The kernel expects float*, so we deliberately pass a float64 tensor.
    # This is expected to yield wrong results because the kernel misinterprets the data.
    got = cuda_module.forward(x, weight, bias, int(stride), int(padding), int(dilation))

    # Because the kernel is written for float32, the computed values will be off.
    # We check that (with a loose tolerance) the kernel's output does not match the expected output.
    assert not torch.allclose(got, expected, atol=1e-5), \
        "Test failure expected: using float64 should not yield correct results."

# Issue 2: The kernel does not support grouped convolution.
# When groups > 1, PyTorch rearranges weight tensor dimensions.
def test_grouped_convolution():
    cuda_module = build_kernel()

    # Create a grouped ConvTranspose1d: groups > 1.
    # For a grouped ConvTranspose1d in PyTorch, if in_channels=4, out_channels=4 and groups=2,
    # the actual weight shape is (in_channels, out_channels//groups, kernel_size) = (4, 2, kernel_size).
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    length = 20
    stride = 1
    padding = 0
    dilation = 1
    groups = 2

    # Create an input tensor.
    x = torch.randn(batch_size, in_channels, length, device="cuda")
    # Create weight and bias tensors accordingly.
    # Note: PyTorch's nn.ConvTranspose1d for grouped conv expects weight shape [in_channels, out_channels//groups, kernel_size].
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, device="cuda")
    bias = torch.randn(out_channels, device="cuda")

    # Get output from the PyTorch implementation.
    expected = conv_transpose1d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # The custom kernel expects weight with shape [C_in, C_out, kernel_size] for groups == 1.
    # To force using it with groups > 1, we deliberately reshape the weight to mimic a non-grouped layout.
    # This is incorrect if groups > 1 and should cause the output to differ from PyTorch's expected output.
    # For example, we can reinterpret the weight as if:
    #   effective_weight = weight.view(in_channels, out_channels, kernel_size)
    weight_for_kernel = weight.view(in_channels, out_channels, kernel_size)

    got = cuda_module.forward(x, weight_for_kernel, bias, int(stride), int(padding), int(dilation))

    # The two outputs should differ because the kernel does not support grouped convolution.
    assert not torch.allclose(got, expected, atol=1e-5), \
        "Test failure expected: grouped convolution is not supported by this kernel, so outputs must differ."

