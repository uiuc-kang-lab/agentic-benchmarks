
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA kernel from kernel.cu.
    # Make sure that kernel.cu is in the working directory.
    return load(
        name="avg_pool_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )

def reference_avg_pool1d(x, kernel_size, stride, padding):
    # Compute average pooling using PyTorch's native implementation.
    avg_pool = nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
    return avg_pool(x)

# Issue 1: Non-contiguous input tensor handling.
def test_non_contiguous_input():
    kernel_module = build_kernel()
    batch_size = 4
    in_channels = 8
    input_length = 64
    kernel_size = 3
    stride = 1
    padding = 1

    # Create a contiguous input tensor.
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    # Force non-contiguity by transposing and then transposing back.
    x_non_contig = x.transpose(1, 2).transpose(1, 2)
    assert not x_non_contig.is_contiguous(), "Input tensor should be non-contiguous for this test."

    # Run the custom CUDA kernel.
    output_kernel = kernel_module.forward(x_non_contig, kernel_size, stride, padding)
    # Run the reference PyTorch avg pool.
    output_ref = reference_avg_pool1d(x_non_contig, kernel_size, stride, padding)

    # We expect that, due to the kernel’s assumption of contiguity, the outputs will differ.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
        "Kernel output should differ from the reference output on non-contiguous inputs."
    )

# Issue 2: Data type limitation (only float32 is supported).
def test_input_dtype():
    kernel_module = build_kernel()
    batch_size = 4
    in_channels = 8
    input_length = 64
    kernel_size = 3
    stride = 1
    padding = 1

    # Create an input tensor with half precision.
    x_half = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.half)

    # The kernel is written for float32 so passing half precision should raise an error.
    with pytest.raises(RuntimeError):
        kernel_module.forward(x_half, kernel_size, stride, padding)

# Issue 3: Runtime kernel_size prevents proper unrolling.
def test_dynamic_kernel_size():
    kernel_module = build_kernel()
    batch_size = 4
    in_channels = 8
    input_length = 128
    # Choose a kernel_size that is not typical (and thus less amenable to compile-time unrolling)
    kernel_size = 7
    stride = 2
    padding = 2

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    output_kernel = kernel_module.forward(x, kernel_size, stride, padding)
    output_ref = reference_avg_pool1d(x, kernel_size, stride, padding)

    # Because of the use of #pragma unroll on a runtime parameter, rounding or unrolling issues may cause
    # the kernel’s output to deviate from the expected reference.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
        "Kernel output should differ from the reference output when using a dynamic kernel_size."
    )
