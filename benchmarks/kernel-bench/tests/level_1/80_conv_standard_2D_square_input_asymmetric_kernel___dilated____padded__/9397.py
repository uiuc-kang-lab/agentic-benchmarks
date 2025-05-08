
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to compile the CUDA extension.
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Kernel hard-coded to work with float32.
# Test: Passing double precision tensors should produce incorrect results (or crash)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_float32_dtype():
    kernel = build_kernel()
    # Create double precision inputs; our kernel expects float32.
    batch_size = 2
    in_channels = 3
    height = width = 16
    out_channels = 4
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)

    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float64)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float64)

    # This call will pass pointers to double data to a kernel expecting floats.
    # The result is expected to be incorrect (or cause a runtime error).
    with pytest.raises(Exception):
        output = kernel.forward(x, weight, bias, stride, padding, dilation)
        # Synchronize to catch kernel errors.
        torch.cuda.synchronize()


# Issue 2: Kernel may fail when out_channels exceed allowed gridDim.z range.
# Test: Create a conv with an extremely large number of output channels.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_out_channels():
    kernel = build_kernel()
    batch_size = 1
    in_channels = 1
    height = width = 8
    # Using a very large out_channels value to force gridDim.z overflow.
    # (Note: This test may consume a large amount of memory or trigger a launch error.)
    out_channels = 70000  # typically exceeds maximum gridDim.z
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)

    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    with pytest.raises(Exception):
        output = kernel.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()


# Issue 3: No post-launch error check in kernel.
# Test: Force an error by using a non-contiguous tensor; the TORCH_CHECK should fire.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_input():
    kernel = build_kernel()
    batch_size = 2
    in_channels = 3
    height = width = 16
    out_channels = 4
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)

    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    # Make x non contiguous
    x = x.transpose(2, 3)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    with pytest.raises(RuntimeError):
        output = kernel.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()


# Issue 4: Stride is only supported as an integer (uniform stride).
# Test: Provide a tuple for stride; this should raise a type error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asymmetric_stride_not_supported():
    kernel = build_kernel()
    batch_size = 2
    in_channels = 3
    height = width = 16
    out_channels = 4
    kernel_size = (3, 3)
    # Here we supply a tuple for stride even though the kernel expects an int.
    stride = (1, 2)
    padding = (1, 1)
    dilation = (1, 1)

    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    with pytest.raises(TypeError):
        # The argument conversion in Python should raise an error if stride is not an int.
        output = kernel.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()
