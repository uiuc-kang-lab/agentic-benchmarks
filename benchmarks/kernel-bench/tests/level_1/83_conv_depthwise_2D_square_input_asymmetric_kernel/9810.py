
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Make sure the file kernel.cu is in the current directory.
    module = load(
        name="depthwise_conv_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

@pytest.fixture(scope="module")
def cuda_module():
    return build_kernel()

def get_output_dims(input_size, kernel_size, stride, padding, dilation, kernel_width=1):
    # Compute output dimensions based on the formulas in the forward() function.
    in_h, in_w = input_size
    out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - (kernel_width - 1) - 1) // stride + 1
    return out_h, out_w

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_width_greater_than_one(cuda_module):
    # This test will trigger issue 1:
    # The CUDA kernel assumes kernel width == 1.
    # We create a weight tensor with kernel width > 1 and pass it to the kernel.
    batch = 2
    channels = 3
    in_h, in_w = 32, 32
    kernel_h, kernel_w = 3, 3  # kernel width > 1
    stride = 1
    padding = 1
    dilation = 1

    # Create an input tensor and a weight tensor with an explicit kernel width > 1.
    x = torch.randn(batch, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    # Simulate weight shape: (channels, 1, kernel_h, kernel_w). The current kernel
    # only reads the kernel_h dimension and ignores the kernel width.
    weight = torch.randn(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    # Pass bias as None to get zeros.
    bias = torch.zeros(channels, device="cuda", dtype=torch.float32)

    # Compute expected output dimensions assuming proper handling of kernel width:
    # For a real convolution, out_w should be computed with kernel_w, but our kernel uses (in_w+2*padding-1)
    out_h, out_w = get_output_dims((in_h, in_w), kernel_h, stride, padding, dilation, kernel_width=kernel_w)

    # Call the CUDA kernel function.
    # Note: The forward() function in the CUDA module does not account for kernel width,
    # so the result is expected to be wrong if the kernel width is >1.
    output = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups=channels)

    # The test does not check for numerical correctness since our expected output
    # (computed as in a true convolution) would differ from the incorrect CUDA kernel result.
    # We trigger a failure by asserting a property that should hold if the kernel were general.
    assert output.size(3) == out_w, "Output width calculated in CUDA kernel does not match expected (generalized) output width for kernel width > 1. This triggers Issue 1."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_float32(cuda_module):
    # This test will trigger issue 2:
    # The kernel does not support tensor types other than float32.
    batch = 2
    channels = 3
    in_h, in_w = 32, 32
    kernel_h = 3  # With kernel width assumed to be 1 by the kernel
    stride = 1
    padding = 0
    dilation = 1

    # Create double precision input and weight tensors.
    x = torch.randn(batch, channels, in_h, in_w, device="cuda", dtype=torch.float64)
    weight = torch.randn(channels, 1, kernel_h, 1, device="cuda", dtype=torch.float64)
    bias = torch.zeros(channels, device="cuda", dtype=torch.float64)

    # The kernel expects float pointers. Thus, calling the CUDA kernel with double tensors potentially
    # leads to incorrect memory interpretation.
    with pytest.raises(RuntimeError):
        # Expect an error because the CUDA kernel code uses float pointers to access data.
        cuda_module.forward(x, weight, bias, stride, padding, dilation, groups=channels)
