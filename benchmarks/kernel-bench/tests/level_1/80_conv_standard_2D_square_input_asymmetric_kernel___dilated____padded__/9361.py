
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu.
    cuda_module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 Test: Input data type not float32
def test_dtype_support():
    cuda_module = build_kernel()
    device = "cuda"
    batch_size = 1
    in_channels = 3
    out_channels = 8
    height, width = 32, 32
    kernel_size = (3, 3)
    # Create double precision input and weights, even though our kernel only supports float32.
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device=device, dtype=torch.float64)
    bias = torch.randn(out_channels, device=device, dtype=torch.float64)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)
    # Expect a RuntimeError (or similar) due to type mismatch when launching the kernel.
    with pytest.raises(RuntimeError):
        # This should fail because our kernel assumes float (float32) inputs.
        cuda_module.forward(x, weight, bias, stride, padding, dilation)

# Issue 2 Test: Kernel size larger than MAX_KERNEL_SIZE
def test_kernel_size_limit():
    cuda_module = build_kernel()
    device = "cuda"
    batch_size = 1
    in_channels = 3
    out_channels = 8
    height, width = 64, 64
    # Use a kernel size larger than MAX_KERNEL_SIZE (7), e.g. 9x9.
    kernel_size = (9, 9)
    # Create input tensors of type float32
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device=device, dtype=torch.float32)
    bias = torch.randn(out_channels, device=device, dtype=torch.float32)
    stride = 1
    # Set padding to correctly match; here just use 4 for 9x9 kernel.
    padding = (4, 4)
    dilation = (1, 1)
    # Run our custom CUDA kernel
    out_custom = cuda_module.forward(x, weight, bias, stride, padding, dilation)
    # Compute a reference output using PyTorch's built in conv2d function.
    out_ref = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    # The custom kernel is not designed to support kernels larger than MAX_KERNEL_SIZE.
    # We expect that the result will be numerically different.
    # This test asserts that the maximum absolute difference exceeds a tolerance.
    max_diff = (out_custom - out_ref).abs().max().item()
    assert max_diff > 1e-3, f"Expected significant error for kernels larger than MAX_KERNEL_SIZE, got max diff {max_diff}"
