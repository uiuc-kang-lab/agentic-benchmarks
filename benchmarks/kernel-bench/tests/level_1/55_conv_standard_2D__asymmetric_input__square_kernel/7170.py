
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA kernel extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper to create a dummy convolution operation using the CUDA kernel.
# We assume the weight tensor is of shape (out_channels, in_channels, kernel_h, kernel_w).
def conv2d_cuda(cuda_module, x, weight, bias, stride, padding, dilation, groups):
    return cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_groups_not_supported():
    # Create a valid contiguous float32 input and weight.
    batch = 4
    in_channels = 3
    out_channels = 8
    height, width = 32, 32
    kernel_size = 3

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Weight expected shape: (out_channels, in_channels, kernel_size, kernel_size)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = None  # or torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    cuda_module = build_kernel()
    # groups != 1 should trigger a TORCH_CHECK error.
    with pytest.raises(RuntimeError, match="Only groups==1 is supported."):
        conv2d_cuda(cuda_module, x, weight, bias, stride=1, padding=0, dilation=1, groups=2)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_square_kernel():
    # Although the high-level PyTorch API might enforce square kernels when given an int,
    # here we manually create a non-square weight tensor.
    batch = 2
    in_channels = 3
    out_channels = 5
    height, width = 28, 28
    kernel_h, kernel_w = 3, 5  # non-square kernel

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Create non-square weight tensor.
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    cuda_module = build_kernel()
    # The kernel assumes square kernels (using weight.size(2) as kernel size). This mismatch
    # may lead to out-of-bound accesses or wrong computation.
    # Here, we at least assert that the kernel runs without a proper check,
    # but the output will be incorrect; we simulate the potential issue by checking a result mismatch.
    out = conv2d_cuda(cuda_module, x, weight, bias, stride=1, padding=0, dilation=1, groups=1)
    # We intentionally do not know the correct output. Instead, we check that the output shape does not match
    # what would be computed for a square kernel.
    expected_kernel = kernel_h  # kernel_h != kernel_w, so the assumption is violated.
    expected_out_height = (height + 0 - 1 * (expected_kernel - 1) - 1) + 1
    expected_out_width  = (width + 0 - 1 * (expected_kernel - 1) - 1) + 1
    assert out.shape[2] != expected_out_height or out.shape[3] != expected_out_width, \
           "Non-square kernel should trigger an issue with output dimensions."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_float32_input():
    # Create an input tensor with a non-supported data type (float64).
    batch = 4
    in_channels = 3
    out_channels = 8
    height, width = 32, 32
    kernel_size = 3

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float64)
    bias = None
    
    cuda_module = build_kernel()
    # Since the kernel calls data_ptr<float>() it expects float32.
    # This should either lead to a runtime error or produce incorrect results,
    # so we wrap it in a pytest.raises to catch any runtime error.
    with pytest.raises(RuntimeError):
        conv2d_cuda(cuda_module, x, weight, bias, stride=1, padding=0, dilation=1, groups=1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_input():
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    batch = 4
    in_channels = 3
    out_channels = 8
    height, width = 32, 32
    kernel_size = 3

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Make x non-contiguous.
    x_non_contiguous = x.transpose(1, 2)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        conv2d_cuda(cuda_module, x_non_contiguous, weight, bias, stride=1, padding=0, dilation=1, groups=1)
