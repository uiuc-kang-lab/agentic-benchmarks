
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Helper to build and load the CUDA extension module
def build_kernel():
    cuda_module = load(
        name="fused_avg_pool2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference implementation using PyTorch's AvgPool2d
def pytorch_avg_pool(input, kernel_size, stride, padding):
    avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    return avg_pool(input)

# Test case 1: Trigger potential issues with runtime loop parameters that prevent compile‐time loop unrolling
def test_runtime_loop_unroll():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Choose a kernel size that is not small (and thus less likely to be optimized as a constant)
    kernel_size = 7
    stride = kernel_size
    padding = 1  # non-zero to also exercise border behavior
    N, C, H, W = 4, 3, 32, 32
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    ref = pytorch_avg_pool(x, kernel_size, stride, padding)

    fused_kernel = build_kernel()
    out = fused_kernel.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    # Allow a certain tolerance since floating point might differ slightly
    assert torch.allclose(out, ref, atol=1e-5), "Output from fused kernel differs from PyTorch reference (runtime loop unroll test)"

# Test case 2: Use float16 input to trigger the unsupported half precision issue
def test_unsupported_half_precision():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    kernel_size = 3
    stride = kernel_size
    padding = 0
    N, C, H, W = 4, 3, 16, 16
    # Create a half precision tensor
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    fused_kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the AT_DISPATCH_FLOATING_TYPES macro to not cover float16, and thus throw an error.
        _ = fused_kernel.forward(x, kernel_size, stride, padding)

# Test case 3: Provide a non-contiguous input to trigger the assumption of contiguity in the kernel
def test_non_contiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    kernel_size = 3
    stride = kernel_size
    padding = 0
    N, C, H, W = 4, 3, 20, 20
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    # Create a non-contiguous version of the input (e.g., by transposing)
    x_noncontig = x.transpose(1, 2)
    # The host part of the extension calls .contiguous(), so even a non-contiguous input should work correctly.
    fused_kernel = build_kernel()
    out = fused_kernel.forward(x_noncontig, kernel_size, stride, padding)
    # As the kernel expects NCHW, we need to convert back to compare with reference.
    # Permute the output back to the original layout for comparison.
    out = out.transpose(1, 2)
    ref = pytorch_avg_pool(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), "Output for non-contiguous input does not match the reference output"

# Test case 4: Use input sizes that may not be ideal for the fixed thread block configuration,
# potentially exposing underutilization or grid configuration issues.
def test_fixed_thread_block_config():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Choose an input size that generates a very small output grid.
    kernel_size = 3
    stride = 3
    padding = 0
    N, C, H, W = 2, 2, 10, 10  # small spatial dimensions
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    ref = pytorch_avg_pool(x, kernel_size, stride, padding)
    
    fused_kernel = build_kernel()
    out = fused_kernel.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), "Output from fused kernel with small grid does not match PyTorch reference"
