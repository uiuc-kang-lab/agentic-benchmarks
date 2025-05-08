
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: test that using non-float32 types triggers an error or leads to incorrect behavior.
def test_dtype_issue():
    mod = build_kernel()
    # Using double instead of float32
    B, in_channels, in_size = 4, 3, 32
    out_channels = 8
    kernel_size = 3
    stride = 1
    dilation = 1
    # Create double tensors (which the kernel is not designed for)
    x = torch.randn(B, in_channels, in_size, device='cuda', dtype=torch.double)
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda', dtype=torch.double)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.double)
    with pytest.raises(RuntimeError):
        # The kernel loads raw pointers as float* and thus using double will typically cause a launch error or memory corruption.
        mod.forward(x, weight, bias, stride, dilation)

# Issue 2: test that non-contiguous inputs are rejected.
def test_non_contiguous_input():
    mod = build_kernel()
    B, in_channels, in_size = 4, 3, 32
    out_channels = 8
    kernel_size = 3
    stride = 1
    dilation = 1
    # Create a contiguous tensor then make it non-contiguous via transpose.
    x = torch.randn(B, in_channels, in_size, device='cuda', dtype=torch.float32).transpose(1,2)
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        mod.forward(x, weight, bias, stride, dilation)

# Issue 3: test for cases where the convolution parameters (valid convolution) lead to an invalid output size.
def test_invalid_output_size():
    mod = build_kernel()
    # Choose parameters so that the computed output size is <= 0.
    # out_size = (in_size - dilation*(kernel_size-1) - 1) / stride + 1 <= 0  => in_size too small.
    B, in_channels, in_size = 2, 3, 5   # Too small input
    out_channels = 4
    kernel_size = 5
    stride = 2
    dilation = 2
    x = torch.randn(B, in_channels, in_size, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda', dtype=torch.float32)
    # bias is optional
    with pytest.raises(RuntimeError, match="Invalid output size"):
        mod.forward(x, weight, None, stride, dilation)

# Issue 4: test for potential problems with the fixed unroll factor when kernel_size < 4.
def test_unroll_issue():
    mod = build_kernel()
    # Using a kernel_size less than 4 to force the unroll limit to overshoot the loop iterations.
    # If the unroll directive is misapplied, the kernel result might be incorrect.
    B, in_channels, in_size = 2, 3, 20
    out_channels = 4
    kernel_size = 2  # less than the unroll factor of 4
    stride = 1
    dilation = 1
    x = torch.randn(B, in_channels, in_size, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    # Compute expected result with torch's conv1d (with no padding, valid convolution)
    ref = torch.nn.functional.conv1d(x, weight, bias=bias, stride=stride, dilation=dilation)
    out = mod.forward(x, weight, bias, stride, dilation)
    # If the unroll directive does not work correctly for kernel_size < 4 the result may differ.
    assert torch.allclose(out, ref, atol=1e-5), f"Kernel output ({out}) differs from reference ({ref})!"

