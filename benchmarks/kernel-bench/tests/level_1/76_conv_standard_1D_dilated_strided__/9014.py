
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import math

# Utility to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="conv1d_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test for float32-only support
def test_dtype_issue():
    # Create input and weight with float64 (double) type.
    B, in_channels, length = 4, 3, 32
    out_channels, kernel_size = 8, 3
    x = torch.randn(B, in_channels, length, dtype=torch.float64, device='cuda')
    weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.float64, device='cuda')
    bias = torch.randn(out_channels, dtype=torch.float64, device='cuda')
    
    conv1d_kernel = build_kernel()
    
    # Expecting the kernel to fail or produce incorrect results because it was designed for float32.
    with pytest.raises(Exception):
        conv1d_kernel.forward(x, weight, bias, stride=1, dilation=1)

# Issue 2: Test for lack of dilation check (dilation <= 0)
def test_dilation_zero_issue():
    # Create a simple test case where dilation = 0.
    B, in_channels, length = 2, 3, 20
    out_channels, kernel_size = 4, 3
    # Using float32 (as required by the kernel)
    x = torch.randn(B, in_channels, length, dtype=torch.float32, device='cuda')
    weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.float32, device='cuda')
    bias = torch.randn(out_channels, dtype=torch.float32, device='cuda')
    
    conv1d_kernel = build_kernel()
    # When dilation=0, the kernel will use the same input element repeatedly.
    output = conv1d_kernel.forward(x, weight, bias, stride=1, dilation=0)
    
    # Construct a regular nn.Conv1d with dilation=1 for comparison.
    # Since PyTorch's Conv1d does not allow dilation=0, we emulate what the kernel does:
    # It will repeatedly use the input at start_pos for every kernel element.
    # Hence, our expected output is computed manually.
    B, C_in, L = x.shape
    C_out = out_channels
    # Calculate output length using the kernel's formula:
    out_size = (L - 0 * (kernel_size - 1) - 1) // 1 + 1
    expected = torch.empty(B, C_out, out_size, device='cuda', dtype=torch.float32)
    for b in range(B):
        for oc in range(C_out):
            for o in range(out_size):
                s = 0.0
                start_pos = o  # stride=1, dilation=0: every index is start_pos + k*0 = start_pos.
                for ic in range(in_channels):
                    for k in range(kernel_size):
                        # All kernel elements use the same input position if dilation == 0.
                        if start_pos < L:
                            s += x[b, ic, start_pos] * weight[oc, ic, k]
                s += bias[oc].item()
                expected[b, oc, o] = s
                
    # The output from the kernel should differ from the true convolution result (which would use dilation>=1)
    # Therefore, if the results match the manual emulation, it reflects that dilation=0 is not handled as in standard conv.
    # Here we simply check that the kernel produces the "wrong" result.
    if torch.allclose(output, expected, atol=1e-5):
        pytest.fail("Kernel did not error out or warn when using dilation==0; it produced a result "
                    "that may be inconsistent with a standard convolution operation.")

# Issue 3: Test for non-contiguous tensor failure.
def test_non_contiguous_issue():
    B, in_channels, length = 4, 3, 32
    out_channels, kernel_size = 8, 3
    x = torch.randn(B, in_channels, length, device='cuda', dtype=torch.float32).transpose(1, 2)
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    conv1d_kernel = build_kernel()
    # Our host code checks for contiguity so non-contiguous input should trigger a TORCH_CHECK.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        conv1d_kernel.forward(x, weight, bias, stride=1, dilation=1)

# Issue 4: Test fixed stream configuration behavior with small batch size.
def test_fixed_streams_issue():
    # When B is very small (e.g., 1), the stream splitting code may cause unexpected behavior.
    B, in_channels, length = 1, 3, 32
    out_channels, kernel_size = 8, 3
    x = torch.randn(B, in_channels, length, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    conv1d_kernel = build_kernel()
    output = conv1d_kernel.forward(x, weight, bias, stride=1, dilation=1)
    
    # Also compute expected output using PyTorch's nn.Conv1d
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=True)
    # Manually set conv parameters to match our inputs
    conv.weight.data = weight.clone()
    conv.bias.data = bias.clone()
    conv = conv.to('cuda')
    expected = conv(x.contiguous())
    
    # Although the fixed stream ordering should not affect numerical correctness,
    # if there is an issue due to the stream splitting for a small batch size, 
    # the outputs will differ.
    torch.cuda.synchronize()
    assert torch.allclose(output, expected, atol=1e-4), "Kernel output does not match expected result when batch size is small."

