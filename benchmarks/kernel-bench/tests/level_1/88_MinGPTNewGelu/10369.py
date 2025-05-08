
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    cuda_module = load(
        name="gelu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that passing a non-float32 tensor triggers an error.
def test_dtype_check():
    my_module = build_kernel()
    # Create tensor with type float64
    x = torch.randn(2000, 2000, dtype=torch.float64, device="cuda", requires_grad=False)
    with pytest.raises(RuntimeError):
        # The CUDA kernel is only valid for float32,
        # so launching the kernel on a float64 tensor should cause an error.
        y = my_module.forward(x)
        torch.cuda.synchronize()

# Issue 2: Test misaligned memory due to sub-tensor slicing.
def test_alignment_error():
    my_module = build_kernel()
    # Create a larger tensor ensuring proper allocation.
    x_full = torch.randn(2000, 2001, dtype=torch.float32, device="cuda")
    # Slice along the second dimension to potentially create a misaligned view.
    # The slicing is done such that the underlying pointer no longer has 16-byte alignment.
    x = x_full[:, 1:]
    # Make sure the tensor is contiguous.
    x = x.contiguous()
    # The kernel assumes proper alignment due to reinterpret_cast to float4.
    # This test will check if the kernel misbehaves (by crashing or returning faulty results).
    y = my_module.forward(x)
    torch.cuda.synchronize()
    
    # Use reference PyTorch implementation of GELU.
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715
    # computing GELU in PyTorch, following the formula used in the kernel.
    inner = math.sqrt(2.0 / math.pi) * (x + coeff * x.pow(3))
    y_ref = 0.5 * x * (1.0 + torch.tanh(inner))
    # We expect the results to be close, if alignment is not an issue.
    # If the misalignment creates issues, this test may either crash or produce a noticeable error.
    assert torch.allclose(y, y_ref, atol=1e-5), "Kernel output does not match reference GELU computation due to potential misaligned memory access."

import math
