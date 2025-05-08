
import pytest
import torch
import math
from torch.utils.cpp_extension import load
import os

# Build the kernel module from kernel.cu in the current directory.
def build_kernel():
    module = load(
        name="frobenius_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test 1: Trigger mis‐alignment issue
# This test creates an input tensor with an offset into a larger storage,
# which might break the expected 16-byte alignment required for safe float4 loads.
def test_alignment_issue():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create a tensor with extra element then slice from an offset.
    base = torch.randn(1025, device='cuda', dtype=torch.float32)
    # Take a sub-tensor starting at offset 1. This sub-tensor is still contiguous 
    # but its data pointer may not be 16-byte aligned.
    input_tensor = base.narrow(0, 1, 1024)
    mod = build_kernel()
    try:
        output = mod.forward(input_tensor)
    except Exception as e:
        pytest.fail("Kernel failed with misaligned input: " + str(e))
    # We simply check that output is computed (even though the behavior is undefined).
    assert output.shape == input_tensor.shape

# Test 2: Trigger division-by-zero situation.
# By passing a tensor that contains all zeros, the computed norm will be zero,
# causing an invalid division.
def test_division_by_zero():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    input_tensor = torch.zeros(512, device='cuda', dtype=torch.float32)
    mod = build_kernel()
    output = mod.forward(input_tensor)
    # When dividing by zero, we expect the output to contain NaNs or Infs.
    if not (torch.isnan(output).any() or torch.isinf(output).any()):
        pytest.fail("Expected NaNs or Infs due to division by zero, but got a valid tensor.")

# Test 3: Test type checking, using non-float32 type.
def test_input_tensor_type():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    input_tensor = torch.randn(256, device='cuda', dtype=torch.float64)
    mod = build_kernel()
    with pytest.raises(RuntimeError, match="Input must be float32"):
        mod.forward(input_tensor)

# Test 4: Test non-contiguous tensor.
def test_non_contiguous_tensor():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create a contiguous tensor then transpose so it becomes non-contiguous.
    input_tensor = torch.randn(32, 64, device='cuda', dtype=torch.float32).t()
    mod = build_kernel()
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        mod.forward(input_tensor)

# Test 5: Test a tensor with less than 4 elements.
# This will force the kernel to launch with 0 blocks for the vectorized part,
# and the tail processing on host will be used.
def test_small_tensor():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Only 3 elements (not divisible by 4)
    # Use non-zero values so that norm is not zero.
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda', dtype=torch.float32)
    mod = build_kernel()
    output = mod.forward(input_tensor)
    # Compute expected normalized result on host.
    norm = math.sqrt((1.0**2 + 2.0**2 + 3.0**2))
    expected = input_tensor / norm
    if not torch.allclose(output, expected, atol=1e-5):
        pytest.fail("Output for small tensor does not match expected normalized values.")
