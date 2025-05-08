
import os
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Non-contiguous tensor (Issue 1)
def test_non_contiguous_tensor():
    # Create a contiguous tensor and then make it non-contiguous by transposing
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Now non-contiguous
    # Compute expected result using PyTorch
    expected = F.hardsigmoid(x_noncontig)
    # Call the CUDA kernel regardless; note that the kernel does not check for contiguity.
    module = build_kernel()
    # This is expected to produce an incorrect result or crash due to misinterpreted memory layout.
    result = module.forward(x_noncontig)
    # Compare the result with the expected, expecting a significant discrepancy.
    max_diff = (result - expected).abs().max()
    assert max_diff > 1e-3, f"Non-contiguous tensor input did not trigger an issue (max diff={max_diff})."

# Test 2: Misaligned tensor (Issue 1)
def test_misaligned_tensor():
    # Create a tensor with additional storage padding so that a slice starting at a non-zero offset
    # is mis-aligned relative to the vectorized load requirements.
    base = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Create a misaligned slice by skipping the first element.
    x_misaligned = base[1:]
    # Ensure the tensor is contiguous (it is, but its data pointer is offset)
    assert x_misaligned.is_contiguous()
    expected = F.hardsigmoid(x_misaligned)
    module = build_kernel()
    result = module.forward(x_misaligned)
    max_diff = (result - expected).abs().max()
    assert max_diff > 1e-3, f"Misaligned tensor input did not trigger an issue (max diff={max_diff})."

# Test 3: Unsupported dtype (Issue 3)
def test_unsupported_dtype():
    # Create a half precision tensor. The kernel dispatch only supports float32 and float64.
    x_half = torch.randn(128, device="cuda", dtype=torch.float16)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel launch to fail or throw an error due to unhandled dtype.
        module.forward(x_half)
