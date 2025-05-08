
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="hardsigmoid_ext",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel does not support half precision.
def test_half_precision_not_supported():
    # Create a half-precision tensor.
    x = torch.randn(1024, device='cuda', dtype=torch.float16)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        # This should trigger an error because the kernel was not designed
        # for half precision (it will likely choose the double path or fail to compile/run).
        y = kernel_module.forward(x)
        torch.cuda.synchronize()
    assert "half" in str(excinfo.value).lower(), f"Expected error message about half precision, got: {excinfo.value}"

# Issue 2: Using unqualified fma instead of type-specialized version.
# This test checks numerical inconsistency for float32 inputs.
def test_fma_numerical_accuracy():
    # For a predictable input, hardsigmoid should compute: out = clamp((x+3)/6, 0, 1)
    # We choose a value that is in the linear region so that fma precision matters.
    x = torch.tensor([0.5], device='cuda', dtype=torch.float32)
    expected = (x + 3.0) / 6.0
    kernel_module = build_kernel()
    y = kernel_module.forward(x)
    torch.cuda.synchronize()
    # Allow a tight tolerance in case a less appropriate fma is used.
    assert torch.allclose(y, expected, atol=1e-6), f"Numerical mismatch: expected {expected.item()}, got {y.item()}"

# Issue 3: Using generic min/max intrinsics may lead to incorrect clamping.
# This test applies values that should be clamped to 0 or 1.
def test_min_max_clamping():
    # Create inputs that are below 0 and above 1 after transformation.
    # For x such that (x+3)/6 < 0, choose x = -4; for (x+3)/6 > 1, choose x = 4.
    x = torch.tensor([-4.0, 4.0], device='cuda', dtype=torch.float32)
    # Expected: clamp((-4+3)/6, 0,1) = 0 and clamp((4+3)/6,0,1) = 1
    expected = torch.tensor([0.0, 1.0], device='cuda', dtype=torch.float32)
    kernel_module = build_kernel()
    y = kernel_module.forward(x)
    torch.cuda.synchronize()
    assert torch.allclose(y, expected, atol=1e-6), f"Clamping error: expected {expected}, got {y}"
