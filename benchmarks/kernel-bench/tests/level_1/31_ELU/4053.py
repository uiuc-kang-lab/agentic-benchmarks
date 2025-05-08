
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to build the CUDA extension
def build_kernel():
    # Force rebuild to pick up any changes in kernel.cu
    module = load(
        name="efficient_elu_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Test that passing a tensor with a type different than float32 raises an error.
def test_input_tensor_dtype():
    my_module = build_kernel()
    # Create a tensor of type float64. The kernel expects float32.
    x_double = torch.randn(1024, dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError) as excinfo:
        # This should fail due to CHECK_CUDA and/or type incompatibility.
        my_module.forward(x_double, 1.0)
    assert "must be a CUDA tensor" not in str(excinfo.value)  # Ensure it is a dtype issue

# Issue 2: Test that a non-contiguous tensor input raises an error.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then make a non-contiguous view by transposing.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    x_non_contiguous = x.t()  # Transpose typically makes tensor non-contiguous.
    assert not x_non_contiguous.is_contiguous(), "Tensor is unexpectedly contiguous."
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(x_non_contiguous, 1.0)
    assert "must be contiguous" in str(excinfo.value)

# Issue 3: Test that kernel launch errors (if any) are caught by forcing an error scenario.
def test_kernel_launch_error():
    my_module = build_kernel()
    # A possible way to force an error is to create an empty tensor with 0 elements.
    # While a zero-size kernel launch is generally OK in CUDA,
    # we can force an error by deliberately providing an incorrect alpha value type.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Pass an invalid alpha (e.g., a tensor instead of a float) to provoke an error.
    with pytest.raises(RuntimeError):
        my_module.forward(x, torch.tensor(1.0, device="cuda"))
