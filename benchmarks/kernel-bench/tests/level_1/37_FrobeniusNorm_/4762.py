
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="my_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1: No synchronization/error checking after kernel launches.
# We can expose this problem by using a very small kernel launch configuration
# that might force asynchronous behavior. Since the kernel does not check for errors,
# we will force a scenario where the computed norm is not properly synchronized.
# This test indirectly checks (by comparing with CPU result) that the result is off.
def test_kernel_synchronization_error():
    my_module = build_kernel()
    # Use a moderately sized tensor where kernel asynchrony could show up.
    N = 1 << 20  # 2^20 elements
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    # Manually call the forward function from our custom kernel.
    # We compare the result with a CPU computed norm normalization.
    # Due to lack of synchronization, the output might be erroneous.
    y = my_module.forward(x)
    # Recompute Frobenius norm on host.
    norm_val = torch.norm(x, p='fro')
    y_ref = x / norm_val
    # If the kernel had synchronized properly, the result should be close.
    # Here we expect a failure (mismatch) because the computed norm might be incorrect.
    # We take a loose tolerance so that even a small discrepancy triggers a failure.
    assert not torch.allclose(y, y_ref, atol=1e-3), (
        "The kernel appears to be synchronized correctly, but an asynchrony/err-check issue was expected."
    )

# Issue 2: No handling for division by zero.
# If the input is all zeros, the computed norm will be zero and division will yield NaNs or Infs.
def test_division_by_zero():
    my_module = build_kernel()
    x = torch.zeros((1024,), device='cuda', dtype=torch.float32)
    y = my_module.forward(x)
    # Check that the output contains non-finite numbers (NaNs or Infs)
    # because 0/0 is undefined.
    assert not torch.isfinite(y).all(), (
        "Expected non-finite values (NaN/Inf) when dividing by zero norm, but got finite values."
    )

# Issue 3: Kernel only supports contiguous float32 tensors.
# Test 3a: Pass a non-contiguous tensor.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(32, 64, device='cuda', dtype=torch.float32)
    x_non_contiguous = x.t()  # Transpose makes it non-contiguous.
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        _ = my_module.forward(x_non_contiguous)

# Test 3b: Pass a tensor with the wrong data type (e.g. float64)
def test_wrong_dtype():
    my_module = build_kernel()
    x = torch.randn(32, 64, device='cuda', dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Input must be float32"):
        _ = my_module.forward(x)
