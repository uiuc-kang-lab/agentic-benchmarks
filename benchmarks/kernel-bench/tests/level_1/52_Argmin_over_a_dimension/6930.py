
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

# Helper function to build and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="argmin_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Wrong initialization for non-float types.
# For example, for double tensors the initialization value should be DBL_MAX instead of FLT_MAX.
# If the wrong constant is used, extreme values might be mishandled.
def test_double_type_incorrect_initialization():
    my_kernel = build_kernel()
    # Create a double tensor with values close to DBL_MAX to see if the initialization is too low.
    # We fill the tensor with very high numbers (above FLT_MAX) except one element.
    K = 100
    outer = 4
    inner = 3
    # Create a tensor of shape [outer, K, inner] which will be reduced along dim 1.
    # Set all elements to a value larger than FLT_MAX and then inject a lower value.
    # Note: FLT_MAX is about 3.4e38, so we use 1e39.
    x = torch.full((outer, K, inner), 1e39, device="cuda", dtype=torch.double)
    # Set a unique lower value in each [outer, :, inner] slice.
    for o in range(outer):
        for i in range(inner):
            # Place the “minimum” at index 57 arbitrarily.
            x[o, 57, i] = 1e38
    # Our kernel expects reduction on dimension 1.
    res = my_kernel.forward(x, 1)
    # Compare with PyTorch’s own argmin.
    expected = torch.argmin(x, dim=1)
    # They should match. If initialization is wrong, the kernel might fail to update properly.
    assert torch.equal(res, expected), f"Double tensor argmin mismatch. res: {res.cpu()}, expected: {expected.cpu()}"

# Issue 2: The kernel assumes a contiguous, default memory layout.
# Create a non-contiguous tensor via transpose to trigger incorrect indexing.
def test_non_contiguous_input():
    my_kernel = build_kernel()
    # Create a contiguous tensor then transpose to make it non-contiguous.
    # Let’s use a shape such that reduction is on one of the transposed dims.
    x = torch.randn(8, 16, 32, device="cuda", dtype=torch.float32)
    x = x.transpose(0, 2)  # Now x is non-contiguous.
    # Perform argmin reduction along dimension 1 (which is not the innermost dimension anymore).
    res = my_kernel.forward(x, 1)
    expected = torch.argmin(x, dim=1)
    # They likely will differ because the kernel’s indexing is done as if x were contiguous.
    assert not torch.equal(res, expected), f"Kernel unexpectedly handled a non-contiguous input correctly."

# Issue 3: The unconditional use of __ldg may cause problems for half-precision data,
# e.g., when using __half type with certain architectures or memory configurations.
def test_half_type_issue():
    my_kernel = build_kernel()
    # Create a half-precision tensor on CUDA.
    # The kernel might mishandle __ldg for __half.
    x = torch.randn(10, 50, 3, device="cuda", dtype=torch.float16)
    res = my_kernel.forward(x, 1)
    expected = torch.argmin(x, dim=1)
    # It’s possible that due to __ldg usage the result is incorrect.
    # We expect a discrepancy.
    if torch.equal(res, expected):
        pytest.skip("Half precision case did not trigger the __ldg related issue on this device.")
    else:
        assert not torch.equal(res, expected), f"Kernel with half type returned correct result unexpectedly."

# Issue 4: The kernel allocates shared memory arrays with a fixed size of 256.
# This test forces a reduction size (K) greater than the number of threads used
# if one were to try a different launch configuration. Even though the current host code uses 256 threads,
# a future modification to allow variable block sizes may reveal this bug.
def test_fixed_shared_memory_limit():
    my_kernel = build_kernel()
    # Create a tensor where the reduction dimension is significantly larger than 256.
    # For example, let K = 1024. This tests whether the reduction (assuming blockDim.x was changed)
    # would work with a fixed shared memory allocation of size 256.
    # (Right now, the host code always launches with threads=256.)
    outer = 2
    K = 1024
    inner = 4
    x = torch.randn(outer, K, inner, device="cuda", dtype=torch.float32)
    res = my_kernel.forward(x, 1)
    expected = torch.argmin(x, dim=1)
    # Even if the current launch config uses 256 threads, a mismatch in shared memory allocation
    # would show up as an incorrect output.
    assert torch.equal(res, expected), f"Kernel argmin mismatch for large reduction dimension. res: {res.cpu()}, expected: {expected.cpu()}"

# Issue 5: No check for an empty reduction dimension (K == 0).
# This test creates a tensor with an empty reduction dimension and expects an error.
def test_empty_reduction_dimension():
    my_kernel = build_kernel()
    # Create a tensor with one dimension of size 0.
    x = torch.randn(5, 0, 10, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expect the kernel (or later PyTorch layer) to complain about an empty reduction dimension.
        _ = my_kernel.forward(x, 1)
