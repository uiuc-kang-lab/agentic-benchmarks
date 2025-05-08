
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build the CUDA kernel extension
def build_kernel():
    cuda_module = load(
        name="argmin_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Triggering shared memory misalignment error.
# Using a data type where sizeof(scalar_t) != sizeof(int) (i.e. double)
# may yield incorrect results as the reduction over indices from shared memory
# could be corrupted.
def test_shared_memory_misalignment():
    cuda_module = build_kernel()
    # Create a double tensor (64-bit floating point).
    x = torch.randn(10, 20, device="cuda", dtype=torch.double)
    # We choose dim=1 reduction.
    out_cuda = cuda_module.forward(x, 1)
    # Compute reference using torch.argmin (which works correctly regardless
    # of contiguity or shared memory issues).
    out_ref = torch.argmin(x, dim=1)
    # This test is expected to fail (i.e. the results will differ) if the shared
    # memory misalignment issue is present.
    assert not torch.equal(out_cuda, out_ref), (
        "Test for shared memory misalignment did not trigger the issue. "
        "Kernel output matched reference output unexpectedly."
    )

# Test case 2: Using a non-contiguous input tensor.
# The kernel’s pointer arithmetic assumes contiguity, so passing a non‐contiguous tensor
# should result in an incorrect index result.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create a contiguous tensor and then transpose to make it non-contiguous.
    x = torch.randn(8, 16, 32, device="cuda")
    x_noncontig = x.transpose(1, 2)  # Make the tensor non-contiguous.
    # Use reduction along dim=1 (which is perturbed by the transpose).
    out_cuda = cuda_module.forward(x_noncontig, 1)
    out_ref = torch.argmin(x_noncontig, dim=1)
    # Expect that the kernel's result is wrong due to non-contiguous memory layout.
    assert not torch.equal(out_cuda, out_ref), (
        "Test for non-contiguous input did not trigger the issue. "
        "Kernel output matched reference output unexpectedly."
    )

# Test case 3: Reduction over an empty dimension.
# The kernel does not check for reduction dimension size == 0. In such cases,
# torch.argmin should raise an error. We expect our kernel to misbehave (or crash).
def test_empty_reduction_dim():
    cuda_module = build_kernel()
    # Create an input tensor with an empty reduction dimension.
    # For example, dim=1 is empty.
    x = torch.randn(4, 0, 16, device="cuda")
    with pytest.raises(RuntimeError):
        # Expect the kernel call to raise an exception (or produce an error)
        # because no valid reduction exists.
        cuda_module.forward(x, 1)
