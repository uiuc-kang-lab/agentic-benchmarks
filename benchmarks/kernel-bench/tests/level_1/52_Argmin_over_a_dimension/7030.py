
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build/load the CUDA extension from kernel.cu in the current directory.
def build_kernel():
    cuda_module = load(
        name="argmin_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    mod = build_kernel()
    return mod

# Issue 1: Non-contiguous memory layout.
# This test intentionally creates a non-contiguous tensor using transpose.
def test_non_contiguous_tensor(cuda_module):
    # Create a contiguous tensor and then make it non-contiguous.
    x = torch.randn(16, 32, 64, device="cuda")
    x_nc = x.transpose(1, 2)  # This makes x_nc non-contiguous.
    # Use PyTorch armin as reference.
    ref = torch.argmin(x_nc, dim=1)
    # Call the custom kernel.
    try:
        out = cuda_module.forward(x_nc, 1)
    except RuntimeError as e:
        pytest.fail(f"Kernel raised an unexpected RuntimeError for non-contiguous input: {e}")
    # They may differ due to misinterpreting non-contiguous layout.
    # We expect the kernel output to be different from the reference.
    if torch.equal(ref, out):
        pytest.fail("Kernel did not trigger error for non-contiguous input; it returned correct result unexpectedly.")
    else:
        # The test case triggers the issue.
        assert True

# Issue 2: Misaligned memory accesses from __ldg.
# We create a tensor view that may produce a misaligned pointer.
def test_misaligned_tensor(cuda_module):
    # Create a contiguous tensor.
    x = torch.randn(17, 33, 65, device="cuda")
    # Slicing on the first column can produce a tensor with an offset.
    # This view is still contiguous as a sub-tensor, but the data pointer
    # might not be aligned to 128-bit boundaries required by __ldg.
    x_misaligned = x[:, 1:]
    ref = torch.argmin(x_misaligned, dim=0)
    # We choose a reduction dimension that is present in our data layout.
    try:
        out = cuda_module.forward(x_misaligned, 0)
    except RuntimeError as e:
        pytest.fail(f"Kernel raised an unexpected RuntimeError for misaligned input: {e}")
    # With misalignment, the kernel may read data incorrectly.
    if torch.equal(ref, out):
        pytest.fail("Kernel returned correct result for misaligned input unexpectedly.")
    else:
        # The test case triggers the misalignment issue.
        assert True

# Issue 3: Empty reduction dimension.
# The kernel does not handle the case when the reduction dimension size is zero.
def test_empty_reduction_dim(cuda_module):
    # Create a tensor with an empty reduction dim.
    x = torch.randn(10, 0, 20, device="cuda")
    with pytest.raises(RuntimeError):
        # Depending on how the kernel is integrated, we may get an error from the kernel launch.
        _ = cuda_module.forward(x, 1)

# Issue 4: Sequential reduction may be inefficient for large K.
# While not an error per se, we can trigger a test case where K is very large.
def test_large_reduction_dim(cuda_module):
    # Create a tensor where the reduction dimension (dim=2) is very large.
    # We use a moderately large size to avoid running out of time/resources in the test.
    x = torch.randn(4, 4, 16384, device="cuda")
    ref = torch.argmin(x, dim=2)
    try:
        out = cuda_module.forward(x, 2)
    except RuntimeError as e:
        pytest.fail(f"Kernel raised an unexpected RuntimeError for large reduction dimension: {e}")
    # Due to sequential reduction, results may be computed correctly but performance is poor.
    # Here we only check that the result is different from the reference as the kernel's strategy
    # does not support parallel reduction (thus triggering the limitation).
    if torch.equal(ref, out):
        pytest.fail("Kernel returned correct result for large reduction dim unexpectedly, "
                    "which means it did not trigger the intended issue of sequential reduction inefficiency.")
    else:
        assert True
