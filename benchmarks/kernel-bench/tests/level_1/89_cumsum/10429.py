
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension module.
def build_kernel():
    cuda_module = load(
        name="cumsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that non-float32 tensors cause an error.
def test_non_float_tensor():
    kernel_mod = build_kernel()
    # Create a CUDA tensor with double precision
    x = torch.randn(16, 32, device="cuda", dtype=torch.double)
    # Expecting that the CHECK_CUDA and direct cast to float* cause an error:
    with pytest.raises(Exception):
        # This call should error because the kernel only works with float32.
        kernel_mod.forward(x, 1)

# Issue 2: Test that outputs are computed correctly even though shared memory is allocated but unused.
# (If shared memory were used incorrectly the output might be corrupted. This test verifies that the
# cumulative sum along the scanning dimension is computed as expected.)
def test_cumsum_correctness():
    kernel_mod = build_kernel()
    # Use a float32 tensor
    # For simplicity, consider a 2D tensor where we perform the cumulative sum along dim=1.
    x = torch.randn(8, 4000, device="cuda", dtype=torch.float32)
    # Compute reference result using PyTorch's built-in function.
    ref = torch.cumsum(x, dim=1)
    out = kernel_mod.forward(x, 1)
    torch.cuda.synchronize()
    # Allow small numerical differences.
    assert torch.allclose(out, ref, atol=1e-5), "Cumulative sum output mismatches reference."

# Issue 3: Test that non-contiguous input tensors are rejected.
def test_non_contiguous_tensor():
    kernel_mod = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous via transpose.
    x = torch.randn(16, 32, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # transpose makes it non-contiguous
    assert not x_noncontig.is_contiguous(), "Test setup error: tensor should be non-contiguous."
    # The kernel is expected to raise an error for non-contiguous inputs.
    with pytest.raises(Exception):
        kernel_mod.forward(x_noncontig, 0)
