
import torch
import pytest
from torch.utils.cpp_extension import load

# This function builds the CUDA kernel from kernel.cu.
def build_kernel():
    return load(
        name="custom_reverse_cumsum",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# A reference implementation of reverse cumulative sum matching the intended PyTorch behavior.
def reference_reverse_cumsum(x, dim):
    # reversal along dim, cumsum along that reversed dim, then reverse back.
    return torch.cumsum(x.flip(dim), dim=dim).flip(dim)

# Issue 1: the cumulative sum dimension is assumed to be the last (innermost) dimension.
# This test uses dim=0 on a 2D tensor so that the kernel’s indexing arithmetic is invalid.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_issue_non_last_dimension():
    kernel = build_kernel()
    # Create a 2D tensor where dim=0 is not contiguous along the fast axis.
    x = torch.randn(128, 4000, device="cuda", dtype=torch.float32).contiguous()
    # Use reverse cumsum along the 0-th dimension.
    out_kernel = kernel.forward(x, 0)
    out_ref = reference_reverse_cumsum(x, 0)
    # Expect a significant difference because of the wrong indexing.
    diff = (out_kernel - out_ref).abs().max().item()
    assert diff > 1e-3, f"Test did not trigger the issue on non-last dimension (max diff={diff})"

# Issue 2: the kernel assumes the tensor is 2D (or that the target dimension splits the flattened array evenly).
# This test uses a 3D tensor and performs reverse cumsum along a middle dimension.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_issue_higher_dimensional_tensor():
    kernel = build_kernel()
    # Create a 3D tensor; the middle dimension is not the innermost contiguous dimension.
    x = torch.randn(4, 5, 6, device="cuda", dtype=torch.float32).contiguous()
    out_kernel = kernel.forward(x, 1)
    out_ref = reference_reverse_cumsum(x, 1)
    diff = (out_kernel - out_ref).abs().max().item()
    assert diff > 1e-3, f"Test did not trigger the issue on higher-dimensional input (max diff={diff})"

# Issue 3: the use of __shfl_sync assumes warp alignment.
# This test uses a tensor whose size along the reversed dimension is not a multiple of 32,
# so that threads within a warp may not be working on data from the same slice.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_issue_non_multiple_warp():
    kernel = build_kernel()
    # Choose a tensor with a dimension size that is not a multiple of 32.
    x = torch.randn(64, 33, device="cuda", dtype=torch.float32).contiguous()
    out_kernel = kernel.forward(x, 1)
    out_ref = reference_reverse_cumsum(x, 1)
    diff = (out_kernel - out_ref).abs().max().item()
    assert diff > 1e-3, f"Test did not trigger the warp shuffle issue (max diff={diff})"
