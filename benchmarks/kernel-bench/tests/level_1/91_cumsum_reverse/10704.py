
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build/load the CUDA extension module
def build_kernel():
    # Assume kernel.cu is in the same directory as this test file.
    module = load(
        name="reverse_cumsum_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Helper function that calls the reverse_cumsum function in the CUDA module.
def reverse_cumsum_cuda(x, dim):
    mod = build_kernel()
    return mod.forward(x, dim)

# Issue 1 Test:
# When the size of the dimension (n) is not a multiple of 32,
# the kernel’s method for combining partial warp sums is incorrect.
def test_partial_warp_incorrect_sum():
    # Create a tensor with last dimension size not a multiple of 32, e.g., 33.
    batch_size = 128
    n = 33  # 33 is not a multiple of 32 so the last warp is partially filled.
    # Use a fixed seed for reproducibility.
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, n, device="cuda", dtype=torch.float32)

    # Expected result computed using PyTorch’s flip/cumsum/flip operations.
    expected = torch.cumsum(input_tensor.flip(1), dim=1).flip(1)

    # Run the custom CUDA kernel (it will be used because dim==last and n <= 1024)
    output = reverse_cumsum_cuda(input_tensor, 1)
    torch.cuda.synchronize()

    # Due to the bug the kernel output will be incorrect.
    # We assert that the maximum error is larger than some tolerance.
    max_diff = (output - expected).abs().max().item()
    assert max_diff > 1e-4, (
        f"Test failed to trigger issue 1: maximum difference ({max_diff}) "
        "is too small, but the kernel should be computing an incorrect result."
    )

# Issue 2 Test:
# The CUDA kernel is written only for the last dimension.
# For an operation requested along a non‐last dimension the kernel logic (if it were used)
# would be invalid. In our implementation the code falls back to PyTorch’s flip+cumsum+flip.
# We mimic a case where the kernel SHOULD be used but the dimension indexing is not the last.
def test_non_last_dimension_fallback():
    # Create a 2D tensor and request reverse cumulative sum along dim 0 (non-last).
    # This should trigger the fallback path.
    torch.manual_seed(0)
    input_tensor = torch.randn(64, 128, device="cuda", dtype=torch.float32)

    expected = torch.cumsum(input_tensor.flip(0), dim=0).flip(0)
    # Since dim != last dimension, the code falls back and does not run the CUDA kernel.
    # The output should exactly match the expected result.
    output = reverse_cumsum_cuda(input_tensor, 0)
    torch.cuda.synchronize()

    assert torch.allclose(output, expected, atol=1e-5), (
        "Fallback for non-last dimension did not produce the expected result."
    )

# Issue 3 Test:
# The kernel hardcodes a warp size of 32 by using __shfl_up_sync with mask 0xffffffff.
# If a device had a different warp size (or if warpSize should be queried dynamically),
# the hardcoded values could lead to incorrect results.
# While current hardware always uses a warp size of 32, we simulate the effect by
# comparing results for a tensor with length equal to 32 (a full warp) versus one with a mismatched size.
def test_hardcoded_warp_size():
    # Use a tensor with length = 32, which should be computed correctly.
    torch.manual_seed(0)
    batch_size = 128
    n_full = 32
    input_full = torch.randn(batch_size, n_full, device="cuda", dtype=torch.float32)
    expected_full = torch.cumsum(input_full.flip(1), dim=1).flip(1)
    output_full = reverse_cumsum_cuda(input_full, 1)
    torch.cuda.synchronize()
    assert torch.allclose(output_full, expected_full, atol=1e-5), (
        "Kernel produced incorrect result for full warp (n equal to warp size), "
        "which is unexpected if warp size were handled properly."
    )

    # Now, as an additional check, use a tensor with length slightly different from 32.
    # This is essentially reiterating issue 1.
    n_non_full = 31  # 31 < 32, the partial warp case.
    input_non_full = torch.randn(batch_size, n_non_full, device="cuda", dtype=torch.float32)
    expected_non_full = torch.cumsum(input_non_full.flip(1), dim=1).flip(1)
    output_non_full = reverse_cumsum_cuda(input_non_full, 1)
    torch.cuda.synchronize()
    max_diff = (output_non_full - expected_non_full).abs().max().item()
    assert max_diff > 1e-4, (
        "Kernel appears to handle non-full warp correctly even though it uses hardcoded warp sizes; "
        "this is unexpected and should be verified."
    )
