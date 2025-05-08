
import torch
import pytest
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="test_mse_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return cuda_module

# Issue 1: Inflexible block size assumption.
#
# When the kernel is launched, it always uses BLOCK_SIZE (256) threads per block,
# and uses that same number in shared-memory indexing. If the problem size is very small,
# then many threads in the block are “inactive” (i.e. they do not have corresponding input elements)
# and do not write to every slot in the shared memory.
# That means the final reduction loop might sum uninitialized (garbage) values.
#
# We design a test case with very few total elements to trigger this problem.
def test_inflexible_block_size():
    # Create very small input tensors so that only a few elements are valid.
    # We choose 10 elements while the block size is fixed to 256.
    N = 10
    # Use a known constant tensor for easy verification.
    preds = torch.linspace(0, 1, steps=N, device="cuda", dtype=torch.float32)
    tgts = torch.linspace(0, 1, steps=N, device="cuda", dtype=torch.float32)
    # Expected result: mean((preds - tgts)^2) should be exactly zero.
    expected = torch.tensor(0.0, device="cuda", dtype=torch.float32)
    
    kernel = build_kernel()
    out = kernel.forward(preds, tgts)
    torch.cuda.synchronize()
    
    # If the kernel doesn't use an adaptive launch (and uses the assumed block size)
    # it might read garbage from shared memory and produce a nonzero error.
    assert math.isclose(out.item(), expected.item(), rel_tol=1e-5), \
        f"Issue 1 triggered: Expected {expected.item()}, got {out.item()}"

# Issue 2: Non-contiguous input handling.
#
# The kernel assumes that the underlying data is contiguous, but if a non‐contiguous version
# of the tensor is provided the flat indexing via data_ptr will lead to incorrect results.
def test_non_contiguous_input():
    # Create contiguous tensors
    N = 1024
    preds_full = torch.randn(N, device="cuda", dtype=torch.float32)
    tgts_full = torch.randn(N, device="cuda", dtype=torch.float32)
    # Make them non‐contiguous by creating a view with a stride (for example, by slicing with a step)
    # Note: [::2] will yield roughly half the elements but not contiguous.
    preds = preds_full[::2]
    tgts = tgts_full[::2]
    # Compute the expected MSE using PyTorch (which works correctly on non‐contiguous inputs)
    expected = torch.mean((preds - tgts) ** 2)

    kernel = build_kernel()
    out = kernel.forward(preds, tgts)
    torch.cuda.synchronize()
    
    # The kernel, however, does not check or adjust for non‐contiguity. The result is likely off.
    assert not torch.allclose(out, expected, atol=1e-5), \
        f"Issue 2 triggered: Expected a mismatch for non-contiguous inputs, but got {out} vs {expected}"

# Issue 3: Hard-coded warp reduction offsets assuming warp size of 32.
#
# If the input size is such that the kernel’s warp-level reduction gets invoked
# with a warp that does not have 32 active threads (e.g. a nearly empty warp), the
# hard-coded offsets (16, 8, 4, 2, 1) may cause incorrect accumulation.
def test_hard_coded_warp_size():
    # Choose an input size that leads to a final (partial) warp.
    # For instance, choose N such that the last warp in the block has only a few active threads.
    # Since BLOCK_SIZE is 256, there are 256/32 = 8 warps per block.
    # Force one block to be used and choose N so that the last warp is only partially filled.
    N = 256 - 3  # Last warp will have only 29 active threads instead of 32.
    # Use known values for predictability.
    preds = torch.full((N,), 2.0, device="cuda", dtype=torch.float32)
    tgts = torch.full((N,), 1.0, device="cuda", dtype=torch.float32)
    # Expected mse = mean((2 - 1)^2) = 1.
    expected = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    kernel = build_kernel()
    out = kernel.forward(preds, tgts)
    torch.cuda.synchronize()
    
    # Because of the hard-coded warp reduction offsets, the kernel may sum in extra garbage
    # from uninitialized shared memory leading to an error in the reduction.
    assert not torch.allclose(out, expected, atol=1e-5), \
        f"Issue 3 triggered: Expected a mismatch due to warp-size assumptions, got {out} vs {expected}"
