
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension module from kernel.cu.
def build_kernel():
    module = load(
        name="triplet_loss_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test case 1: Trigger the potential issue with the unqualified max()
# Here, we design the input so that the computed loss should be clamped to 0.
# For a valid triplet, if (sqrt(dist_pos) - sqrt(dist_neg) + margin) is negative,
# the loss must be 0. If max() is not working as expected, the negative value might be returned.
def test_max_function_issue():
    kernel_module = build_kernel()
    # Use one sample with a clear negative margin offset.
    # Let anchor and positive be identical (distance = 0) and negative be far (distance = 1).
    # With margin 0.5, the standard triplet margin loss should yield:
    #   loss = max(0, 0 - 1 + 0.5) = max(0, -0.5) = 0
    batch_size = 1
    feat_size = 4096
    anchor = torch.zeros((batch_size, feat_size), device="cuda", dtype=torch.float32)
    positive = torch.zeros((batch_size, feat_size), device="cuda", dtype=torch.float32)
    negative = torch.ones((batch_size, feat_size), device="cuda", dtype=torch.float32)
    margin = 0.5

    loss = kernel_module.forward(anchor, positive, negative, margin)
    torch.cuda.synchronize()
    # The expected loss is 0.
    assert torch.isclose(loss, torch.tensor(0.0, device="cuda", dtype=torch.float32), atol=1e-4), \
           f"max function issue: expected 0 loss, got {loss.item()}"

# Test case 2: Trigger the block reduction issue.
# If the feature size is very small (e.g., 1), then only a few threads do work,
# and the block-level reduction (which assumes blockDim.x is a multiple of 32) may incorporate uninitialized shared memory.
def test_block_reduction_issue():
    kernel_module = build_kernel()
    batch_size = 1
    # Use a feature size smaller than the warp size.
    feat_size = 1
    # For a simple case, set anchor and positive identical, negative with a nonzero value.
    anchor = torch.zeros((batch_size, feat_size), device="cuda", dtype=torch.float32)
    positive = torch.zeros((batch_size, feat_size), device="cuda", dtype=torch.float32)
    negative = torch.ones((batch_size, feat_size), device="cuda", dtype=torch.float32)
    margin = 1.0
    # Expected loss = max(0, 0 - 1 + 1) = 0.
    expected_loss = 0.0

    loss = kernel_module.forward(anchor, positive, negative, margin)
    torch.cuda.synchronize()
    # Because of the reduction bug, the result may not match the host-computed expected_loss.
    # We purposely fail the test if the kernel returns the correct result.
    if torch.isclose(loss, torch.tensor(expected_loss, device="cuda", dtype=torch.float32), atol=1e-4):
        pytest.fail(f"Block-level reduction issue not detected: expected a mis-reduction when feat_size is small, but got {loss.item()}")
    
# Test case 3: Trigger the issue of mismatched feature dimensions.
# The kernel does not validate that anchor, positive, and negative have the same feature dimensionality.
def test_shape_mismatch_issue():
    kernel_module = build_kernel()
    batch_size = 4
    feat_size_anchor = 100
    feat_size_positive = 101   # Deliberate mismatch.
    feat_size_negative = 100

    anchor = torch.randn((batch_size, feat_size_anchor), device="cuda", dtype=torch.float32)
    positive = torch.randn((batch_size, feat_size_positive), device="cuda", dtype=torch.float32)
    negative = torch.randn((batch_size, feat_size_negative), device="cuda", dtype=torch.float32)
    margin = 1.0

    with pytest.raises(RuntimeError):
        # This should lead to an out-of-bounds read in the kernel.
        loss = kernel_module.forward(anchor, positive, negative, margin)
        torch.cuda.synchronize()
