
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build and load the CUDA extension (kernel.cu)
def build_kernel():
    # Make sure we compile with at least C++17 support (required for if constexpr).
    cuda_module = load(
        name="triplet_loss_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Non-contiguous tensors
def test_non_contiguous():
    module = build_kernel()
    batch_size = 8
    feat_size = 4096
    # Create contiguous inputs.
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    # Make them non-contiguous by transposing a 2D view (only works for square matrices)
    # To simulate non-contiguity, create a tensor, unsqueeze and then view a transpose.
    anchor_nc = anchor[:, :].t().t()  # a no-op transpose trick that often produces non-contiguous result
    positive_nc = positive[:, :].t().t()
    negative_nc = negative[:, :].t().t()
    # Assert that they are non-contiguous
    assert not anchor_nc.is_contiguous(), "Anchor tensor should be non-contiguous for this test case"
    
    # Calling the kernel with non-contiguous tensors should produce an incorrect result or raise an error.
    with pytest.raises(RuntimeError):
        # The kernel assumes contiguous input, so this may lead to an out-of-bound index or wrong results.
        loss = module.forward(anchor_nc, positive_nc, negative_nc)
        torch.cuda.synchronize()
    
# Test case 2: Mismatched feature dimensions
def test_mismatched_dimensions():
    module = build_kernel()
    batch_size = 8
    feat_size_anchor = 4096
    feat_size_positive = 4096
    feat_size_negative = 2048  # mismatched!
    
    anchor = torch.randn(batch_size, feat_size_anchor, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size_positive, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size_negative, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError):
        # Expect failure because the kernel does not check dimension compatibility;
        # this should lead to an illegal memory access.
        loss = module.forward(anchor, positive, negative)
        torch.cuda.synchronize()

# Test case 3: Small effective block (simulate warp-level reduction issues)
def test_small_block():
    module = build_kernel()
    # Use a batch size of 1 and a very small feature dimension so that
    # the vectorized loop hardly does any work and only a few threads are active.
    batch_size = 1
    feat_size = 1   # so that only a subset of threads perform any work
    margin = 1.0
    
    # Create inputs with a single feature.
    # In this scenario, most threads will not participate in the computation and the fixed
    # __shfl_down_sync mask (0xffffffff) might produce unexpected behavior if the warp is not full.
    anchor = torch.tensor([[2.0]], device="cuda", dtype=torch.float32)
    positive = torch.tensor([[1.0]], device="cuda", dtype=torch.float32)
    negative = torch.tensor([[0.0]], device="cuda", dtype=torch.float32)
    
    # Here we are not expecting a correct loss value because of the warp reduction issue.
    # Instead, we check that the kernel either raises an error or produces a result that violates
    # the expected triplet loss computation: loss = max(0, sqrt((a-p)^2) - sqrt((a-n)^2) + margin)
    # The expected loss for these inputs would be max(0, 1.0 - 2.0 + 1.0) = 0.
    loss = module.forward(anchor, positive, negative)
    torch.cuda.synchronize()
    # If the warp-level shuffle reduction misbehaves, the loss may become nonzero.
    assert torch.abs(loss.item()) < 1e-6, f"Due to improper warp reduction, expected loss 0 but got {loss.item()}"
