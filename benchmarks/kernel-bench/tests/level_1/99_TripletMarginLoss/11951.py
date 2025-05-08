
import math
import torch
import pytest
from torch.utils.cpp_extension import load

# Build and load the kernel extension.
def build_kernel():
    # We compile the kernel from kernel.cu.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def reference_triplet_margin_loss(anchor, positive, negative, margin):
    # Compute loss per sample: loss = max( sqrt(sum((a-p)^2)) - sqrt(sum((a-n)^2)) + margin, 0 )
    batch_size = anchor.size(0)
    losses = []
    for b in range(batch_size):
        a = anchor[b]
        p = positive[b]
        n = negative[b]
        dist_pos = torch.norm(a - p, p=2)
        dist_neg = torch.norm(a - n, p=2)
        loss = dist_pos - dist_neg + margin
        losses.append(max(0.0, loss.item()))
    # Return mean loss as computed by the kernel.
    return sum(losses) / len(losses)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_indexing():
    """
    This test is designed to trigger the incorrect base indexing in the kernel.
    We use a feature size that is much smaller than the default threads per block.
    The kernel will read the wrong feature elements and compute a loss different from the reference.
    """
    cuda_module = build_kernel()
    batch_size = 8
    feat_size = 10  # small feature dimension to magnify the indexing offset
    margin = 1.0

    # Create input tensors on the GPU.
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)

    loss_kernel = cuda_module.forward(anchor, positive, negative, margin)
    loss_ref = reference_triplet_margin_loss(anchor.cpu(), positive.cpu(), negative.cpu(), margin)

    # Because of the indexing error, the kernel loss should differ significantly from the reference.
    assert not math.isclose(loss_kernel.item(), loss_ref, rel_tol=1e-3), (
        f"Kernel loss ({loss_kernel.item()}) unexpectedly matches the reference loss ({loss_ref}) despite indexing error."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_divisible_warp():
    """
    This test is designed to trigger the assumption that the block dimension is a multiple of 32.
    We use a feature size that forces only a non-divisible number of threads (i.e. active threads)
    to perform the reduction. In such cases, parts of the shared memory used for warp reduction
    remain uninitialized leading to an incorrect final loss.
    """
    cuda_module = build_kernel()
    batch_size = 8
    feat_size = 33  # deliberately choose a size that will lead to a partial warp being active.
    margin = 1.0

    # Create input tensors on the GPU.
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)

    loss_kernel = cuda_module.forward(anchor, positive, negative, margin)
    loss_ref = reference_triplet_margin_loss(anchor.cpu(), positive.cpu(), negative.cpu(), margin)

    # Because of the shared memory reduction issue, the kernel loss should differ from the reference.
    assert not math.isclose(loss_kernel.item(), loss_ref, rel_tol=1e-3), (
        f"Kernel loss ({loss_kernel.item()}) unexpectedly matches the reference loss ({loss_ref}) despite shared memory reduction issue."
    )
