
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper to build the CUDA extension from kernel.cu
def build_kernel():
    module = load(
        name="test_triplet_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

@pytest.fixture(scope="module")
def kernel_module():
    # Make sure we build the module only once.
    return build_kernel()

def compute_triplet_loss(anchor, positive, negative, margin):
    """
    Compute the loss per sample using Euclidean distances and the triplet loss formula:
       loss = max(0, ||anchor - positive|| - ||anchor - negative|| + margin)
    Returns: a single scalar that is the mean over the batch.
    """
    dist_pos = torch.norm(anchor - positive, p=2, dim=1)
    dist_neg = torch.norm(anchor - negative, p=2, dim=1)
    loss = torch.clamp(dist_pos - dist_neg + margin, min=0.0)
    return loss.mean()

# Test case to trigger the device max() issue.
# Using a different floating point type could force the issue if the device overload is not found.
def test_max_function_issue(kernel_module):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Use double precision to see if the kernel compiled use of max causes any issue.
    batch_size = 16
    feat_size = 32  # Equal to warp size, so reduction uses only warp-level shuffles.
    margin = 1.0
    anchor = torch.randn(batch_size, feat_size, dtype=torch.double, device="cuda")
    positive = torch.randn(batch_size, feat_size, dtype=torch.double, device="cuda")
    negative = torch.randn(batch_size, feat_size, dtype=torch.double, device="cuda")
    # The kernel uses AT_DISPATCH_FLOATING_TYPES which includes double.
    # We compute the expected output using the same reduction.
    expected_loss = compute_triplet_loss(anchor, positive, negative, margin)
    # Run kernel:
    kernel_loss = kernel_module.forward(anchor, positive, negative, margin)
    torch.cuda.synchronize()
    # In case the wrong "max" function is used, the computation may be off.
    # We expect the kernel to produce nearly the same value as the reference.
    assert torch.allclose(kernel_loss, expected_loss, atol=1e-5), \
        f"Kernel loss ({kernel_loss.item()}) does not match expected ({expected_loss.item()})."

# Test case to trigger the shared memory reduction issue.
def test_shared_memory_reduction_issue(kernel_module):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Create an input where feat_size > warp_size to force the use of shared memory reduction.
    # We also choose a batch size such that a block might process threads from more than one sample.
    batch_size = 8
    feat_size = 64  # greater than warp_size (32) so that shared memory reduction path is used.
    margin = 0.5

    # Construct inputs with constant values so that manual computation is trivial.
    # For instance, let:
    # anchor = ones, positive = ones, negative = zeros.
    # Then, for each sample:
    #   dist_pos = ||ones - ones|| = 0, dist_neg = ||ones - zeros|| = sqrt(64) = 8,
    #   loss = max(0, 0 - 8 + margin) = 0.
    # The expected loss is 0.
    anchor = torch.ones(batch_size, feat_size, dtype=torch.float32, device="cuda")
    positive = torch.ones(batch_size, feat_size, dtype=torch.float32, device="cuda")
    negative = torch.zeros(batch_size, feat_size, dtype=torch.float32, device="cuda")

    expected_loss = compute_triplet_loss(anchor, positive, negative, margin)
    # Run kernel:
    kernel_loss = kernel_module.forward(anchor, positive, negative, margin)
    torch.cuda.synchronize()

    # Due to the shared memory reduction issue, the computed loss might be nonzero.
    assert not torch.allclose(kernel_loss, expected_loss, atol=1e-5), \
        f"Kernel loss ({kernel_loss.item()}) unexpectedly matches the correct computation ({expected_loss.item()}); shared memory reduction bug may be hidden."

# Additional test to check correctness when each block contains threads from a single sample.
def test_block_single_sample(kernel_module):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Force a scenario where each CUDA block is likely to handle features from one sample by using
    # a very large feature dimension and a batch size of 1.
    batch_size = 1
    feat_size = 512  # This should be processed in one block if blockDim.x is set high enough.
    margin = 1.0

    # Random input.
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    expected_loss = compute_triplet_loss(anchor, positive, negative, margin)
    kernel_loss = kernel_module.forward(anchor, positive, negative, margin)
    torch.cuda.synchronize()

    # In this contrived case, the reduction across one sample may work correctly.
    # However, if the shared memory reduction bug is present even then, the kernel loss will differ.
    # We test that in a single-sample and large feature case, at least some accuracy is observed.
    assert torch.allclose(kernel_loss, expected_loss, atol=1e-4), \
        f"Kernel loss ({kernel_loss.item()}) does not match expected ({expected_loss.item()}) in single-sample block scenario."
