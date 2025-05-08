
import torch
import pytest
from torch.utils.cpp_extension import load
import os
import shutil

def build_kernel():
    # Build the CUDA extension from kernel.cu.
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

def triplet_loss_reference(anchor, positive, negative, margin):
    # Compute triplet margin loss per sample with Euclidean distance
    # and then return the mean.
    diff_pos = anchor - positive
    diff_neg = anchor - negative
    d_pos = torch.sqrt(torch.sum(diff_pos * diff_pos, dim=1))
    d_neg = torch.sqrt(torch.sum(diff_neg * diff_neg, dim=1))
    loss = torch.clamp(d_pos - d_neg + margin, min=0.0)
    return loss.mean()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unqualified_max_float32():
    # This test verifies the kernel behavior with float32 inputs.
    # If the unqualified max is not correctly handled in device code,
    # the computed loss may differ from the PyTorch reference.
    batch_size = 128
    feat_size = 4096

    torch.manual_seed(42)
    anchor = torch.randn(batch_size, feat_size, device='cuda', dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size, device='cuda', dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size, device='cuda', dtype=torch.float32)
    margin = 1.0

    # Build the kernel module
    cuda_module = build_kernel()
    # Run the CUDA kernel implementation via the exported "forward" function.
    kernel_loss = cuda_module.forward(anchor, positive, negative, margin)
    # Compute the reference loss using PyTorch
    ref_loss = triplet_loss_reference(anchor, positive, negative, margin)

    # Tolerance: if the max issue causes a wrong clamp, the outputs will differ.
    # The test fails if the kernel output is not close to the reference.
    assert torch.allclose(kernel_loss, ref_loss, atol=1e-4), (
        f"Float32 kernel loss {kernel_loss} does not match reference {ref_loss}. "
        "This may be due to an issue using an unqualified max in device code."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared_memory_allocation_double():
    # This test forces the kernel to be instantiated with double precision.
    # Since the shared memory size was always computed with sizeof(float),
    # using double (64-bit) inputs will likely produce incorrect results.
    batch_size = 64
    feat_size = 1024

    torch.manual_seed(123)
    anchor = torch.randn(batch_size, feat_size, device='cuda', dtype=torch.float64)
    positive = torch.randn(batch_size, feat_size, device='cuda', dtype=torch.float64)
    negative = torch.randn(batch_size, feat_size, device='cuda', dtype=torch.float64)
    margin = 1.0

    cuda_module = build_kernel()
    kernel_loss = cuda_module.forward(anchor, positive, negative, margin)
    ref_loss = triplet_loss_reference(anchor, positive, negative, margin)

    # If the shared memory allocation size is not adjusted for double type,
    # the computed loss will deviate from the reference result.
    assert not torch.allclose(kernel_loss, ref_loss, atol=1e-4), (
        "Double precision test did not trigger an error. "
        "This indicates the shared memory allocation size issue may not be detected."
    )

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
