
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to compile the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="balanced_triplet_loss_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to compute reference triplet margin loss.
def reference_triplet_loss(anchor, positive, negative, margin):
    # Computes per-sample Euclidean distances
    d_pos = (anchor - positive).norm(2, dim=1)
    d_neg = (anchor - negative).norm(2, dim=1)
    loss = torch.clamp(d_pos - d_neg + margin, min=0)
    return loss.mean()

# Test 1: Trigger issue with CPU tensors
def test_cpu_tensor_error():
    kernel_module = build_kernel()
    device = torch.device("cpu")
    batch_size = 4
    feat_size = 512
    # Create CPU tensors (kernel expects CUDA tensors)
    anchor = torch.randn(batch_size, feat_size, device=device)
    positive = torch.randn(batch_size, feat_size, device=device)
    negative = torch.randn(batch_size, feat_size, device=device)
    margin = 1.0
    with pytest.raises(Exception) as excinfo:
        loss = kernel_module.forward(anchor, positive, negative, margin)
    assert "CUDA" in str(excinfo.value)

# Test 2: Trigger issue by passing an unsupported tensor type (e.g. integer type)
def test_invalid_datatype_error():
    kernel_module = build_kernel()
    batch_size = 4
    feat_size = 512
    # Create integer tensors (kernel dispatches only floating types)
    anchor = torch.randint(0, 5, (batch_size, feat_size), device="cuda", dtype=torch.int32)
    positive = torch.randint(0, 5, (batch_size, feat_size), device="cuda", dtype=torch.int32)
    negative = torch.randint(0, 5, (batch_size, feat_size), device="cuda", dtype=torch.int32)
    margin = 1.0
    with pytest.raises(RuntimeError) as excinfo:
        loss = kernel_module.forward(anchor, positive, negative, margin)
    # Confirm the error message mentions unsupported type or dispatch failure.
    assert "AT_DISPATCH" in str(excinfo.value) or "not implemented" in str(excinfo.value)

# Test 3: Simulate a potential error arising from static shared memory size insufficiency and usage of unqualified max.
# We simulate this by invoking the kernel on an input configuration that would require more reduction warps.
def test_reduction_overflow():
    kernel_module = build_kernel()
    # Choose a feature size and batch size such that blocksPerSample becomes large.
    # With threads per block = 256, blocksPerSample = ceil(feat_size/256) should be >32.
    # For example, feat_size = 256 * 40 = 10240 ensures blocksPerSample=40 > 32.
    batch_size = 4
    feat_size = 256 * 40  # 10240 features
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    margin = 1.0
    # Though the mathematical formulation is valid,
    # the static shared memory arrays (size 32) in the kernel for inter-warp reduction will be overflowed.
    # The outcome is likely to deviate from the reference implementation.
    loss = kernel_module.forward(anchor, positive, negative, margin)
    ref_loss = reference_triplet_loss(anchor, positive, negative, margin)
    # This assertion is expected to fail if the shared memory overflow issue affects the result.
    # (If the kernel were correct, the results should match closely.)
    assert not torch.allclose(loss, ref_loss, atol=1e-3), (
        f"Kernel loss ({loss.item()}) unexpectedly matches reference loss ({ref_loss.item()}) despite potential shared memory issues."
    )
