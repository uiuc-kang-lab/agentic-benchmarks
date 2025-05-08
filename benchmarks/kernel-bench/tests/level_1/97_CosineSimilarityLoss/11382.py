
import pytest
import torch
from torch.utils.cpp_extension import load
import math

# Helper function to build the CUDA extension
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Test 1: Duplicate shared memory declaration issue.
# With a small D, we create a simple case where predictions and targets are identical.
# The expected cosine similarity should be exactly 1 for each sample, hence loss = 0.
def test_duplicate_shared_memory():
    cuda_module = build_kernel()
    batch_size = 32
    D = 16  # small dimension to force shared memory usage and warp reduction overlap
    # Create inputs where predictions and targets are exactly the same.
    inputs = torch.randn(batch_size, D, device='cuda', dtype=torch.float32)
    # cosine similarity for identical vectors is 1 => loss = 0 per sample
    loss_kernel = cuda_module.forward(inputs, inputs)
    torch.cuda.synchronize()
    # Due to duplicate shared memory region the kernel may accumulate non-zero loss.
    # We expect the correct loss to be 0.
    expected_loss = torch.tensor(0.0, device='cuda')
    assert not torch.allclose(loss_kernel, expected_loss, atol=1e-5), (
        "Test should trigger an error in the shared memory usage causing an incorrect loss."
    )

# Test 2: Missing offset for warp-level reduction arrays.
# Choose D that is not a multiple of the warp size (32) to stress the reduction.
def test_missing_offset_reduction():
    cuda_module = build_kernel()
    batch_size = 64
    D = 37  # not divisible by 32; forces uneven loading and reduction
    # Create predictions and targets with a known cosine similarity.
    # For example, use normalized vectors.
    predictions = torch.randn(batch_size, D, device='cuda', dtype=torch.float32)
    predictions = predictions / predictions.norm(dim=1, keepdim=True)
    targets = torch.randn(batch_size, D, device='cuda', dtype=torch.float32)
    targets = targets / targets.norm(dim=1, keepdim=True)
    
    # Compute expected loss using PyTorch built-in functions.
    cos_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
    expected_loss = torch.mean(1 - cos_sim)
    
    loss_kernel = cuda_module.forward(predictions, targets)
    torch.cuda.synchronize()
    # Because of the missing offset the kernel should yield a different loss.
    assert not torch.allclose(loss_kernel, expected_loss, atol=1e-5), (
        "Kernel loss is unexpectedly correct; the missing offset in shared memory for reduction should trigger an error."
    )

# Test 3: Unsafe shared memory allocation size.
# This test passes a very large D (if possible) to trigger potential out-of-bound shared memory usage.
# This test expects the kernel launch to fail (i.e. raise an error).
def test_shared_memory_overflow():
    cuda_module = build_kernel()
    batch_size = 1
    # Choose a large D such that 2*D floats plus warp reduction space exceed typical shared memory limits.
    # Most GPUs provide 48 KB to 96 KB of dynamic shared memory per block.
    # Here, D is chosen to be high (e.g., 10000) so that 2*D*4 bytes is 80 KB, likely exceeding half the GPU limit.
    D = 10000  
    predictions = torch.randn(batch_size, D, device='cuda', dtype=torch.float32)
    targets = torch.randn(batch_size, D, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # The kernel launch should raise an error due to insufficient shared memory.
        loss_kernel = cuda_module.forward(predictions, targets)
        torch.cuda.synchronize()

if __name__ == "__main__":
    pytest.main([__file__])
