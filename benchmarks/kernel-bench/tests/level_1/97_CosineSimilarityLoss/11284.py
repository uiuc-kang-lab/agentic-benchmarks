
import pytest
import torch
from torch.utils.cpp_extension import load

# A helper function to build (or rebuild) the CUDA extension.
def build_kernel(extra_cuda_cflags=None):
    extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="cosine_loss_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger potential misaligned memory access.
# We create non‐contiguous tensors by slicing an originally contiguous tensor.
def test_misaligned_input():
    # Create a base tensor that is contiguous.
    batch_size = 128
    D = 4096
    # Create a larger tensor then slice a sub-tensor with an offset along dim=1.
    base_pred = torch.randn(batch_size, D + 1, device="cuda", dtype=torch.float32)
    base_targ = torch.randn(batch_size, D + 1, device="cuda", dtype=torch.float32)
    # Slice to force a potential misalignment (the underlying pointer offset may not be 16-byte aligned)
    predictions = base_pred[:, 1:].clone()  # clone to preserve the same storage layout but may lose 16-byte alignment
    targets = base_targ[:, 1:].clone()
    
    # Ensure the tensors are non-contiguous.
    assert not predictions.is_contiguous()
    assert not targets.is_contiguous()
    
    # Build the kernel module.
    ker = build_kernel()
    # Try invoking the kernel forward.
    # Depending on the device and allocation, misaligned loads could silently produce wrong results.
    # Here we compare with the PyTorch CPU (functional) implementation.
    loss_kernel = ker.forward(predictions, targets)
    # Compute loss on CPU (or via the standard PyTorch CUDA implementation)
    cosine_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
    loss_cpu = torch.mean(1.0 - cosine_sim)
    # The misalignment may lead to a result that is noticeably different from the expected one.
    # We trigger the issue if the difference is large.
    diff = (loss_kernel - loss_cpu).abs().item()
    assert diff > 1e-3, f"Misaligned load issue not triggered (diff={diff}). It is possible that the tensor was still aligned."

# Test case 2: Simulate a shared memory overflow situation.
# This test forces a launch with a large block size by rebuilding the extension with an extra flag that
# overrides the block size (if the kernel is modified to use such a macro). Since the current kernel hardcodes
# block_size=256, we simulate the scenario by forcing a block size greater than 1024 (i.e. > 32 warps).
def test_large_block_shared_memory():
    # Check if the device supports launching with a very large block size.
    # We simulate this scenario by defining a macro to override the block size in the kernel
    # (the kernel source must be modified to use BLOCK_SIZE macro if present).
    # For testing purpose, we assume such a modification is in place.
    extra_flags = ["-O3", "--use_fast_math", "-DBLOCK_SIZE_OVERRIDE=2048"]
    ker = build_kernel(extra_cuda_cflags=extra_flags)
    
    batch_size = 128  # still use one block per row
    D = 4096
    predictions = torch.randn(batch_size, D, device="cuda", dtype=torch.float32)
    targets = torch.randn(batch_size, D, device="cuda", dtype=torch.float32)
    
    # The forward function is expected to launch with block_size=2048 now (assuming the kernel honors BLOCK_SIZE_OVERRIDE).
    # With 2048 threads per block, there are 2048/32 = 64 warps per block, but only 32 shared memory slots, so
    # we expect an out-of-bounds shared memory write. This typically leads to a CUDA error.
    with pytest.raises(RuntimeError):
        loss = ker.forward(predictions, targets)
        torch.cuda.synchronize()

# Test case 3: Trigger inflexibility due to one-block-per-row assumption.
# We provide an input with a shape that does not follow the assumed pattern.
def test_incorrect_batch_dimension():
    # The kernel requires predictions and targets to be 2D and uses gridIdx.x as the row index.
    # If a user erroneously passes a 1D tensor (or a tensor with an unexpected shape) the kernel should error out.
    ker = build_kernel()
    
    # Create a 1D tensor, which is not acceptable.
    predictions = torch.randn(4096, device="cuda", dtype=torch.float32)
    targets = torch.randn(4096, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        loss = ker.forward(predictions, targets)
        torch.cuda.synchronize()
