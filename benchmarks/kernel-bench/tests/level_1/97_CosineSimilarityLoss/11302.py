
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_tensors():
    # Issue 1: Pass non-contiguous input tensors that are not suitable for vectorized loads.
    batch_size = 128
    D = 4096
    # Create a tensor and then take a non-contiguous slice.
    base_pred = torch.randn(batch_size, D * 2, device='cuda', dtype=torch.float32)
    base_tgt = torch.randn(batch_size, D * 2, device='cuda', dtype=torch.float32)
    predictions = base_pred[:, ::2]  # Non-contiguous view with correct shape
    targets = base_tgt[:, ::2]       # Non-contiguous view with correct shape

    # Although the kernel does not check for contiguity,
    # using non-contiguous tensors may lead to undefined behavior.
    module = build_kernel()
    # We do not know the expected output exactly because the kernel might work incorrectly,
    # so we at least run the kernel to see if it errors or produces a result.
    with pytest.raises(RuntimeError):
        # It is expected that misaligned loads cause an error or incorrect behaviour.
        output = module.forward(predictions, targets)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_wrong_dtype():
    # Issue 2: Pass a tensor of wrong dtype (float64 instead of float32) to trigger the scalar_type check.
    batch_size = 128
    D = 4096
    predictions = torch.randn(batch_size, D, device='cuda', dtype=torch.float64)
    targets = torch.randn(batch_size, D, device='cuda', dtype=torch.float64)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        output = module.forward(predictions, targets)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_warp_sync_inactive_lanes():
    # Issue 3: Use a very small D to cause many threads in a block to be idle
    # and trigger potential incorrect behavior in warp-level reduction if active lanes mask isn't computed.
    # Here, D is chosen to be smaller than the warp size (and smaller than vec_size), forcing only a few threads
    # to perform work while the rest are idle.
    batch_size = 128
    D = 2  # Very small D; since vec_size is 4, the vectorized loop is skipped
    predictions = torch.randn(batch_size, D, device='cuda', dtype=torch.float32)
    targets = torch.randn(batch_size, D, device='cuda', dtype=torch.float32)
    module = build_kernel()
    
    # Since the kernel uses a constant mask for __shfl_down_sync,
    # the reduction may be computed incorrectly.
    # We compare the output with the expected loss computed with PyTorch operations.
    output = module.forward(predictions, targets)
    torch.cuda.synchronize()
    
    # Compute reference loss using PyTorch function:
    cos_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
    reference_loss = torch.mean(1.0 - cos_sim)
    
    # If the kernel reduction is flawed, the difference may be significant.
    # Note: We use a loose tolerance here since atomicAdd order may introduce minor differences.
    assert not torch.allclose(output, reference_loss, atol=1e-3), \
        "Kernel output matches the reference loss unexpectedly; the warp reduction might not be triggering the bug."

