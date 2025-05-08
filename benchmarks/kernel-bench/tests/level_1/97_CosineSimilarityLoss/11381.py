
import pytest
import torch
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="cosine_loss_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Incomplete block reduction across warps.
# Create an input with a high D so that the number of threads used per block is greater than a single warp (i.e. >32).
# This test compares the CUDA kernel output with PyTorch’s own computation.
# Because of the bug, the outputs are expected to differ.
def test_incomplete_reduction():
    cuda_module = build_kernel()
    # Use a single row so that each CUDA block processes one row.
    # Set D high enough so that thread count (min(256, ceil(D/4))) > 32.
    N = 1
    D = 2048  # high dimension ensures threads per block > 32
    predictions = torch.randn(N, D, device="cuda", dtype=torch.float32)
    targets = torch.randn(N, D, device="cuda", dtype=torch.float32)
    
    # Expected loss computed using PyTorch's built-in operations.
    cosine_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
    expected_loss = torch.mean(1 - cosine_sim)
    
    # Run the CUDA kernel
    loss = cuda_module.forward(predictions, targets)
    torch.cuda.synchronize()
    # Due to incomplete reduction (bug), the kernel output will differ significantly.
    # We assert that the difference is above a tolerance.
    diff = abs(loss.item() - expected_loss.item())
    assert diff > 1e-3, f"Kernel reduction bug not triggered; diff={diff}"

# Test case 2: Non-float32 inputs to trigger float4 alignment assumption check.
# The kernel explicitly checks that the input tensors are of type float32.
# Here, providing float64 inputs should trigger a TORCH_CHECK failure.
def test_input_tensor_type():
    cuda_module = build_kernel()
    N = 128
    D = 4096
    predictions = torch.randn(N, D, device="cuda", dtype=torch.float64)
    targets = torch.randn(N, D, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="predictions must be float32"):
        _ = cuda_module.forward(predictions, targets)
        
if __name__ == "__main__":
    pytest.main([__file__])
