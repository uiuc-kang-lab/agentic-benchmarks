
import os
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    return load(
        name="softmax_kernel_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )

# Issue 1: Test for divergent __syncthreads() in blocks with very low num_features.
# In such a case, some warps have no valid features (warp_start >= num_features) and might skip the barrier.
def test_divergent_syncthreads():
    cuda_module = build_kernel()
    # Choose a very small number of features so that only a few warps have work.
    # For THREADS_PER_BLOCK=256 and WARP_SIZE=32, there are 8 warps.
    # Let num_features = 10, so that warps 0 to 4 get work and warps 5-7 do nothing.
    batch_size = 4
    num_features = 10
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    try:
        y = cuda_module.forward(x)
        # If the kernel hangs due to deadlock, the test will timeout.
        # If it completes, check that softmax-like properties hold.
        # Softmax outputs should sum to 1 along dimension 1.
        sums = y.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "Output rows do not sum to 1."
    except Exception as e:
        pytest.fail(f"Kernel execution failed (potential __syncthreads divergence): {e}")

# Issue 2: Test kernel behavior when num_features > MAX_FEATURES.
# In this branch the kernel avoids using shared memory for exponent values.
def test_large_num_features():
    cuda_module = build_kernel()
    # Use a number of features larger than MAX_FEATURES (2048)
    batch_size = 2
    num_features = 4096
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    try:
        y = cuda_module.forward(x)
        # Verify softmax: rows should sum to nearly 1.
        sums = y.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "For large num_features, output rows do not sum to 1."
    except Exception as e:
        pytest.fail(f"Kernel execution failed for num_features > MAX_FEATURES: {e}")

# Issue 3: Test lack of proper asynchronous error checking.
# Although error checking is done via TORCH_CHECK for input properties,
# we test if a misbehaving input (if any) causes a detectable error.
def test_invalid_data_access():
    cuda_module = build_kernel()
    batch_size = 4
    num_features = 128
    # Create a tensor with proper dimensions but intentionally use a non-contiguous tensor.
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32).t().contiguous().t()
    try:
        y = cuda_module.forward(x)
        sums = y.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "Output rows do not sum to 1 for non-contiguous input."
    except Exception as e:
        pytest.fail(f"Kernel execution failed on non-standard memory layouts: {e}")

# Issue 4: Test that the kernel only accepts float32 inputs.
# Passing a tensor with a different dtype should trigger a TORCH_CHECK failure.
def test_input_dtype():
    cuda_module = build_kernel()
    batch_size = 4
    num_features = 256
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float64)  # Wrong dtype
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        # The TORCH_CHECK in the kernel wrapper should reject wrong dtype.
        _ = cuda_module.forward(x)

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
