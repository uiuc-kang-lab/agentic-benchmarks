
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from the provided kernel source.
def build_kernel():
    cuda_module = load(
        name="l1_norm_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger misaligned memory access.
# We create a misaligned tensor by allocating a larger tensor and slicing out a sub-tensor with an offset.
def test_misaligned_memory():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Create a tensor with one extra element so that a slice won't be 16-byte aligned.
    base = torch.randn(16, 16384 + 1, device="cuda", dtype=torch.float32)
    # Use a narrow slice starting from an offset that is likely to be misaligned.
    x = base[:, 1:]  # this slice may be misaligned relative to a float4 boundary
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # We try to run the kernel. If memory alignment is required, undefined behavior (or a launch error)
        # may be triggered. We catch a RuntimeError as a sign of failure.
        y = kernel.forward(x)
        torch.cuda.synchronize()

# Test 2: Use a tensor with a number of columns that forces a thread configuration that might not fully
# satisfy the assumption that blockDim.x is a multiple of 32. For instance a very tiny D.
def test_unusual_thread_config():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Choosing D such that (D+3)//4 becomes 1: e.g., D = 1,2,or 3
    D = 3
    x = torch.randn(16, D, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # The kernel reduction code assumes the blockDim.x is a multiple of 32. For a very small D, the computed
    # thread count will be 32 (which is a multiple of 32) but the work distribution may be imbalanced.
    # We check that the output, while computed by the kernel, is (numerically) consistent with L1 norm.
    y = kernel.forward(x)
    torch.cuda.synchronize()
    # Compute reference result using PyTorch
    norm = torch.sum(torch.abs(x), dim=1, keepdim=True).clamp(min=1e-12)
    y_ref = x / norm
    assert torch.allclose(y, y_ref, atol=1e-5), "Kernel output differs for small D (unusual thread config)!"

# Test 3: Pass an input tensor with a wrong data type (e.g., float64) to trigger a failure.
def test_input_tensor_type():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Create a tensor of type float64
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float64)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel expects float32. The extension should check (or fail during reinterpret_cast)
        y = kernel.forward(x)
        torch.cuda.synchronize()

# Test 4: Pass a non-2D tensor to trigger the dimensionality assumption failure.
def test_input_tensor_dimension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Create a 3D tensor
    x = torch.randn(4, 16, 16384, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel explicitly checks that the input is 2D.
        y = kernel.forward(x)
        torch.cuda.synchronize()
        
if __name__ == "__main__":
    pytest.main([__file__])
