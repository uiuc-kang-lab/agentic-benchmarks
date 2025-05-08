
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper function to build and load the CUDA kernel extension.
def build_kernel():
    # Ensure we compile the CUDA kernel from kernel.cu
    module = load(
        name="mse_cuda_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test 1: Mismatched work partitioning.
# Create an input whose number of elements is not 
# a multiple of 2 * BLOCK_SIZE * gridDim (when gridDim 
# is computed from (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE).
def test_grid_size_mismatch():
    module = build_kernel()
    # Choose a size that is "awkward": not a multiple of 2 * blockDim * gridDim.
    # For example, choose a tensor with a total number of elements that is small and odd.
    N = 103  # arbitrary odd number
    preds = torch.randn(N, device="cuda", dtype=torch.float32)
    targets = torch.randn(N, device="cuda", dtype=torch.float32)
    
    # Compute reference MSE in PyTorch
    mse_ref = torch.mean((preds - targets) ** 2).item()

    # Run our CUDA kernel. Due to the grid size / stride mismatch,
    # the kernel might accumulate an incorrect sum.
    mse_cuda = module.forward(preds, targets).item()
    
    # The error may be subtle; we check with a very strict tolerance.
    assert abs(mse_cuda - mse_ref) < 1e-5, (
        f"Grid size mismatch issue: expected MSE {mse_ref}, but got {mse_cuda}"
    )

# Test 2: Atomic double compatibility.
# Although we cannot force an old GPU in the test,
# we can trigger the kernel with large input which stresses the atomicAdd.
# On GPUs without double atomic support, the kernel launch should fail or throw an error.
def test_atomic_double_compatibility():
    module = build_kernel()
    # Create a large tensor to force many atomicAdd calls.
    N = 1 << 20  # about 1 million elements
    preds = torch.randn(N, device="cuda", dtype=torch.float32)
    targets = torch.randn(N, device="cuda", dtype=torch.float32)
    try:
        mse_cuda = module.forward(preds, targets)
        # If no error and we get a result, then we assume the atomicAdd for double is working.
        # We still compare numerically.
        mse_ref = torch.mean((preds - targets) ** 2)
        assert torch.allclose(mse_cuda.float(), mse_ref, atol=1e-5), (
            f"Atomic double accumulation might be wrong: got {mse_cuda}, expected {mse_ref}"
        )
    except RuntimeError as e:
        # If the error message mentions lack of support for double atomic adds, then that's the expected failure.
        assert "atomicAdd" in str(e), f"Unexpected error: {e}"

# Test 3: Non-contiguous input tensors.
# Create non-contiguous inputs (e.g., via transposition) to see if the kernel, which expects contiguous memory, misbehaves.
def test_non_contiguous_inputs():
    module = build_kernel()
    # Create a 2D tensor and then transpose it to make it non-contiguous.
    preds = torch.randn(128, 4096, device="cuda", dtype=torch.float32).t()
    targets = torch.randn(128, 4096, device="cuda", dtype=torch.float32).t()
    # Although the total number of elements match, the underlying memory layout is non-contiguous.
    with pytest.raises(RuntimeError):
        # The kernel should either detect the non-contiguity or compute an incorrect result.
        # Here, we expect it to raise an error.
        module.forward(preds, targets)

# Test 4: Lack of post-kernel error checking.
# Pass in mismatched tensors to trigger the TORCH_CHECK and ensure the kernel launch fails before execution.
def test_shape_mismatch():
    module = build_kernel()
    preds = torch.randn(100, device="cuda", dtype=torch.float32)
    targets = torch.randn(101, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        module.forward(preds, targets)
        
if __name__ == "__main__":
    pytest.main([__file__])
