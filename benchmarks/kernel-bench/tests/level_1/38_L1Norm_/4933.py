
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Compile the CUDA extension from the 'kernel.cu' file.
def build_kernel():
    cuda_module = load(
        name="l1_norm_cuda",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True
    )
    return cuda_module

# Issue 1: Test for misaligned memory due to vectorized load/store assumptions.
# We create a tensor that is not 16-byte aligned by making a tensor with extra column
# and then taking a narrow view so that the storage pointer for each row is shifted.
def test_misaligned_input():
    # Create a tensor slightly larger than needed.
    N = 4
    D_full = 17  # not a multiple of 4, and extra column to allow offset slicing.
    # Allocate on CUDA.
    full_tensor = torch.randn(N, D_full, device="cuda", dtype=torch.float32)
    # Take a narrow view starting from column 1, which may misalign the row pointer.
    # This view has shape (N, D) where D = D_full - 1.
    x = full_tensor.narrow(1, 1, D_full - 1)
    
    # Run the custom CUDA L1 normalization kernel.
    kernel = build_kernel()
    try:
        out = kernel.forward(x)
        # Check that the norm of each row is 1 (or near 1) within numerical tolerance.
        row_norms = torch.sum(torch.abs(out), dim=1)
        assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-4), \
            f"Row norms do not equal 1: {row_norms.cpu().numpy()}"
    except Exception as e:
        pytest.fail(f"Kernel failed on misaligned input: {e}")

# Issue 2: Test for the inconsistency between dynamic and static shared memory usage.
# The test uses an input size that forces the kernel to compute a nontrivial dynamic shared memory size.
def test_dynamic_vs_static_shared():
    # Use a D that is a multiple of 4 to enforce the vectorized path.
    N = 8
    D = 256  # large D will result in many threads per block when using the kernel launch logic.
    
    # Create a normally allocated tensor (which will be 4-byte aligned)
    x = torch.randn(N, D, device="cuda", dtype=torch.float32)
    
    kernel = build_kernel()
    try:
        out = kernel.forward(x)
        # Verify that the L1 normalization was performed correctly.
        row_norms = torch.sum(torch.abs(out), dim=1)
        assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-4), \
            f"Row norms do not equal 1: {row_norms.cpu().numpy()}"
    except Exception as e:
        pytest.fail(f"Kernel failed in dynamic/shared memory configuration test: {e}")
        
if __name__ == "__main__":
    pytest.main([__file__])
