
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure we compile the latest version of the kernel from kernel.cu.
    module = load(
        name="test_kernel_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: max() and min() undefined in device code.
# This issue should trigger a kernel compile failure.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_min_undefined():
    # Attempt to build the kernel; if the kernel compiles successfully, this test fails.
    with pytest.raises(Exception) as excinfo:
        build_kernel()
    assert "max" in str(excinfo.value) or "min" in str(excinfo.value), \
        "Expected compile error regarding max/min not defined."

# Issue 2: non-contiguous inputs.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_input():
    module = build_kernel()
    N = 1024
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")

    # Create non-contiguous tensors by transposing twice.
    A_nc = A.t().contiguous().t()  # This ensures non-contiguity sometimes.
    B_nc = B.t().contiguous().t()

    # Force non-contiguity by slicing every other element in one dimension.
    A_nc = A_nc[:, ::2]
    B_nc = B_nc[:, ::2]

    # Expand back to square by zero-padding.
    # For testing, we rebuild square matrices from non-contiguous inputs.
    # We expect the kernel to either error out or produce incorrect results.
    N_new = A_nc.size(0)
    # If the kernel is not handling non-contiguous memory, the computed result will differ.
    try:
        C = module.forward(A_nc, B_nc)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.skip("Kernel did not process non-contiguous inputs as expected: " + str(e))
    # Compute reference result using torch.matmul (which handles non-contiguous inputs correctly)
    C_ref = torch.tril(torch.matmul(A_nc, B_nc))
    # We expect the outputs not to be equal due to wrong assumptions in memory layout.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel unexpectedly handled non-contiguous inputs correctly."

# Issue 3: Input tensor type other than float32.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_input_tensor_dtype():
    module = build_kernel()
    N = 512
    # Using double (float64) to trigger the type assumption issue.
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        module.forward(A, B)
    assert "float" in str(excinfo.value) or "dtype" in str(excinfo.value), \
        "Expected error when using non-float32 tensors."

# Issue 4: Limited generality of the kernel for non-divisible dimensions or different triangular forms.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_divisible_dimensions():
    module = build_kernel()
    # Use a dimension that is not a multiple of TILE_SIZE (TILE_SIZE == 32).
    N = 70
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    # Ensure matrices are lower triangular.
    A = torch.tril(A)
    B = torch.tril(B)
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result using torch.matmul and then trilu.
    C_ref = torch.tril(torch.matmul(A, B))
    # Due to the hard-coded loop boundaries and tile conditions, the kernel might 
    # compute incorrect results for dimensions not divisible by TILE_SIZE.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel incorrectly computes correct results for non-divisible dimension, expected failure for generality issues."
