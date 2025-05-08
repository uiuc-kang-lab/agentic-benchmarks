
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="triangular_mm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test for dtype issue: using double (float64) instead of float32.
def test_dtype_issue():
    device = "cuda"
    N = 256
    # Create double precision lower-triangular matrices.
    A = torch.randn(N, N, device=device, dtype=torch.float64)
    B = torch.randn(N, N, device=device, dtype=torch.float64)
    A = torch.tril(A)
    B = torch.tril(B)
    
    kernel = build_kernel()
    # The kernel does not support double, so its result should be wrong,
    # i.e. it will be interpreted as float32 data.
    # We compute a reference in double precision then cast to float32.
    C_ref = torch.tril(torch.matmul(A, B))
    # Expect the kernel output to NOT be close to the correct value.
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    
    # Check that the error is large.
    diff = (C - C_ref).abs().max()
    assert diff > 1e-3, f"Kernel unexpectedly produced correct results with double dtype! (max diff: {diff})"

# 2. Test for batched input issue: passing a 3D tensor should trigger an error.
def test_batched_input():
    device = "cuda"
    batch = 4
    N = 64
    # Create a batched lower-triangular matrix.
    A = torch.randn(batch, N, N, device=device, dtype=torch.float32)
    B = torch.randn(batch, N, N, device=device, dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    
    kernel = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        # This call should fail because our kernel only accepts 2D inputs.
        kernel.forward(A, B)
    assert "Tensor A must be 2D" in str(excinfo.value)

# 3. Test for precision issue in index mapping: using very large matrix sizes.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_matrix_precision():
    device = "cuda"
    # Use a size large enough that the float-based sqrt computation could be imprecise.
    # (This test may be a stress test; adjust N if necessary to trigger precision issues.)
    N = 8192
    A = torch.randn(N, N, device=device, dtype=torch.float32)
    B = torch.randn(N, N, device=device, dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    
    kernel = build_kernel()
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A, B))
    
    # With a correct implementation the results should be very close.
    # Here we expect that if rounding in index computation affected results,
    # the difference would be larger than a tight tolerance.
    tol = 1e-4
    diff = (C - C_ref).abs().max().item()
    assert diff < tol, f"Precision issue: maximum difference {diff} exceeds tolerance {tol}"

# 4. Test for non-contiguous input issue: passing noncontiguous tensors.
def test_non_contiguous():
    device = "cuda"
    N = 256
    A_full = torch.randn(N, N, device=device, dtype=torch.float32)
    B_full = torch.randn(N, N, device=device, dtype=torch.float32)
    A_full = torch.tril(A_full)
    B_full = torch.tril(B_full)
    
    # Make the tensors non-contiguous by transposing them.
    A_nc = A_full.t()
    B_nc = B_full.t()
    # Even after tril, transposition may leave unexpected strides.
    kernel = build_kernel()
    
    # The kernel does not check for contiguity so the result may be wrong.
    C = kernel.forward(A_nc, B_nc)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A_nc, B_nc))
    
    # We expect a significant difference due to wrong memory accesses.
    diff = (C - C_ref).abs().max()
    assert diff > 1e-3, f"Kernel unexpectedly handled non-contiguous inputs correctly! (max diff: {diff})"
