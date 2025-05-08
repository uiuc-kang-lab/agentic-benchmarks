
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="triangular_mm",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to compute the reference result
def triangular_mm_reference(A, B):
    # Compute full matmul then extract lower triangular part
    C = torch.matmul(A, B)
    return torch.tril(C)

# Test case 1: Out‐of‐bounds risk in B load due to wrong boundary check.
# Use a matrix size not a multiple of TILE_SIZE to force threads near the boundary.
def test_B_boundary_check():
    # TILE_SIZE is defined as 16 in the kernel; choose N such that N % 16 != 0.
    N = 18  # 18 is not a multiple of 16, so some tiles go near the boundary
    device = torch.device("cuda")
    # Create proper lower triangular matrices.
    A = torch.randn(N, N, device=device, dtype=torch.float32)
    B = torch.randn(N, N, device=device, dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    
    module = build_kernel()
    # This call is expected to potentially trigger an out–of–bounds error
    C_kernel = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = triangular_mm_reference(A, B)
    # The result may differ if out–of–bounds read occurs.
    assert torch.allclose(C_kernel, C_ref, atol=1e-4), \
        f"B boundary check issue: max diff {(C_kernel-C_ref).abs().max().item()}"

# Test case 2: Incorrect triangular condition for loading matrix B.
# In a lower triangular matrix B, entries above the diagonal should be zero.
# We deliberately set nonzero values in B’s upper triangle.
def test_B_triangular_condition():
    N = 64  # A moderate size so that many tiles are processed.
    device = torch.device("cuda")
    A = torch.randn(N, N, device=device, dtype=torch.float32)
    # Create a matrix B that is lower triangular then force its upper triangle to have nonzero values.
    B = torch.tril(torch.randn(N, N, device=device, dtype=torch.float32))
    # Introduce a clear error: set the upper triangle to a constant (which should be ignored in a proper lower-triangular multiply).
    B += torch.triu(torch.full((N, N), 5.0, device=device, dtype=torch.float32), diagonal=1)
    
    # The reference multiplication should use only the lower triangular part of A and B.
    C_ref = triangular_mm_reference(A, B)
    
    module = build_kernel()
    C_kernel = module.forward(A, B)
    torch.cuda.synchronize()
    
    # If the kernel does not enforce k >= col on B, then it might use the erroneous values from B’s upper part.
    assert torch.allclose(C_kernel, C_ref, atol=1e-4), \
        f"B triangular condition issue: max diff {(C_kernel-C_ref).abs().max().item()}"

# Test case 3: Missing boundary checks for matrix A’s shared memory load.
# Use a size that forces the tile load for A to approach the end of the matrix.
def test_A_boundary_check():
    N = 18  # Again, an N not a multiple of TILE_SIZE to stress the boundary conditions.
    device = torch.device("cuda")
    A = torch.randn(N, N, device=device, dtype=torch.float32)
    B = torch.randn(N, N, device=device, dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    
    module = build_kernel()
    C_kernel = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = triangular_mm_reference(A, B)
    # A boundary error in A would lead to a wrong output.
    assert torch.allclose(C_kernel, C_ref, atol=1e-4), \
        f"A boundary check issue: max diff {(C_kernel-C_ref).abs().max().item()}"

if __name__ == "__main__":
    pytest.main([__file__])
