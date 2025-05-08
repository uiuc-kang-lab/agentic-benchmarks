
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Kernel only supports float32 inputs.
def test_input_type_float64():
    my_module = build_kernel()
    N = 64
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)  # double type
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # Expect the kernel to fail because it only supports float (float32)
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Warp divergence due to early return.
# We trigger this by ensuring a significant part of the output is in the upper triangular region.
def test_upper_triangular_zero():
    my_module = build_kernel()
    N = 128
    # Create proper lower triangular inputs so that multiplication
    # produces zeros in the upper triangular part
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Check that any element C[i,j] with i < j is exactly zero (within a tolerance)
    for i in range(N):
        for j in range(i+1, N):
            assert abs(C[i, j].item()) < 1e-5, f"C[{i},{j}] is not zero, got {C[i, j].item()}"

# Issue 3: Kernel launch configuration inefficiency (full grid launch).
# We trigger this by using a matrix size that is not a multiple of the tile size.
def test_non_divisible_dimensions():
    my_module = build_kernel()
    # Use a matrix dimension that is not divisible by TILE_SIZE (32)
    N = 70
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result using torch to compare, enforcing lower triangular result.
    C_ref = torch.tril(torch.matmul(A, B))
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from reference!"

# Issue 4: Kernel assumes contiguous input.
def test_non_contiguous_inputs():
    my_module = build_kernel()
    N = 64
    A_full = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B_full = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A_full = torch.tril(A_full)
    B_full = torch.tril(B_full)
    # Create non-contiguous tensors by transposing
    A = A_full.t()
    B = B_full.t()
    # Even after transposition, apply tril to mimic lower triangular structure.
    A = torch.tril(A)
    B = torch.tril(B)
    # The kernel assumes contiguous memory. This test should trigger an error or incorrect result.
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 5: Kernel only supports square matrices.
def test_non_square_input():
    my_module = build_kernel()
    M, N = 64, 32  # non-square dimensions
    A = torch.randn(M, N, device="cuda", dtype=torch.float32)
    B = torch.randn(M, N, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
