
import pytest
import torch
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

def test_non_float_type():
    # Issue 2: The kernel assumes float32 input.
    # Using double precision should raise an error because the kernel uses data_ptr<float>().
    N = 64
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        kernel.forward(A, B)

def test_non_square_matrix_A():
    # Issue 4: The kernel checks for square matrices.
    # Passing a non-square matrix for A should trigger a TORCH_CHECK error.
    A = torch.randn(64, 32, dtype=torch.float32, device='cuda')
    B = torch.randn(64, 64, dtype=torch.float32, device='cuda')
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        kernel.forward(A, B)

def test_non_square_matrix_B():
    # Issue 4: Passing a non-square matrix for B should also trigger a TORCH_CHECK error.
    A = torch.randn(64, 64, dtype=torch.float32, device='cuda')
    B = torch.randn(64, 32, dtype=torch.float32, device='cuda')
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        kernel.forward(A, B)

def test_mismatched_sizes():
    # Issue 4: The kernel requires the matrices to have the same size.
    # Using mismatched dimensions should trigger an error.
    A = torch.randn(64, 64, dtype=torch.float32, device='cuda')
    B = torch.randn(32, 32, dtype=torch.float32, device='cuda')
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        kernel.forward(A, B)

def test_non_multiple_of_block_size():
    # Although the kernel does check bounds, this case is provided for completeness.
    # Issue 3 (and implicitly Issue 1) hints at performance pitfalls on non-standard sizes.
    # The result will be numerically correct but this test ensures that tiling works with arbitrary sizes.
    N = 70  # Not a multiple of BLOCK_SIZE (32)
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    kernel = build_kernel()
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from reference output! Max diff: {(C - C_ref).abs().max()}"
