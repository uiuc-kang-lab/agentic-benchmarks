
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_square_matrices():
    # Issue 1: Kernel is hard-coded to work on square matrices.
    # This test deliberately passes non-square matrices and expects the kernel
    # to trigger a runtime error via TORCH_CHECK.
    N1 = 64
    N2 = 32  # non-square because 64 != 32
    A = torch.randn(N1, N2, dtype=torch.float32, device="cuda")
    B = torch.randn(N2, N1, dtype=torch.float32, device="cuda")
    my_module = build_kernel()
    with pytest.raises(RuntimeError, match="square matrix"):
        # Should fail the check: "A must be a square matrix" or "B must be a square matrix"
        my_module.forward(A, B)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wrong_dtype():
    # Issue 2: Kernel only supports float32.
    # This test passes double (float64) tensors to trigger the type-check.
    N = 128
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    my_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be a float32 tensor"):
        my_module.forward(A, B)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Issue 3: Kernel requires contiguous tensors.
    # This test makes a tensor non-contiguous (e.g. via transpose) and expects the kernel
    # to trigger the contiguous check.
    N = 128
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    # Create non-contiguous tensors by transposing
    A_nc = A.t()
    B_nc = B.t()
    my_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        my_module.forward(A_nc, B_nc)
