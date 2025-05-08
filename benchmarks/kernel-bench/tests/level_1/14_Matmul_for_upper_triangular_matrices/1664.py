
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test to trigger issue for unsupported data types (e.g. torch.double)
def test_input_tensor_type():
    # Create double tensors on CUDA.
    N = 64
    A = torch.triu(torch.randn(N, N, device="cuda", dtype=torch.double))
    B = torch.triu(torch.randn(N, N, device="cuda", dtype=torch.double))
    mod = build_kernel()
    
    # Expect the kernel to be operating on float tensors.
    # The kernel call will use A.data_ptr<float>(), so the data interpretation is wrong.
    C_kernel = mod.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute reference result using PyTorch (which casts appropriately, but our module is wrong).
    C_ref = torch.triu(torch.matmul(A.float(), B.float()))
    
    # The error is not necessarily an exception but a wrong output.
    # Hence, we assert the outputs are not close.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-3), (
        "Kernel unexpectedly produced correct result with double input; expected failure due to type mismatch."
    )

# 2. Test to trigger issue for non-square matrices.
def test_non_square_matrix():
    # Create non-square tensors on CUDA.
    # For instance, A is (N, M) and B is (M, M) such that A is upper-triangular in its (N x M) shape.
    N, M = 64, 96
    # Create a non-square "upper-triangular" style matrix: zeros appear in the lower part of each row.
    A = torch.triu(torch.randn(N, M, device="cuda", dtype=torch.float32))
    B = torch.triu(torch.randn(M, M, device="cuda", dtype=torch.float32))
    
    mod = build_kernel()
    # The kernel uses A.size(0) (i.e. N) and assumes the input is square.
    # Thus, we expect that it will not compute the correct output for non-square matrices.
    with pytest.raises(RuntimeError):
        # We expect an error because our kernel indexing uses N from A.size(0) and then accesses B with A.size(0)
        C_kernel = mod.forward(A, B)
        torch.cuda.synchronize()

# 3. Test to trigger issue when input tensors are not contiguous.
def test_non_contiguous_tensor():
    # Create contiguous upper-triangular matrices first.
    N = 64
    A = torch.triu(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.triu(torch.randn(N, N, device="cuda", dtype=torch.float32))
    
    # Make non-contiguous copies by transposing. Transpose produces a non-contiguous tensor.
    A_nc = A.t()
    B_nc = B.t()
    
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel directly accesses data_ptr<float>() which may yield inconsistent results if non-contiguous.
        C_kernel = mod.forward(A_nc, B_nc)
        torch.cuda.synchronize()
