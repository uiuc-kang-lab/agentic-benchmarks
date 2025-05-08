
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA module from kernel.cu
def build_kernel():
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1 test: tile-dimension assumptions 
# We trigger the issue by constructing matrices with dimensions that force K not to be a multiple of BLOCK_SIZE_X.
# (Even though the kernel has conditionals for out-of-bound accesses, the mismatched tile loading for B will yield error.)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tile_dimension_assumption():
    my_module = build_kernel()
    # Choose dimensions such that K is not a multiple of 32.
    # We set A to be (M x K) and B to be (K x N) with K not matching the tile loop assumption.
    # Note: The kernel uses BLOCK_SIZE_X for iterating over K in the compute phase,
    # so if K is not a multiple of BLOCK_SIZE_X, the partial product accumulation might be off.
    M = 128
    K = 50  # not a multiple of 32
    N = 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Expect that due to tile-loading mismatch the results will differ significantly.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-3), (
        "Kernel output unexpectedly matched the reference; tile dimension issue not triggered"
    )

# Issue 2 test: Incorrect tensor type (non-float32)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incorrect_tensor_type():
    my_module = build_kernel()
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float64)
    B = torch.randn(K, N, device="cuda", dtype=torch.float64)
    # Even though the kernel does not check the type, the use of data_ptr<float>() on float64 data
    # should yield wrong results.
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.float(), B.float())
    # The results from the kernel (interpreting the memory as float) will be significantly off.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-3), (
        "Kernel output matches reference despite incorrect tensor type, which is unexpected"
    )

# Issue 3 test: Lack of CUDA error checking.
# We simulate a launch error by passing CPU tensors so that the kernel immediately throws an exception.
def test_cpu_tensor_input():
    my_module = build_kernel()
    M, K, N = 32, 32, 32
    A = torch.randn(M, K, device="cpu", dtype=torch.float32)
    B = torch.randn(K, N, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = my_module.forward(A, B)

# Issue 4 test: Transpose handling in complex stride scenarios.
# We trigger the potential pitfall by using non-contiguous tensors by transposing them.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_transpose_handling():
    my_module = build_kernel()
    # Create contiguous matrices.
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # Create non-contiguous versions: transpose A and B so that underlying memory layout changes.
    A_t = A.t()  # Now A_t shape is (K, M) and non-contiguous
    B_t = B.t()  # Now B_t shape is (N, K) and non-contiguous
    # According to the kernel logic, this should trigger the transposed branch.
    C_kernel = my_module.forward(A_t, B_t)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Due to primitive stride handling in the kernel, the result may be wrong.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-3), (
        "Kernel output unexpectedly matched the reference in transposed/non-contiguous scenario"
    )
