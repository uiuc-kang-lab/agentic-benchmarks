
import torch
import pytest
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

def get_reference_result(A, B):
    # The operation is C = A.T * B.
    return torch.matmul(A.t(), B)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Issue 2: Test that non-contiguous inputs produce wrong results.
    # Create contiguous inputs then make them non-contiguous by a transpose.
    M, K, N = 64, 128, 96
    # Create A with shape (K, M) and B with shape (K, N)
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # Make them non contiguous. For example, take a transpose that does not yield a contiguous tensor.
    A_noncontig = A.t()  # now shape (M, K) and non-contiguous
    B_noncontig = B.t()  # shape (N, K) and non-contiguous
    # We need to transpose them back to the expected shape (K, M) and (K, N) for the kernel.
    # But if one mistakenly passes non-contiguous tensors to the kernel, the raw pointer access in C++ will assume
    # contiguous layout. So we simulate that mistake by intentionally passing the non-contiguous tensors.
    # Note: Although the Python code normally would call .contiguous(), this test is meant to trigger the issue.
    cuda_module = build_kernel()
    # do not call contiguous() so that the memory layout is wrong
    out = cuda_module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    
    ref = get_reference_result(A_noncontig, B_noncontig)
    # Because of the layout mismatch, the kernel’s output will be incorrect.
    max_diff = (out - ref).abs().max().item()
    assert max_diff > 1e-3, f"Kernel unexpectedly produced correct result with non-contiguous input! Max difference: {max_diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tile_dimension_edge_case():
    # Issue 1: Test a case where the K dimension (number of rows in A)
    # is such that the last tile loads fewer than TILE_SIZE elements.
    #
    # The kernel assumes a tile of fixed size and unrolls the loop unconditionally in steps of 4.
    # This test uses dimensions that are not multiples of TILE_SIZE (which is 16 in the kernel).
    # If a non-multiple of 4 tile size were ever used, the unrolling logic might access shared memory out-of-bound.
    #
    # Here we simulate a boundary condition by choosing a K that forces a partially full tile.
    M = 23    # arbitrary value for M (columns in A)
    K = 18    # chosen so that (K + 15) // 16 = 2 tiles, with the second tile having only 2 valid elements
    N = 31    # arbitrary value for N (columns in B)
    
    # Create A of shape (K, M) and B of shape (K, N)
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    cuda_module = build_kernel()
    out = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute reference (notice the kernel does C = A.T * B)
    ref = get_reference_result(A, B)
    
    # In a correct kernel the result should match the reference.
    # In our unrolling scenario, if TILE_SIZE assumption (divisible by 4) is violated in other cases,
    # this test may reveal an error if the out-of-bound shared memory accesses occur.
    if not torch.allclose(out, ref, atol=1e-5):
        max_diff = (out - ref).abs().max().item()
        pytest.fail(f"Kernel result differs from reference for edge-case dimensions. Max diff: {max_diff}")

