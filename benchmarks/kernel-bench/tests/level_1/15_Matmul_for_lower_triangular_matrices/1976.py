
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def reference_triangular_mm(A, B):
    # Compute the triangular matmul similar to the kernel routine.
    # (C[i,j] = sum_{k=j}^{i} A[i,k] * B[k,j] for i>=j, zero otherwise)
    N = A.size(0)
    C = torch.zeros_like(A)
    for i in range(N):
        for j in range(N):
            if i < j:
                C[i, j] = 0.0
            else:
                s = 0.0
                for k in range(j, i + 1):
                    s += A[i, k] * B[k, j]
                C[i, j] = s
    return C

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_type_issue():
    # Issue 1: Pass double tensors instead of float32.
    N = 64
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float64))
    B = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float64))
    mod = build_kernel()
    with pytest.raises(Exception):
        # Expect the kernel call to fail or produce an error because the kernel 
        # assumes float (float32) inputs.
        mod.forward(A, B)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_inputs_issue():
    # Issue 2: Create non-contiguous tensors.
    N = 64
    # Create a tensor and then an operation that yields a non-contiguous view.
    A = torch.tril(torch.randn(N, N, device='cuda')).t().clone().t()
    B = torch.tril(torch.randn(N, N, device='cuda')).t().clone().t()
    # Force non-contiguity if not already.
    if A.is_contiguous():
        A = A.clone().transpose(0, 1)
    if B.is_contiguous():
        B = B.clone().transpose(0, 1)
    mod = build_kernel()
    with pytest.raises(Exception):
        # Expect failure because kernel raw pointer arithmetic assumes contiguous memory.
        mod.forward(A, B)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_square_matrices_issue():
    # Issue 3: Passing non-square matrices.
    N = 64
    M = 32  # Non-square: A is 64x32 and B is 32x64, for instance.
    A = torch.tril(torch.randn(N, M, device='cuda', dtype=torch.float32))
    B = torch.tril(torch.randn(M, N, device='cuda', dtype=torch.float32))
    mod = build_kernel()
    with pytest.raises(Exception):
        # Expect error due to dimension mismatch inside the kernel (kernel assumes square matrices)
        mod.forward(A, B)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_small_matrix_streaming_issue():
    # Issue 4: A very small matrix might trigger edge conditions in stream chunking.
    N = 10  # Very small matrix compared to the fixed NUM_STREAMS=4 in kernel
    A = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device='cuda', dtype=torch.float32))
    mod = build_kernel()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = reference_triangular_mm(A, B)
    # Since the stream chunking might not work properly for small matrices,
    # we expect a significant difference.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Expected the kernel output to differ from the reference result due to streaming issues."
    )
