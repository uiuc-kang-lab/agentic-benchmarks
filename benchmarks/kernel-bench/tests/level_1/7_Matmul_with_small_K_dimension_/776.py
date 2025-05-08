
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

# Test 1: Input tensor different than float32.
def test_dtype_error():
    my_module = build_kernel()
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.double)  # double precision
    B = torch.randn(K, N, device="cuda", dtype=torch.double)
    with pytest.raises(RuntimeError):
        # Expect error from CUDA kernel usage because of wrong data type.
        my_module.forward(A, B)
    torch.cuda.synchronize()

# Test 2: Batched (3D) input which the kernel does not support.
def test_batched_input_error():
    my_module = build_kernel()
    # Create a batched tensor, e.g., shape (batch, M, K) which is not supported.
    batch, M, K, N = 2, 64, 32, 64
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, N, device="cuda", dtype=torch.float32)
    with pytest.raises(IndexError):
        # The kernel forward function assumes 2D shape. Accessing batched inputs might lead to index errors.
        my_module.forward(A, B)
    torch.cuda.synchronize()

# Test 3: Non-contiguous input tensors.
def test_non_contiguous_input_error():
    my_module = build_kernel()
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Create non-contiguous versions by transposing
    A_noncontig = A.t()
    B_noncontig = B.t()
    with pytest.raises(RuntimeError):
        # This should raise because the forward() explicitly checks for contiguous inputs.
        my_module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()

# Test 4: K not divisible by TILE_SIZE.
def test_inexact_tile_size():
    my_module = build_kernel()
    # Here TILE_SIZE is defined as 16. We choose K not divisible by 16, e.g., K=30.
    M, K, N = 64, 30, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Although the kernel uses guards in the loading phase, summing over TILE_SIZE strides in the inner loop
    # may lead to extraneous work. The result should still match torch.matmul though.
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differ for non-divisible K: diff max {(C - C_ref).abs().max()}"

if __name__ == "__main__":
    pytest.main([__file__])
