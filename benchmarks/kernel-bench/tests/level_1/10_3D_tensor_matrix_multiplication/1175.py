
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility to build and load the CUDA kernel module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger the inefficiency of fixed CHUNK_SIZE.
# Use an input tensor with N not divisible by CHUNK_SIZE so that the last chunk
# has fewer than CHUNK_SIZE rows. This test verifies that the kernel still produces
# the correct result even though extra threads perform an out‐of-bound check.
def test_non_multiple_chunk_size():
    cuda_module = build_kernel()
    # Set dimensions so that N is not divisible by CHUNK_SIZE (CHUNK_SIZE is 32 in the kernel)
    N = 50  # Not divisible by 32, so last chunk has only 18 rows valid.
    M = 128
    K = 64
    L = 32

    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    output = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    output_ref = torch.matmul(A, B)
    assert torch.allclose(output, output_ref, atol=1e-5), \
        f"Output does not match reference for non-multiple of chunk size. Max diff: {(output - output_ref).abs().max()}"

# Test case 2: Trigger potential issues with the unrolling pragma.
# Although we cannot force a compile-time constant for K, we can test with a large K
# to see if the performance or correctness is affected.
def test_large_K_value():
    cuda_module = build_kernel()
    N = 16
    M = 64
    K = 257  # a non-power-of-two and non-unrolled-friendly size
    L = 32

    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    output = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    output_ref = torch.matmul(A, B)
    assert torch.allclose(output, output_ref, atol=1e-5), \
        f"Output does not match reference for large K. Max diff: {(output - output_ref).abs().max()}"

# Test case 3: Pass a non-CUDA tensor to trigger CHECK_CUDA.
def test_non_cuda_input():
    cuda_module = build_kernel()
    N = 16
    M = 64
    K = 128
    L = 32

    # Create CPU tensors instead of CUDA
    A = torch.randn(N, M, K, device="cpu", dtype=torch.float32)
    B = torch.randn(K, L, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError) as excinfo:
        _ = cuda_module.forward(A, B)
    assert "must be a CUDA tensor" in str(excinfo.value)

# Test case 4: Pass a non-contiguous tensor to trigger CHECK_CONTIGUOUS.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    N = 16
    M = 64
    K = 128
    L = 32

    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)

    # Make A non-contiguous by transposing two dimensions and then transposing back partially.
    A_non_contig = A.transpose(1, 2)
    with pytest.raises(RuntimeError) as excinfo:
        _ = cuda_module.forward(A_non_contig, B)
    assert "must be contiguous" in str(excinfo.value)

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
