
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

# Test case for Issue 1 and 2: Incorrect asynchronous memcpy and stream misuse.
# We run a simple case with float32 matrices on GPU.
# If the kernel were correct, the result should match torch.tril(torch.matmul(A, B)).
# Due to the unnecessary memory copies and stream mismanagement,
# the output will likely be wrong.
def test_incorrect_memcpy_and_stream_usage():
    cuda_module = build_kernel()
    N = 128
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A, B))
    # Expect the output to be incorrect due to the memory-copy and stream issues.
    assert not torch.allclose(C, C_ref, atol=1e-4), (
        "Kernel output unexpectedly matches reference output despite memory copy / stream issues."
    )

# Test case for Issue 3: The k_tile loop boundaries may be wrong when N is not a multiple of TILE_SIZE.
# Use a matrix size that is not divisible by 32.
def test_non_divisible_dimension():
    cuda_module = build_kernel()
    N = 70  # Not a multiple of TILE_SIZE (32)
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A, B))
    # Expect incorrect accumulation over the k dimension.
    assert not torch.allclose(C, C_ref, atol=1e-4), (
        "Kernel output unexpectedly matches reference output for non-divisible dimensions."
    )

# Test case for Issue 4: The kernel does not support double precision.
# Supplying double precision inputs should raise an error.
def test_double_precision():
    cuda_module = build_kernel()
    N = 64
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    A = torch.tril(A)
    B = torch.tril(B)
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(A, B)
