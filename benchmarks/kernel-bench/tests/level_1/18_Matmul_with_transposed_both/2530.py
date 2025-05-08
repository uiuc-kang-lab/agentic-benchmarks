
import torch
import pytest
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger wrong results due to inconsistent shared memory indexing (B indexing issue)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incorrect_result_due_to_B_indexing():
    # Use dimensions that are not exact multiples of TILE_SIZE to expose indexing issues.
    # Here: M (output row dimension), K (reduction dimension), N (output column dimension)
    M = 33    # not a multiple of 32
    K = 35    # not a multiple of 32
    N = 37    # not a multiple of 32
    # According to the Python wrapper, A and B should be of shape (K, M) and (N, K) respectively.
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    # The forward function in the Python code does: torch.matmul(A.T, B.T)
    ref = torch.matmul(A.T, B.T)
    kernel_mod = build_kernel()
    C = kernel_mod.forward(A, B)
    torch.cuda.synchronize()
    # This assertion is expected to fail if the shared memory indexing is wrong.
    assert torch.allclose(C, ref, atol=1e-3), f"Kernel output differs from reference output! Max diff: {(C - ref).abs().max()}"

# Test case 2: Trigger error when input tensor type is not supported (e.g. half precision)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_half_precision_input_not_supported():
    # Use half (float16) precision input that is not handled by the AT_DISPATCH_FLOATING_TYPES.
    M = 64
    K = 64
    N = 64
    A = torch.randn(K, M, device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16)
    kernel_mod = build_kernel()
    with pytest.raises(RuntimeError):
        C = kernel_mod.forward(A, B)
        torch.cuda.synchronize()

# Test case 3: Test with dimensions not exactly divisible by tile size to trigger potential boundary issues
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tile_boundary_conditions():
    # Choose dimensions such that N is much smaller than a multiple of the tile size, which stresses the "active_N" logic.
    M = 32  # exactly one tile row for output
    K = 50  # not an exact multiple of 32
    N = 10  # small number that forces a partial tile
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    ref = torch.matmul(A.T, B.T)
    kernel_mod = build_kernel()
    C = kernel_mod.forward(A, B)
    torch.cuda.synchronize()
    # This assertion may fail if the chunk division or active_N handling is not correct.
    assert torch.allclose(C, ref, atol=1e-3), f"Kernel output differs from reference output in tile boundaries! Max diff: {(C - ref).abs().max()}"
