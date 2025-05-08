
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

# Test 1: Non-contiguous inputs
def test_non_contiguous():
    my_module = build_kernel()
    # Create contiguous matrices and then take a transpose to force non-contiguity.
    # Note: our kernel expects A to be of shape (K, M) and computes Aᵀ * B.
    K = 128
    M = 64
    N = 256
    A_cont = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B_cont = torch.randn(K, N, device='cuda', dtype=torch.float32)
    # Force non-contiguity by transposing A_cont (making it non-contiguous) and then transposing it back:
    A_noncontig = A_cont.T.T  # Although mathematically equivalent, some versions remain non-contiguous.
    assert not A_noncontig.is_contiguous(), "Test setup error: A_noncontig should be non-contiguous"
    C = my_module.forward(A_noncontig, B_cont)
    # The kernel computes C = Aᵀ * B.
    C_ref = torch.matmul(A_noncontig.t(), B_cont)
    torch.cuda.synchronize()
    assert torch.allclose(C, C_ref, atol=1e-5), (
        f"Kernel failed with non-contiguous input! "
        f"Max difference: {(C - C_ref).abs().max()}"
    )

# Test 2: Reference mismatch – using a “natural” matmul (without an explicit transpose) leads to error.
def test_reference_mismatch():
    my_module = build_kernel()
    K = 100
    M = 50
    N = 80
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    # Intentionally compute the reference as if the kernel were doing A * B
    C = my_module.forward(A, B)
    C_ref_incorrect = torch.matmul(A, B)
    torch.cuda.synchronize()
    # This test is meant to trigger an issue: the kernel computes Aᵀ * B.
    assert not torch.allclose(C, C_ref_incorrect, atol=1e-5), (
        "Expected mismatch since kernel computes A.T * B but reference used A * B"
    )

# Test 3: Boundary conditions for dimensions not divisible by TILE_SIZE
def test_inexact_tile_dimensions():
    my_module = build_kernel()
    # Choose dimensions that are not multiples of TILE_SIZE (which is 16).
    K = 73   # arbitrary K > TILE_SIZE that is not a multiple of 16
    M = 45   # arbitrary M not multiple of 16
    N = 37   # arbitrary N not multiple of 16
    # Remember, kernel expects A to be shape (K, M) and computes Aᵀ * B, so result is (M, N)
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C = my_module.forward(A, B)
    C_ref = torch.matmul(A.t(), B)
    torch.cuda.synchronize()
    assert torch.allclose(C, C_ref, atol=1e-5), (
        f"Kernel failed for inexact tile dimensions! Max difference: {(C - C_ref).abs().max()}"
    )

# Test 4: Incorrect data type should trigger an error
def test_incorrect_dtype():
    my_module = build_kernel()
    K = 64
    M = 32
    N = 48
    # Create inputs with double precision instead of float32.
    A = torch.randn(K, M, device='cuda', dtype=torch.float64)
    B = torch.randn(K, N, device='cuda', dtype=torch.float64)
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)
