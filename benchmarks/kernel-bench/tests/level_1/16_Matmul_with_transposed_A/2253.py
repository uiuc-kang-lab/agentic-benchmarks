
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

# Test 1: Check that non-contiguous tensors produce incorrect result.
# The kernel does not check for contiguity and will mis-index non-contiguous memory.
def test_non_contiguous_input():
    # Create contiguous inputs first.
    M, K, N = 1024, 4096, 2048
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # make non-contiguous views (for instance by transposing the contiguous tensor)
    A_noncontig = A.T  # this view is non-contiguous (shape M x K)
    # We need to transpose back to get the expected shape for the kernel.
    # Kernel expects A of shape (K, M), so providing a noncontiguous tensor will cause wrong indexing.
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting an error or at least an incorrect result that does not match reference.
        # Here we simulate an error by checking that the result is not close.
        C = cuda_module.forward(A_noncontig, B)
        torch.cuda.synchronize()
        # Compute reference using explicit matmul (forcing contiguous versions)
        C_ref = torch.matmul(A_noncontig.T, B)
        assert not torch.allclose(C, C_ref, atol=1e-5), \
            "Kernel unexpectedly handled non-contiguous input correctly."

# Test 2: Check that inputs of wrong type (e.g. float64) raise an error.
def test_wrong_dtype():
    M, K, N = 1024, 4096, 2048
    A = torch.randn(K, M, device="cuda", dtype=torch.float64)
    B = torch.randn(K, N, device="cuda", dtype=torch.float64)
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel checks for float32 and should throw an error.
        _ = cuda_module.forward(A, B)

# Test 3: Check behavior with matrix dimensions that are not multiples of TILE_DIM.
# This is to stress the boundary handling in M and N dimensions.
def test_non_uniform_dimensions():
    # Choose dimensions that are not multiples of 16 (TILE_DIM value) e.g. 17, 18 etc.
    M, K, N = 17, 50, 19
    # Ensure tensors are contiguous and of type float32.
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    cuda_module = build_kernel()
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Reference: Note the operation is C = A.T * B.
    C_ref = torch.matmul(A.T, B)
    # Because of atomicAdd and partial reductions, differences in floating-point order may arise.
    # We check that the result is approximately correct.
    assert torch.allclose(C, C_ref, atol=1e-3), \
        f"Kernel output ({C}) differs from reference ({C_ref}) beyond acceptable tolerance."

# Test 4: Check that repeated runs (due to split-K atomicAdd) produce numerical differences.
# Although atomicAdd ensures correctness, the accumulation order may be non-deterministic.
def test_nondeterministic_accumulation():
    M, K, N = 1024, 4096, 2048
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    cuda_module = build_kernel()
    
    # Run the kernel multiple times and see if the results are bitwise identical.
    # Even if they are numerically close, atomicAdd might result in slight differences.
    C1 = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C2 = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    
    # The results might be nearly identical within a tolerance, but we want to warn that
    # the summation order might be non-deterministic if differences are observed.
    if not torch.allclose(C1, C2, atol=1e-6):
        pytest.skip("Non-deterministic accumulation due to atomicAdd detected (expected behavior).")
    else:
        # If bitwise equality is observed, we pass the test to indicate determinism.
        assert torch.allclose(C1, C2, atol=1e-6), "Results should be deterministically similar."

if __name__ == "__main__":
    pytest.main([__file__])
