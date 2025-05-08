
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel does not check for errors after launch.
# We try to trigger an error by intentionally providing dimension-mismatched inputs.
def test_dimension_mismatch():
    # A is expected to be (K, M) and B (N, K)
    # Provide wrong dimensions for B so that K mismatch occurs.
    M, K, N = 32, 64, 32
    # A of shape (K, M)
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    # B of shape (N, K+1) --> Wrong: extra column making B.T shape (K+1, N)
    B = torch.randn(N, K + 1, device="cuda", dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should result in an error (or wrong memory access) since the kernel does not check dims.
        _ = module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Kernel assumes contiguous inputs.
def test_noncontiguous_input():
    M, K, N = 128, 256, 64
    # Create contiguous inputs first
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    # Make them non-contiguous by transposing (the original API already does a .T,
    # but here we'll force non-contiguity in the underlying tensor).
    A_nc = A.T  # Now shape (M, K) but non-contiguous; note: our kernel expects A of shape (K, M)
    B_nc = B.T  # Now shape (K, N) but non-contiguous; kernel expects B of shape (N, K).
    
    # To intentionally trigger the issue, we pass non-contiguous tensors.
    module = build_kernel()
    # Expect that the kernel, which uses data_ptr without enforcing contiguity,
    # will produce incorrect results.
    C = module.forward(A_nc, B_nc)
    torch.cuda.synchronize()
    
    # Compute reference output assuming the intended mathematical operation
    # Note: The expected operation is C = (A_nc)^T * (B_nc)^T, but since A_nc and B_nc
    # are already transposed views, the math becomes ambiguous.
    # We force a reference computation by making the inputs contiguous.
    C_ref = torch.matmul(A_nc.contiguous().T, B_nc.contiguous().T)
    # We deliberately use a loose tolerance because the non-contiguous access may completely break the result.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly handled non-contiguous inputs correctly."

# Issue 3: Kernel does not support half precision.
def test_unsupported_dtype():
    M, K, N = 64, 128, 32
    A = torch.randn(K, M, device="cuda", dtype=torch.half)
    B = torch.randn(N, K, device="cuda", dtype=torch.half)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because half precision (torch.float16) is not handled by the dispatch macro.
        _ = module.forward(A, B)
        torch.cuda.synchronize()

# Issue 1 (additional): Kernel does not perform error checking after launch.
def test_missing_error_checking():
    M, K, N = 32, 64, 32
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    # We intentionally pass valid inputs to get a result.
    # However, because the kernel does not call cudaGetLastError(),
    # if there were any asynchronous errors they might go unnoticed.
    # We can simulate detection by checking cuda.synchronize() afterwards.
    C = module.forward(A, B)
    # Here we simply call synchronize and assume that if an error occurred,
    # torch.cuda.synchronize() would raise an exception.
    torch.cuda.synchronize()
    # Compute a reference result.
    C_ref = torch.matmul(A.T, B.T)
    # The test is only meant to check that no silent errors are present,
    # so we just compare the outputs.
    assert torch.allclose(C, C_ref, atol=1e-5), "Kernel output differs from reference output, which may indicate silent errors."

if __name__ == "__main__":
    pytest.main([os.path.realpath(__file__)])
