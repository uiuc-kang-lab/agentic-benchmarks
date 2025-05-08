
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Data type flexibility (only supports float32)
def test_input_tensor_type():
    cuda_module = build_kernel()
    N = 64
    # Create double tensors which are not supported by the kernel
    A = torch.randn(N, N, device="cuda", dtype=torch.double)
    B = torch.randn(N, N, device="cuda", dtype=torch.double)
    A = torch.tril(A)
    B = torch.tril(B)
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel expects float pointers.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Contiguity assumption (non-contiguous inputs may produce incorrect behavior)
def test_non_contiguous_inputs():
    cuda_module = build_kernel()
    N = 64
    # Create contiguous lower triangular matrices that are then made non-contiguous by transposition
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    # Force non-contiguity by transposing and then transposing back (which may remove contiguity)
    A_noncontig = A.t()
    B_noncontig = B.t()
    # Although the operation is mathematically the same, the kernel assumes row-major contiguous storage.
    # Depending on the build and input, this might lead to wrong results.
    C = cuda_module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    # Compute a reference result using CPU (and then move to cuda for comparison)
    C_ref = torch.tril(torch.matmul(A_noncontig, B_noncontig))
    # We intentionally use a loose tolerance since non-contiguity might affect the values.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly handled non-contiguous inputs correctly."

# Issue 3: Lack of support for non-lower-triangular (or more general) triangular masks
def test_upper_triangular_input():
    cuda_module = build_kernel()
    N = 64
    # Create upper triangular matrices (which are not supported by the kernel logic)
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.triu(A)
    B = torch.triu(B)
    # The kernel is designed for lower triangular matrices and uses conditions based on row>=col.
    # When provided with an upper triangular matrix, the loaded values will be zero-ed out.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # The expected result for upper triangular multiplication is not computed by the kernel.
    # Thus, the result C should be different from torch.matmul(A, B) processed into upper triangular.
    C_reference = torch.triu(torch.matmul(A, B))
    assert not torch.allclose(C, C_reference, atol=1e-5), "Kernel unexpectedly produced correct results for an unsupported triangular type."
