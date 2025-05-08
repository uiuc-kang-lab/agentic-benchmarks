
import pytest
import torch
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

def test_input_tensor_type():
    # Issue 1: The kernel does not check that input tensors are float32.
    # Passing double (float64) tensors will lead the kernel to misinterpret raw data.
    N = 256
    # Create double precision symmetric matrices.
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')
    A = (A + A.T) / 2
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    B = (B + B.T) / 2

    kernel = build_kernel()
    # The kernel is expecting float32, so we expect the result to be off.
    C_kernel = kernel.forward(A, B)
    torch.cuda.synchronize()
    # Compute a reference result using float32 (casting the inputs)
    C_ref = torch.matmul(A.float(), B.float())
    # If the kernel did proper type-checking, it should have raised an error.
    # Instead, if it ran, the output is likely numerically different.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-3), (
        "Kernel did not exhibit the expected type issue with float64 input."
    )

def test_non_contiguous_input():
    # Issue 2: The kernel does not verify that input tensors are contiguous.
    # Here, we create non-contiguous matrices by transposing the symmetric matrix.
    N = 256
    # Create a symmetric matrix in float32.
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    A = (A + A.T) / 2
    B = torch.randn(N, N, dtype=torch.float32, device='cuda')
    B = (B + B.T) / 2

    # Create non-contiguous tensors by taking a transpose.
    A_noncontig = A.t()
    B_noncontig = B.t()

    kernel = build_kernel()
    C_kernel = kernel.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A_noncontig, B_noncontig)
    # Expect the kernel to produce incorrect results for non-contiguous input.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel did not exhibit the issue with non-contiguous input tensors."
    )

def test_matrix_size_not_divisible_by_block():
    # Issue 3: The kernel assumes that BLOCK_SIZE is a multiple of 4 because of manual loop unrolling.
    # Although BLOCK_SIZE is defined as 32 here, using an input matrix size that is not a multiple of BLOCK_SIZE
    # stresses the edge tile code paths. In such cases, the unrolling may access shared memory out-of-bound.
    N = 70  # deliberately not divisible by BLOCK_SIZE (32)
    # Create symmetric matrices.
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    A = (A + A.T) / 2
    B = torch.randn(N, N, dtype=torch.float32, device='cuda')
    B = (B + B.T) / 2

    kernel = build_kernel()
    C_kernel = kernel.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # In the presence of the BLOCK_SIZE assumption issue, the output for edge tiles may be wrong.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel output appears correct even for matrix sizes not divisible by BLOCK_SIZE. "
        "This suggests that the unrolling assumption may not be enforced."
    )
