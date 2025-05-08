
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension module.
def build_kernel():
    cuda_module = load(
        name="triangular_mm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Non float32 tensor input
def test_non_float_input():
    # Create double precision (float64) lower triangular matrices.
    N = 128
    A = torch.tril(torch.randn(N, N, dtype=torch.float64, device="cuda"))
    B = torch.tril(torch.randn(N, N, dtype=torch.float64, device="cuda"))
    mod = build_kernel()
    # Expecting a failure or incorrect result because the kernel assumes float32.
    with pytest.raises(RuntimeError):
        C = mod.forward(A, B)
        # Force synchronization to trigger any runtime error.
        torch.cuda.synchronize()

# Test 2: Non-contiguous input tensors
def test_non_contiguous_inputs():
    # Create contiguous lower triangular matrices first.
    N = 128
    A_base = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B_base = torch.randn(N, N, dtype=torch.float32, device="cuda")
    A_contig = torch.tril(A_base)
    B_contig = torch.tril(B_base)
    # Create non-contiguous tensors by performing a transpose, which makes them non-contiguous.
    A_noncontig = A_contig.t()  # Transpose will make it non-contiguous.
    B_noncontig = B_contig.t()
    mod = build_kernel()
    # Even though the data is lower-triangular after transpose, the kernel interprets the raw memory
    # assuming row-major contiguous layout. In a general situation this should yield wrong results.
    # We check that the kernel output does not match the expected reference.
    C_kernel = mod.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A_noncontig, B_noncontig))
    # The results are expected NOT to be close because of the misinterpretation of strides.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel unexpectedly produced correct result with non-contiguous inputs. "
        "It should require inputs to be contiguous."
    )

# Test 3: Batched input (3D tensor) which is not supported at all
def test_batched_input():
    # Create a batched lower triangular matrix. The kernel expects 2D inputs.
    BATCH = 4
    N = 64
    A = torch.tril(torch.randn(BATCH, N, N, dtype=torch.float32, device="cuda"))
    B = torch.tril(torch.randn(BATCH, N, N, dtype=torch.float32, device="cuda"))
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        C = mod.forward(A, B)
        torch.cuda.synchronize()
