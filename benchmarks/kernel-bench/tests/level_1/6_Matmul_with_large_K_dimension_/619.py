
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Create inputs that are non-contiguous by transposing after creation.
    M, K, N = 64, 128, 32
    # Create a contiguous version then transpose to break contiguity.
    A_contig = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_contig = torch.randn(K, N, device="cuda", dtype=torch.float32)
    A = A_contig.t()  # now shape becomes (K, M) and non-contiguous
    B = B_contig.t()  # now shape becomes (N, K) and non-contiguous

    # Fix the shapes back to the expected ones, but still non-contiguous.
    A = A.t()
    B = B.t()

    # Confirm non-contiguity
    assert not A.is_contiguous(), "A is contiguous while it should be non-contiguous for this test"
    assert not B.is_contiguous(), "B is contiguous while it should be non-contiguous for this test"

    module = build_kernel()
    # The kernel does not perform any check to ensure that these tensors are contiguous.
    # As a result, the computations will use incorrect strides.
    C_kernel = module.forward(A, B)
    torch.cuda.synchronize()
    # Use contiguous copies to compute the correct result.
    C_ref = torch.matmul(A.contiguous(), B.contiguous())
    # The results are expected to differ because the kernel assumed row-major contiguous layout.
    # In this test we assert that the outputs are NOT all close.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel output unexpectedly matches the reference despite non-contiguous inputs; "
        "this indicates the kernel is not catching non-contiguous memory issues."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batched_input():
    # Create batched inputs which are not supported by the kernel.
    batch, M, K, N = 4, 32, 64, 16
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, N, device="cuda", dtype=torch.float32)
    module = build_kernel()
    # The kernel is designed for 2D matrices.
    # Passing batched inputs should lead to either an error or an incorrect result.
    with pytest.raises(Exception):
        # We wrap the call in an exception check to trigger the issue.
        _ = module.forward(A, B)
