
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

# Helper: Reference computation for upper triangular matrix multiplication.
def reference_upper_triangular(A, B):
    # Compute full matmul then take upper triangular part.
    return torch.triu(torch.matmul(A, B))

# Issue 1: Input tensor type. Kernel only supports float32 but not float64.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_input_tensor_type():
    N = 128
    # Create upper triangular matrices with double precision
    # Note: torch.triu returns a tensor with same dtype.
    A = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float64))
    B = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float64))
    module = build_kernel()
    # This call will lead to undefined behavior because the kernel
    # is compiled for float. We expect the output to differ from the reference.
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = reference_upper_triangular(A, B)
    # We expect the result NOT to be close because wrong type interpretation occurs.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        f"Kernel incorrectly handled double type: got same result as reference, C.max()={C.max()}"
    )

# Issue 2: Non-contiguous input tensors.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_input():
    N = 128
    A_full = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B_full = torch.randn(N, N, device='cuda', dtype=torch.float32)
    # Create non-contiguous upper triangular matrices by transposing a contiguous triu tensor.
    A = torch.triu(A_full).t()  # Transpose produces a non-contiguous tensor.
    B = torch.triu(B_full).t()
    # Even though A.t() is still square, its strides are changed.
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = reference_upper_triangular(A, B)
    # Since the kernel assumes contiguous storage with stride N, the results will be wrong.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        f"Kernel incorrectly handled non-contiguous memory: max difference {(C-C_ref).abs().max()}"
    )

# Issue 3: Non-standard stride (subtensor from a larger tensor).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_standard_stride():
    N = 128
    # Create a larger tensor and then slice a submatrix to get non-standard strides.
    big_tensor = torch.randn(N+10, N+10, device='cuda', dtype=torch.float32)
    # Extract a submatrix that is not starting at (0,0)
    A = torch.triu(big_tensor[5:5+N, 5:5+N])
    B = torch.triu(big_tensor[5:5+N, 5:5+N])
    # Ensure the submatrices are non-contiguous.
    assert not A.is_contiguous(), "Test setup error: A is contiguous but should not be."
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = reference_upper_triangular(A, B)
    # Since the kernel indexes assuming leading dimension equals N and contiguous layout,
    # the results should be incorrect.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        f"Kernel incorrectly handled non-standard stride: max difference {(C-C_ref).abs().max()}"
    )
