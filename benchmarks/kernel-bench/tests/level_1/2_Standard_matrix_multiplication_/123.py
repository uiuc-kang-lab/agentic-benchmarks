
import pytest
import torch
from torch.utils.cpp_extension import load

# Build/load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that a non-float32 input results in incorrect behavior.
def test_dtype_issue():
    cuda_module = build_kernel()
    M, K, N = 128, 64, 32
    # Create inputs as double (float64)
    A = torch.randn(M, K, device="cuda", dtype=torch.float64)
    B = torch.randn(K, N, device="cuda", dtype=torch.float64)
    # Run the kernel function (it will cast pointer to float*, misinterpreting data)
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result in float64
    C_ref = torch.matmul(A, B)
    # The kernel result is computed using the wrong type so it should differ significantly.
    assert not torch.allclose(C.to(torch.float64), C_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct result with non-float32 data!"

# Issue 2: Test that a dimension mismatch (A.size(1) != B.size(0)) is not caught by the kernel.
def test_dimension_mismatch_issue():
    cuda_module = build_kernel()
    # Make A and B with mismatched inner dimensions.
    A = torch.randn(64, 50, device="cuda", dtype=torch.float32)
    B = torch.randn(60, 32, device="cuda", dtype=torch.float32)  # 60 != 50
    # The kernel uses A.size(1) as the 'inner dimension', so it will use 50 even though B's "row"
    # dimension is 60. This should lead to an incorrect result.
    # Note: torch.matmul would normally raise an error.
    with pytest.raises(RuntimeError):
        # In PyTorch, calling torch.matmul on incompatible shapes should error.
        _ = torch.matmul(A, B)
    # However, our kernel function does not perform this check.
    # So calling our kernel will succeed, but will produce a wrong result.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # Verify that the output shape is computed from A and B (using B.size(1)),
    # but the numerical result is meaningless.
    assert C.shape == (64, 32), "Output shape is not as expected from kernel computation."

# Issue 3: Test that the kernel does not check for errors after launch.
# We simulate a launch error by providing a non-contiguous tensor.
def test_non_contiguous_issue():
    cuda_module = build_kernel()
    M, K, N = 128, 64, 32
    A_full = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_full = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Create a non-contiguous tensor by transposing.
    A = A_full.t()
    B = B_full.t()
    # Check that the tensors are indeed non-contiguous.
    assert not A.is_contiguous(), "A should be non-contiguous for this test."
    assert not B.is_contiguous(), "B should be non-contiguous for this test."
    # The kernel CHECK_INPUT macro expects contiguous CUDA tensors and should throw an exception.
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(A, B)
