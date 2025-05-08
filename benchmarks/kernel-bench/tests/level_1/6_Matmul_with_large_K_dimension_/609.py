
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build the extension
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

@pytest.fixture(scope="module")
def cuda_module():
    mod = build_kernel()
    return mod

# Issue 1: Batched input not supported
def test_batched_input(cuda_module):
    M, N, K, BATCH = 128, 64, 256, 4
    # Create batched tensors (3D) although the kernel expects 2D tensors.
    A = torch.randn(BATCH, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(BATCH, K, N, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expect an error due to shape mismatch because our host module_fn only supports 2D matrices.
        _ = cuda_module.forward(A, B)

# Issue 2: Non‐contiguous input tensor
def test_non_contiguous_input(cuda_module):
    M, N, K = 128, 64, 256
    # Create a bigger tensor, then form a non-contiguous view for A and B.
    # as_strided can be used to create a view with different strides.
    A_base = torch.randn(M, K * 2, device="cuda", dtype=torch.float32)
    B_base = torch.randn(K, N * 2, device="cuda", dtype=torch.float32)
    # Create a noncontiguous view with the same shape but with inflated strides.
    A_noncontig = A_base.as_strided((M, K), (A_base.stride(0), A_base.stride(1) * 2))
    B_noncontig = B_base.as_strided((K, N), (B_base.stride(0), B_base.stride(1) * 2))
    # We expect that the kernel (which uses data_ptr() and __ldg loads) will produce a result
    # that is different from torch.matmul which works on contiguous tensors.
    C_kernel = cuda_module.forward(A_noncontig, B_noncontig)
    # Use contiguous versions for reference
    C_ref = torch.matmul(A_noncontig.contiguous(), B_noncontig.contiguous())
    # Here we expect a discrepancy because the kernel did not account for non-contiguity.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), \
        f"Kernel output should differ when using non-contiguous inputs."

# Issue 3: Lack of proper kernel launch error checking
def test_mismatched_dimensions(cuda_module):
    # Provide matrices with mismatched inner dimensions
    M, N, K = 128, 64, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # Create B with an incompatible shape (inner dim mismatch)
    B = torch.randn(K + 1, N, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError) as excinfo:
        _ = cuda_module.forward(A, B)
    # Check that the error message contains the message from TORCH_CHECK about inner dimension.
    assert "Inner dimensions" in str(excinfo.value), "Expected mismatch dimension error not raised."
