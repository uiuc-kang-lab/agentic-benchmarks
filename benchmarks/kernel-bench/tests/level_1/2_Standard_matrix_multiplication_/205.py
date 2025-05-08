
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1:
# When the matrix dimensions are not multiples of TILE_DIM, the improper boundary handling
# (clamping instead of zero‐padding) causes the wrong accumulation in the dot product.
def test_non_multiple_tile():
    my_module = build_kernel()
    # Create matrices with dimensions that are not exact multiples of 32.
    # For example: A (35 x 45) and B (45 x 33)
    M, K, N = 35, 45, 33
    A = torch.randn(M, K, device='cuda', dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device='cuda', dtype=torch.float32).contiguous()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # The output will be wrong when boundaries are improperly handled.
    # Hence we check that the result is NOT close to the reference.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Output should be incorrect due to boundary handling issues, but it matches reference output."
    )

# Issue 2:
# The kernel only supports float32. When passing a tensor of a different dtype (e.g. float64),
# the kernel will incorrectly cast pointer types.
def test_input_dtype():
    my_module = build_kernel()
    N = 64
    # Use float64 tensors.
    A = torch.randn(N, N, device='cuda', dtype=torch.float64)
    B = torch.randn(N, N, device='cuda', dtype=torch.float64)
    # Although CHECK_INPUT only tests for CUDA and contiguity, the data_ptr conversion is fixed to float,
    # so the result is expected to be wrong.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Kernel should fail (or produce wrong results) for non-float32 tensors but produced a matching output."
    )

# Issue 3:
# The kernel does not validate that the inner dimensions match.
# Passing mismatched inner dimensions should lead either to an exception or to incorrect/unpredictable results.
def test_dimension_mismatch():
    my_module = build_kernel()
    # A is (20 x 30) but B is (31 x 40) so the inner dimensions do not match.
    A = torch.randn(20, 30, device='cuda', dtype=torch.float32).contiguous()
    B = torch.randn(31, 40, device='cuda', dtype=torch.float32).contiguous()
    with pytest.raises(RuntimeError):
        # One might expect the kernel launch or underlying memory accesses to raise an error.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 4:
# The kernel launch does not have an associated error check.
# One way to trigger a runtime error is to supply non‐contiguous tensors.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    N = 64
    # Create contiguous tensors first and then get a non-contiguous view.
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    A_noncontig = A.t()  # transpose returns a non-contiguous view
    with pytest.raises(RuntimeError):
        C = my_module.forward(A_noncontig, B)
        torch.cuda.synchronize()
