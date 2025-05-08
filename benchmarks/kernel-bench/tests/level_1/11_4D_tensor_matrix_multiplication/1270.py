
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="einsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Data type mismatch. The kernel only handles float32,
# so passing double (float64) inputs will cause wrong results.
def test_dtype_issue():
    cuda_module = build_kernel()
    b, i, j, l, k = 2, 4, 5, 3, 6
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float64)  # double precision
    B = torch.randn(l, k, device="cuda", dtype=torch.float64)  # double precision
    # Expected output computed in double precision via einsum
    C_expected = torch.einsum("bijl,lk->bijk", A, B)
    
    # The kernel cast pointers to float so it will compute in float32.
    # The result is expected to have a significant error.
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    C_kernel = cuda_module.forward(A_f, B_f)
    
    # Compare the kernel result (float32) with the expected result computed in float64 and then converted back.
    C_expected_f = C_expected.to(torch.float32)
    # The difference should be significant because of the type mismatch.
    diff = (C_kernel - C_expected_f).abs().max().item()
    assert diff > 1e-3, f"Data type issue not triggered: max diff {diff} is too small."

# Issue 2: Non-contiguous memory. The kernel expects contiguous tensors.
def test_non_contiguous_issue():
    cuda_module = build_kernel()
    b, i, j, l, k = 2, 4, 5, 3, 6
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    
    # Make A non-contiguous by transposing dimensions i and j and then transposing back.
    A_noncontig = A.transpose(1,2)
    # Now, transpose back to the expected shape. Its underlying memory is not contiguous.
    A_noncontig = A_noncontig.transpose(1,2)
    assert not A_noncontig.is_contiguous(), "A is expected to be non-contiguous to trigger the issue."
    
    # Reference computation using einsum (which works fine with non-contiguous tensors)
    C_expected = torch.einsum("bijl,lk->bijk", A_noncontig, B)
    
    # Launch kernel using non-contiguous A; errors in indexing will lead to a mismatch.
    C_kernel = cuda_module.forward(A_noncontig, B)
    torch.cuda.synchronize()
    diff = (C_kernel - C_expected).abs().max().item()
    # With non-contiguous memory the kernel computed result can be far off.
    assert diff > 1e-3, f"Non-contiguous issue not triggered: max diff {diff} is too small."

# Issue 3: Grid mapping inefficiency or grid dimension overflow.
# The kernel currently launches one block per output element (b * i * j * k).
# For very large tensors this may exceed grid dimension limits.
# For testing purposes we trigger an error by passing a tensor shape that leads 
# to an absurdly large number of blocks.
def test_grid_dimension_issue():
    cuda_module = build_kernel()
    # Choose dimensions such that the number of blocks becomes huge.
    # Instead of waiting for an actual CUDA grid launch failure, we use a moderately large k.
    b, i, j, l, k = 1, 1, 1, 2, 2**20  # k is very large, blocks = 1 * 1 * 1 * 2**20 = 1,048,576 blocks
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    
    try:
        C_kernel = cuda_module.forward(A, B)
        torch.cuda.synchronize()
    except RuntimeError as e:
        # We expect a CUDA launch failure for an excessive grid dimension.
        assert "grid" in str(e).lower(), f"Expected grid-dimension issue but got a different error: {e}"
    else:
        # If no exception was thrown, we check if the result is wildly incorrect
        # by comparing against the reference result computed with einsum.
        C_expected = torch.einsum("bijl,lk->bijk", A, B)
        diff = (C_kernel - C_expected).abs().max().item()
        assert diff > 1e-3, f"Grid dimension issue not triggered: max diff {diff} is too small."

if __name__ == "__main__":
    pytest.main([__file__])
