
import pytest
import torch
from torch.utils.cpp_extension import load

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

# Issue 1: Mismatched inner dimensions.
# Expected behavior: torch.matmul should raise an error for mismatched dimensions,
# but our custom kernel does not perform any check and will produce some output tensor.
def test_mismatched_inner_dimensions():
    my_module = build_kernel()
    # Create A with shape [N, M, K]
    A = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)
    # Create B with mismatched inner dimension: B should have shape [K, L] with K==4,
    # so provide a tensor with a different first dimension.
    B = torch.randn(5, 6, device="cuda", dtype=torch.float32)
    
    # torch.matmul on these mismatched shapes should raise a RuntimeError.
    with pytest.raises(RuntimeError):
        torch.matmul(A, B)
    
    # However, the custom kernel is not checking dimensions and will attempt a multiplication.
    # This test verifies that the kernel produces an output shape that does not reflect a proper matmul.
    C = my_module.forward(A, B)
    # Expected output shape from the kernel: [N, M, L] where L is taken from B.size(1).
    # Because K mismatches, the kernel uses K taken from A and ignores the extra row in B.
    assert C.shape == (2, 3, 6), "Output tensor shape is not as expected even though inner dimensions mismatch."

# Issue 2: Non-contiguous input tensors.
# Expected behavior: The helper CHECK_INPUT macros should trigger an error when non-contiguous tensors are passed.
def test_non_contiguous_input():
    my_module = build_kernel()
    A = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
    B = torch.randn(16, 10, device="cuda", dtype=torch.float32)
    # Create non-contiguous versions by transposing some dimensions
    A_nc = A.transpose(1, 2)
    B_nc = B.t()
    
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(A_nc, B)
    assert "must be contiguous" in str(excinfo.value)
    
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(A, B_nc)
    assert "must be contiguous" in str(excinfo.value)

# Issue 3: Unsupported (non-floating point) dtype.
# Expected behavior: The AT_DISPATCH macro used in the kernel only handles floating types and half.
# Passing an integer tensor should cause an error.
def test_non_floating_dtype():
    my_module = build_kernel()
    A = torch.randint(0, 10, (2, 3, 4), device="cuda", dtype=torch.int32)
    B = torch.randint(0, 10, (4, 5), device="cuda", dtype=torch.int32)
    
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)

if __name__ == "__main__":
    pytest.main([__file__])
