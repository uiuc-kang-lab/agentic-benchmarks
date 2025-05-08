
import torch
import pytest
from torch.utils.cpp_extension import load

# Function to build the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that non-float32 inputs lead to incorrect behavior.
def test_non_float_input():
    # Create double precision tensors (float64)
    batch_size, M, K, N = 4, 8, 16, 8
    A = torch.randn(batch_size, M, K, device='cuda', dtype=torch.float64)
    B = torch.randn(batch_size, K, N, device='cuda', dtype=torch.float64)
    module = build_kernel()
    # Call the kernel with double inputs.
    # Since the kernel forces float32 access, the output will be garbage.
    with pytest.raises(RuntimeError):
        # If the kernel or the TORCH_CHECK in C++ does not catch this,
        # then the comparison with torch.bmm (after cast) will fail.
        C = module.forward(A, B)
        torch.cuda.synchronize()
        # Produce a reference result using float64 matmul.
        C_ref = torch.bmm(A, B)
        # The results are expected to differ significantly.
        # Here, we force an error if they are (erroneously) close.
        assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly produced correct result with non-float32 input."

# Issue 2: Test that non-contiguous input tensors produce inconsistent results.
def test_non_contiguous_input():
    # Create contiguous input tensors first.
    batch_size, M, K, N = 4, 8, 16, 8
    A_contig = torch.randn(batch_size, M, K, device='cuda', dtype=torch.float32)
    B_contig = torch.randn(batch_size, K, N, device='cuda', dtype=torch.float32)
    # Make them non-contiguous by transposing the last two dims.
    A_noncontig = A_contig.transpose(1, 2)
    B_noncontig = B_contig.transpose(1, 2)
    # However, the kernel expects inputs of shape (batch_size, M, K) and (batch_size, K, N)
    # So we artificially make them non-contiguous while preserving shape by using .clone() after a transpose.
    A_noncontig = A_noncontig.transpose(1, 2)  # now shape is restored to (batch_size, M, K) but memory is non-contiguous.
    B_noncontig = B_noncontig.transpose(1, 2)  # now shape is (batch_size, K, N)
    
    # Verify that they are indeed non-contiguous.
    assert not A_noncontig.is_contiguous()
    assert not B_noncontig.is_contiguous()
    
    module = build_kernel()
    # Compute kernel result
    C = module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    # Compute reference result by forcing contiguous tensors.
    C_ref = torch.bmm(A_noncontig.contiguous(), B_noncontig.contiguous())
    
    # We expect the kernel not to properly support non-contiguous layouts,
    # so the result should differ.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel produced the same result with non-contiguous input, which is unexpected."

# Issue 3: Test that missing error checking in kernel launch can lead to silent errors.
# For example, providing tensors with incorrect dimensions should trigger TORCH_CHECK in C++.
def test_incorrect_tensor_dimensions():
    # Create a 2D tensor instead of 3D.
    A = torch.randn(8, 16, device='cuda', dtype=torch.float32)
    B = torch.randn(16, 8, device='cuda', dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the C++ function should raise an error.
        module.forward(A, B)

if __name__ == "__main__":
    pytest.main([__file__])
