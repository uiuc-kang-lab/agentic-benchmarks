
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to compile/load the CUDA kernel from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Non-contiguous input tensor.
# This test creates an input that is non-contiguous by transposing it.
# Because the kernel ignores tensor strides, the computed reduction result will differ
# from the expected torch.sum result.
def test_non_contiguous():
    # Create a contiguous tensor
    x = torch.randn(8, 16, 32, device="cuda", dtype=torch.float32)
    # Make it non-contiguous by transposing two dimensions
    x_non_contig = x.transpose(1, 2)  # New shape: (8, 32, 16) and non-contiguous
    # Choose reduction dimension such that the layout assumption is broken.
    reduce_dim = 2
    kernel_module = build_kernel()
    # Call the custom CUDA kernel via its bound name ("forward")
    out_kernel = kernel_module.forward(x_non_contig, reduce_dim)
    # Compute the expected result using PyTorch's native sum
    out_ref = torch.sum(x_non_contig, dim=reduce_dim, keepdim=True)
    # The results should differ because the kernel wrongly computes indices.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), (
        "Test failed: Non-contiguous input produced a result that accidentally matched torch.sum."
    )

# Test 2: Input tensor with non-standard layout (permuted dimensions).
# This test creates an input tensor with a permutation of dimensions so that
# the “reduction” dimension is not laid out contiguously.
def test_non_standard_layout():
    # Create a tensor with shape (4, 5, 6, 7)
    x = torch.randn(4, 5, 6, 7, device="cuda", dtype=torch.float32)
    # Permute the dimensions so that the intended reduction dimension becomes non-contiguous.
    # For instance, assume we want to reduce over what was originally dim=1.
    x_permuted = x.permute(0, 2, 1, 3)  # New shape: (4, 6, 5, 7)
    # If we reduce along dim=2 (which was originally dim=1), the kernel will compute indices
    # under the assumption of contiguous outer/reduce/inner layout.
    reduce_dim = 2
    kernel_module = build_kernel()
    out_kernel = kernel_module.forward(x_permuted, reduce_dim)
    out_ref = torch.sum(x_permuted, dim=reduce_dim, keepdim=True)
    # The outputs should differ because the kernel ignores the non-standard memory layout.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), (
        "Test failed: Permuted (non-standard layout) input produced a result that accidentally matched torch.sum."
    )

if __name__ == "__main__":
    pytest.main([__file__])
