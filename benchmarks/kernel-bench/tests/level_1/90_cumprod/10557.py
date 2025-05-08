
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="cumprod_cuda_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Utility function for cumulative product reference computation using torch.cumprod
def ref_cumprod(input, dim):
    return torch.cumprod(input, dim=dim)

# Test case 1: Non‐contiguous input tensor
def test_non_contiguous_input():
    # Create a tensor and make it non-contiguous by transposing
    # For example, use a 2D tensor where cumprod is on dim=1 (inner dimension in contiguous layout)
    orig = torch.randn(128, 4000, device="cuda")
    # Transpose makes the tensor non-contiguous; now original cumprod dim (say dim=1) 
    # is not contiguous in memory.
    input = orig.transpose(0, 1)
    dim = 1  # cumulative product along dim=1, but tensor is non-contiguous.
    kernel = build_kernel()
    
    output = kernel.forward(input, dim)
    expected = ref_cumprod(input, dim)
    torch.cuda.synchronize()
    # This test is expected to fail because the kernel indexing assumes contiguity.
    assert not torch.allclose(output, expected), "Test expected a failure due to non-contiguous input but got matching results."

# Test case 2: Dimension size not divisible by 4
def test_dim_size_not_divisible_by_4():
    # Create a contiguous tensor where the cumprod dimension size is not a multiple of 4.
    # For example, shape (64, 7) where 7 is not divisible by 4.
    input = torch.randn(64, 7, device="cuda")
    dim = 1
    kernel = build_kernel()
    
    output = kernel.forward(input, dim)
    expected = ref_cumprod(input, dim)
    torch.cuda.synchronize()
    # This test is expected to fail (or produce incorrect output) due to the unrolling assumption.
    assert not torch.allclose(output, expected, atol=1e-5), "Test expected a failure due to non-multiple-of-4 dimension but got matching results."

# Test case 3: Cumulative product computed along a non-innermost dimension
def test_non_inner_dimension():
    # Create a contiguous tensor of shape (10, 20, 30) and compute cumprod along dim=1 (not the innermost).
    input = torch.randn(10, 20, 30, device="cuda")
    dim = 1  # Not the innermost dimension; the kernel indexing incorrectly assumes the cumprod dim is contiguous.
    kernel = build_kernel()
    
    output = kernel.forward(input, dim)
    expected = ref_cumprod(input, dim)
    torch.cuda.synchronize()
    # This test is expected to expose indexing errors due to the kernel’s assumption.
    assert not torch.allclose(output, expected, atol=1e-5), "Test expected a failure when cumprod is computed along a non-innermost dimension but got matching results."
    
if __name__ == "__main__":
    pytest.main([__file__])
