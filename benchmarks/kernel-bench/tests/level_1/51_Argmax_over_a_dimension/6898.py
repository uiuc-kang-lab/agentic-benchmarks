
import torch
import pytest
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="argmax_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function that mimics the PyTorch Model forward using our CUDA kernel.
def argmax_cuda(x: torch.Tensor, dim: int):
    mod = build_kernel()
    return mod.forward(x, dim)

# Issue 1: When all input elements are -infinity, PyTorch returns 0 as the index (first occurrence),
# but the kernel may return -1 due to un-updated thread local values.
def test_all_negative_infinity():
    # Create a tensor filled with -infinity (of type float32) on CUDA.
    # We'll choose a shape such that the argmax is applied over a dimension with more than one element.
    shape = (4, 10, 8)
    x = torch.full(shape, float("-inf"), device="cuda", dtype=torch.float32)
    # Expected: PyTorch's argmax returns 0 along the specified dimension.
    expected = torch.argmax(x, dim=1)
    # Use our CUDA kernel (which may erroneously return -1 for all positions)
    result = argmax_cuda(x, 1)
    torch.cuda.synchronize()
    # Check if any result equals -1. If so, our kernel is failing for issue 1.
    assert (result >= 0).all(), f"Kernel returned -1 indices for input all -infty: {result}"
    # Also test that the output matches PyTorch's behavior.
    assert torch.equal(result, expected), f"Kernel argmax result {result} does not match torch.argmax {expected}"

# Issue 2: Using float2 to store indices can lose precision for large indices.
# Create a tensor with a very large reduction dimension: the index value may exceed the precision of float.
def test_large_index_precision():
    # We simulate a case with a very large dimSize.
    # Although it is uncommon to have a dim of size > 2^24 (the precision limit of float),
    # we craft a tensor where the maximum is at the very end.
    # Note: Creating an enormous tensor is not practical,
    # so we simulate with a 1-by-N-by-1 tensor with N a large number that challenges float precision.
    N = 1 << 24  # 16777216 is around the limit for float precision.
    # We'll add a few extra indices to exceed the precision limit.
    N += 10
    # Create the tensor on CUDA. Use a shape that reduces over the middle dimension.
    # For memory reasons, we create a tensor with shape (1, N, 1)
    x = torch.zeros((1, N, 1), device="cuda", dtype=torch.float32)
    # Set the last element to a high value so that argmax should choose index N-1.
    x[0, -1, 0] = 1.0
    # Expected argmax index is N-1.
    expected = torch.argmax(x, dim=1)
    result = argmax_cuda(x, 1)
    torch.cuda.synchronize()
    # Because the kernel converts the index to float, precision might be lost.
    # We check whether the output index exactly matches, and fail if not.
    assert torch.equal(result, expected), f"Kernel argmax index {result.item()} does not match expected {expected.item()} for large index values."

# Issue 3: When the reduction dimension is smaller than the block size (256), many threads do not process any element,
# and their default value (-∞ with index -1) may interfere with the reduction.
def test_small_dim_vs_blocksize():
    # Create a tensor with a small dimension along which we compute argmax.
    # For example, let the reduction dimension have size 2 (which is much smaller than 256).
    shape = (8, 2, 5)  # argmax will run over dim=1.
    # Fill the tensor with random values, but ensure a unique maximum.
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    # Force the maximum to be in a known position (say, index 1) for each slice.
    x[:, 1, :] += 10.0
    expected = torch.argmax(x, dim=1)
    result = argmax_cuda(x, 1)
    torch.cuda.synchronize()
    # In a correct implementation, result should equal expected (which is all 1's).
    assert torch.equal(result, expected), f"Kernel argmax result {result} does not match expected {expected} for small reduction dimensions."

if __name__ == "__main__":
    pytest.main([__file__])
