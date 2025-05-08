
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu.
def build_kernel():
    module = load(
        name="test_argmax",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Test case #1: Passing a non-float32 tensor should trigger the type check failure.
def test_non_float_input():
    cuda_module = build_kernel()
    # Create a tensor of type float64
    x = torch.randn(4, 5, 6, dtype=torch.double, device="cuda")
    # Attempt to call the CUDA kernel which only supports float32.
    with pytest.raises(RuntimeError, match="Only float32 is supported"):
        # The module function is "forward"; it should trigger an error.
        _ = cuda_module.forward(x, 1)

# Test case #2: Using a tensor whose reduction dimension is smaller than block size.
# The kernel reduction assumes there are 128 threads working, but if dimSize (the size of
# the argmax dimension) is very small (e.g. 2), many threads will not perform any valid computation,
# and the reduction may incorrectly pick the result from an uninitialized thread.
def test_small_dimension_issue():
    cuda_module = build_kernel()
    # Create a tensor where the argmax is computed along dim=1 and its size is smaller than 128.
    # Construct a tensor in which the maximum is in the second position.
    # For example, tensor shape [batch, 2, other] where batch and other size are arbitrary.
    batch = 4
    other = 3
    # We ensure that the max value in dim 1 is at index 1.
    x = torch.empty(batch, 2, other, dtype=torch.float32, device="cuda")
    # Fill with a value that is lower than our intended maximum.
    x.fill_(-10.0)
    # Set the second slice to a higher value.
    x[:, 1, :] = 5.0
    # Call the CUDA kernel over dimension 1.
    indices = cuda_module.forward(x, 1)
    # The expected argmax indices along dimension 1 is 1 for every element.
    expected = torch.ones((batch, other), dtype=torch.int64, device="cuda")
    # This test is designed to trigger the reduction issue due to block configuration assumptions.
    assert torch.equal(indices, expected), f"Expected {expected}, but got {indices}"

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
