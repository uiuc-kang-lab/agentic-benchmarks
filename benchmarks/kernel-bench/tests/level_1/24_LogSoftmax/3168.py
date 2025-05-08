
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper function to build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module


# Test case 1: Check for incorrect accumulation in the reduction.
# This test creates an input tensor and compares the output of the CUDA kernel with PyTorch's log_softmax.
# Due to the bug in shared memory initialization, the CUDA kernel's log_softmax result will differ significantly.
def test_incorrect_shared_memory_reduction():
    # Create an input tensor with nonzero maximum to force a non-negligible block-max.
    batch_size = 8
    dim = 128  # use a dim_size that selects one of the block sizes (e.g. 128)
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    
    # Permute dimensions similar to what the kernel host code does.
    # For simplicity we assume the reduction dim is -1.
    my_kernel = build_kernel()
    # Launch kernel using our forward function in the extension.
    output_kernel = my_kernel.forward(x, -1)
    # Compute reference using PyTorch's log_softmax.
    output_ref = torch.log_softmax(x, dim=-1)
    
    # If the accumulation bug is present, the difference (max absolute error) will be large.
    max_err = (output_kernel - output_ref).abs().max().item()
    # We set a relatively low tolerance for a correct implementation.
    tolerance = 1e-5
    # Expect the error to be larger than tolerance to detect the bug.
    assert max_err > tolerance, (
        f"Test failed to trigger the shared memory accumulation issue: max error {max_err} is not above tolerance {tolerance}"
    )


# Test case 2: Check that the usage of the unqualified max function causes compilation/device problems.
# This test deliberately uses inputs (and types) that cause the ambiguous call to max to be instantiated.
# For instance, we create a tensor with double precision and expect that if the kernel does not properly qualify max,
# a compilation or runtime error might occur or the result will be erroneous.
def test_unqualified_max_function():
    # Create a double tensor to force instantiation for kFloat64.
    batch_size = 4
    dim = 256  # choose a block size where kernel launches with 256 threads
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float64)
    
    try:
        my_kernel = build_kernel()
        output_kernel = my_kernel.forward(x, -1)
    except Exception as e:
        pytest.skip(f"Kernel failed to launch due to unqualified max function: {e}")
    
    output_ref = torch.log_softmax(x, dim=-1)
    # Check whether the output from the kernel is close to the expected values.
    # If the use of max is wrong, then even if compilation passed, the kernel might show large differences.
    max_err = (output_kernel - output_ref).abs().max().item()
    tolerance = 1e-5
    assert max_err > tolerance, (
        f"Test failed to trigger the max function issue in device code: max error {max_err} is unexpectedly low."
    )
