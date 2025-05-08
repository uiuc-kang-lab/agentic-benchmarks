
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

# Utility function to compile the CUDA extension from kernel.cu.
def build_kernel():
    # Ensure we are in a directory where kernel.cu exists.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(current_dir, "kernel.cu")
    cuda_module = load(
        name="gelu_kernel_module",
        sources=[kernel_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger the tail elements issue.
# Provide an input tensor whose total number of elements is not divisible by VEC_SIZE (4)
# and compare the kernel output with PyTorch's GELU. Expect differences due to out-of-bound accesses.
def test_tail_elements_issue():
    my_module = build_kernel()
    # Total elements not divisible by 4, e.g., 10 elements.
    n = 10
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    # Invoke the CUDA kernel
    y_kernel = my_module.forward(x)
    torch.cuda.synchronize()
    # Compute reference GELU using PyTorch
    y_ref = F.gelu(x)
    # Due to the vector loop, some tail elements might be accessed out-of-bound or computed incorrectly.
    # We expect the outputs to NOT be equal.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Test failed: Kernel output matches reference even though input size is not a multiple of VEC_SIZE."
    )

# Test case 2: Trigger the grid configuration issue.
# Use a large tensor where the 2D grid mapping might lead to an incorrect global index calculation.
def test_grid_configuration_issue():
    my_module = build_kernel()
    # Choose a tensor size which stresses the grid mapping.
    # For instance, a tensor with a large number of elements but not a perfect multiple of block dimensions.
    n = 12345  # Arbitrary number that's not aligned with typical block sizes and vectorization.
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y_kernel = my_module.forward(x)
    torch.cuda.synchronize()
    y_ref = F.gelu(x)
    # If the grid configuration is incorrect, the output will be off.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Test failed: Kernel output is correct despite likely grid configuration mapping issues."
    )

# Test case 3: Test behavior with double precision inputs.
# This tests whether the kernel mapping and boundary handling work correctly with double data type.
def test_double_precision_issue():
    my_module = build_kernel()
    # Choose n such that a tail exists (i.e., n not divisible by vector size 4)
    n = 17
    x = torch.randn(n, device="cuda", dtype=torch.double)
    y_kernel = my_module.forward(x)
    torch.cuda.synchronize()
    y_ref = F.gelu(x)
    # Because of the branching flaw, the tail elements in double precision might be computed incorrectly.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-7), (
        "Test failed: Double precision kernel output matches GELU reference, possibly indicating no tail issue."
    )
