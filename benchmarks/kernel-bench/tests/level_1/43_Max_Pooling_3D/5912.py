
import math
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Utility to build and load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="max_pool3d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Passing a non-floating type tensor should trigger an issue
def test_non_floating_input():
    cuda_module = build_kernel()
    # Create an integer tensor - the kernel dispatch is only for floating types.
    x = torch.randint(0, 10, (2, 2, 10, 10, 10), dtype=torch.int32, device="cuda")
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False

    with pytest.raises(RuntimeError):
        # Expect a runtime error because the extension dispatch macro won't find a matching type.
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

# Test 2: Using non-contiguous input should reveal incorrect results.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create a contiguous input tensor.
    x = torch.randn(2, 2, 16, 16, 16, device="cuda", dtype=torch.float32)
    # Make a non-contiguous version (by transposing two spatial dimensions).
    x_noncontig = x.transpose(3, 4)
    # Ensure it's actually non-contiguous.
    assert not x_noncontig.is_contiguous(), "Test input must be non-contiguous."

    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False

    # Use PyTorch's own max_pool3d as a reference with contiguous input.
    ref_out = F.max_pool3d(x_noncontig, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)
    # Call the CUDA kernel.
    test_out = cuda_module.forward(x_noncontig, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    # If the kernel assumes contiguity, results may differ.
    assert not torch.allclose(test_out, ref_out, atol=1e-4), "CUDA kernel did not expose error with non-contiguous input!"

# Test 3: Passing non-symmetric parameters (as tuples) should trigger an error.
def test_non_symmetric_parameters():
    cuda_module = build_kernel()
    # Create a valid input tensor.
    x = torch.randn(2, 2, 16, 16, 16, device="cuda", dtype=torch.float32)
    # Intentionally pass tuples rather than ints.
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    return_indices = False
    ceil_mode = False

    with pytest.raises(TypeError):
        # The CUDA kernel's C++ interface expects ints; thus, passing tuples should raise an error.
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
