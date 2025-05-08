
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load
import math

# Utility function to build the CUDA extension.
def build_kernel():
    module = load(
        name="maxpool_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test case 1: Trigger wrong kernel_size dispatch (kernel_size != 2 or 3).
# This test uses a kernel_size that is not explicitly supported, e.g. 4.
# The CUDA kernel will then run with KERNEL_SIZE==3 and produce an incorrect result.
def test_invalid_kernel_size():
    # Input parameters
    batch_size = 4
    channels = 3
    height = 20
    width = 20
    kernel_size = 4  # Not supported by our dispatch (should be 2 or treated as 3)
    stride = 2
    padding = 1
    dilation = 1

    device = "cuda"
    input_tensor = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)

    # Reference using PyTorch's built-in function:
    ref_output = F.max_pool2d(input_tensor, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

    # Call our CUDA kernel from the extension.
    kernel_module = build_kernel()
    output = kernel_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # Check that the output is NOT equal to the correct output, indicating that the kernel
    # did not handle kernel_size=4 properly.
    # We expect this test to fail (i.e. intentionally produce a difference).
    diff = (ref_output - output).abs().max().item()
    assert diff > 1e-3, f"Test failed to trigger kernel size issue, difference is {diff}"

# Test case 2: Trigger the double precision issue.
# When the input tensor is double, fmaxf should not be used.
def test_double_precision_input():
    # Input parameters (using kernel_size==2 which is supported, but double precision exposed)
    batch_size = 4
    channels = 3
    height = 20
    width = 20
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    device = "cuda"
    input_tensor = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.double)

    # Reference using PyTorch's built-in function:
    ref_output = F.max_pool2d(input_tensor, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

    # Call our CUDA kernel from the extension.
    kernel_module = build_kernel()
    output = kernel_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # Given the misuse of fmaxf for double, we expect the output to be incorrect.
    max_diff = (ref_output - output).abs().max().item()
    assert max_diff > 1e-3, f"Test failed to trigger double precision issue, max difference: {max_diff}"

if __name__ == "__main__":
    pytest.main([__file__])
