
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build/load the CUDA extension from kernel.cu
def build_kernel():
    module = load(
        name="maxpool_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Utility function: a reference maxpool2d using PyTorch's native operator
def ref_maxpool2d(x, kernel_size, stride, padding, dilation):
    return torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, 
                                            padding=padding, dilation=dilation)

# Test case to trigger issue 1:
# Providing a kernel_size > MAX_KERNEL_SIZE should result in output not correctly computed
def test_kernel_size_exceeds_max():
    # Use a kernel size greater than MAX_KERNEL_SIZE (which is 7)
    kernel_size = 8
    stride = 2
    padding = 1
    dilation = 1

    batch_size = 2
    channels = 3
    height = 32
    width = 32

    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    # Load custom kernel
    maxpool_module = build_kernel()

    # Call the CUDA kernel function; it is expected to behave incorrectly,
    # i.e. leave output elements as -infinity.
    y_custom = maxpool_module.forward(x, kernel_size, stride, padding, dilation)

    # Calculate the expected output with PyTorch's native max_pool2d.
    y_ref = ref_maxpool2d(x, kernel_size, stride, padding, dilation)
    
    # Check if the output is different from the reference which is the issue manifestation.
    # We expect the custom kernel to produce incorrect results.
    assert not torch.allclose(y_custom, y_ref, atol=1e-4), \
        f"Custom kernel erroneously produced correct result for kernel_size > MAX_KERNEL_SIZE. y_custom: {y_custom}, y_ref: {y_ref}"

# Test case to trigger issue 2:
# Providing an input tensor with a non-floating type should use AT_DISPATCH_FLOATING_TYPES,
# but since the kernel uses an unqualified max, it may compile or run incorrectly.
def test_non_floating_input():
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    batch_size = 2
    channels = 3
    height = 16
    width = 16

    # Create an integer tensor which is not a floating type.
    x = torch.randint(low=0, high=10, size=(batch_size, channels, height, width), device="cuda", dtype=torch.int32)

    maxpool_module = build_kernel()
    
    # Expect an exception as the kernel dispatch expects floating point inputs.
    with pytest.raises(Exception):
        y_custom = maxpool_module.forward(x, kernel_size, stride, padding, dilation)
        # Attempt a synchronization to force any runtime errors.
        torch.cuda.synchronize()

if __name__ == "__main__":
    pytest.main([os.path.realpath(__file__)])
