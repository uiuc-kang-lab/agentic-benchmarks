
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    cuda_module = load(
        name="maxpool_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger the issue with the device max function usage.
# The kernel’s incorrect use of 'max' may yield wrong output.
def test_incorrect_max_computation():
    module = build_kernel()
    # Create an input tensor with negative values such that the pooling window has only negatives.
    # With proper max pooling the maximum among negatives should be returned.
    # However, if the bug causes the initialization value (-infinity) to be improperly handled,
    # the result might remain -infinity.
    x = -torch.abs(torch.randn(1, 1, 8, 8, dtype=torch.float32, device='cuda'))
    
    # Compute expected output using PyTorch's built-in max_pool2d for the given parameters.
    expected = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=1, dilation=3)
    kernel_out = module.forward(x, 2, 2, 1, 3)
    torch.cuda.synchronize()
    
    # If the CUDA kernel's max function is not working correctly, the results will be off.
    # We expect the output to _not_ match the PyTorch correct output.
    assert not torch.allclose(expected, kernel_out, atol=1e-5), (
        "Kernel output unexpectedly matches the correct output. Issue in the device max function might be unresolved."
    )

# Test case 2: Trigger issue with negative infinity initialization.
# Here we pass a tensor where the correct maximum using negative initialization should be finite,
# but due to potential problems with std::numeric_limits usage in device code,
# the kernel may produce an incorrect value.
def test_incorrect_negative_infinity_initialization():
    module = build_kernel()
    # Construct a tensor with a varied range so that the max within each pooling window is finite.
    # If the negative infinity initialization fails, the pooling result may remain -infinity in some windows.
    x = torch.linspace(-100, 100, steps=64, device='cuda', dtype=torch.float32).reshape(1, 1, 8, 8)
    
    # Expected output from PyTorch's implementation.
    expected = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=1, dilation=3)
    kernel_out = module.forward(x, 2, 2, 1, 3)
    torch.cuda.synchronize()
    
    # Check if any values in the kernel output remain at -infinity.
    assert (kernel_out == float("-inf")).sum() > 0, (
        "No -infinity values were observed in the kernel output even though incorrect negative infinity initialization is expected."
    )
