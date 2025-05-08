
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_maxpool",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# A reference implementation using PyTorch's nn.MaxPool2d
def ref_maxpool2d(input, kernel_size, stride, padding, dilation):
    pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    return pool(input)

# Test 1: Passing a double tensor to trigger type mismatches (issues 1–3)
def test_double_precision():
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    batch_size = 4
    channels = 3
    height = 20
    width = 20

    # Create a double-precision tensor on CUDA
    input_tensor = torch.randn(batch_size, channels, height, width, dtype=torch.double, device="cuda")
    
    # Build and run the custom kernel
    mod = build_kernel()
    output_kernel = mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    
    # Get reference output (PyTorch's own max pooling works correctly for double)
    output_ref = ref_maxpool2d(input_tensor, kernel_size, stride, padding, dilation)
    
    # This test should trigger an issue because the kernel’s internal computations are in float,
    # so the output will likely differ from the reference.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), \
        "Double precision test did not trigger the type mismatch issue."

# Test 2: Passing a half-precision (float16) tensor to check unsupported datatype
def test_half_precision():
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    batch_size = 4
    channels = 3
    height = 20
    width = 20

    # Create a half-precision tensor on CUDA
    input_tensor = torch.randn(batch_size, channels, height, width, dtype=torch.float16, device="cuda")
    
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting the kernel to fail or throw an error because float16 is not supported.
        mod.forward(input_tensor, kernel_size, stride, padding, dilation)

# Test 3: Test with an input that causes the pooling window to only partially overlap valid regions.
# This will help ensure that the -INFINITY initialization (issue 3) and boundary conditions are handled.
def test_pooling_window_boundaries():
    kernel_size = 3  # Use a kernel that is not a divisor of the input dimensions.
    stride = 2
    padding = 2  # Large padding to force many window elements to be out-of-bound.
    dilation = 1
    batch_size = 2
    channels = 1
    height = 10
    width = 10

    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    
    mod = build_kernel()
    output_kernel = mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    output_ref = ref_maxpool2d(input_tensor, kernel_size, stride, padding, dilation)

    # The kernel might produce an incorrect output because of the improper -INFINITY handling.
    # We expect the outputs to not match exactly.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-4), \
        "Boundary test did not trigger an error: output from the custom kernel unexpectedly matches the reference."

if __name__ == '__main__':
    pytest.main([__file__])
