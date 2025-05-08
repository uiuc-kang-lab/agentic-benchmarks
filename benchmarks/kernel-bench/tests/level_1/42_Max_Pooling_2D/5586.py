
import torch
import pytest
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Utility function to build the CUDA kernel module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="max_pool2d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Define parameters common to tests.
batch_size = 2
channels = 3
height = 16
width = 16
kernel_size = 2
stride = 2
padding = 1
dilation = 1  # simple dilation for tests

# Test 1:
# Trigger issue with device incompatibility of std::numeric_limits and std::max.
# Using double precision should force the kernel to try to use these C++ functions 
# in a device context where they are not available or work incorrectly.
def test_issue_std_functions_double():
    device = "cuda"
    my_module = build_kernel()
    
    # Create a double tensor (float64) which is dispatched by AT_DISPATCH_FLOATING_TYPES.
    # If the device code fails to handle std::numeric_limits or std::max, the result
    # will be different from what PyTorch's reference pooling produces.
    input_tensor = torch.randn(batch_size, channels, height, width, dtype=torch.double, device=device)
    
    # Kernel output using our CUDA extension.
    output_kernel = my_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    
    # Reference output using PyTorch's MaxPool2d.
    output_ref = F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    
    # We expect a mismatch if the device functions are not correctly implemented.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
        "Test failed: Kernel output (double) unexpectedly matches reference output. "
        "This may indicate that the device code is not using proper device functions for infinity/max."
    )

# Test 2:
# Trigger lack of error checking by using improper grid/block configuration
# via non-contiguous input. If the kernel assumes contiguous memory,
# processing a non-contiguous tensor may lead to incorrect results.
def test_issue_non_contiguous_input():
    device = "cuda"
    my_module = build_kernel()
    
    # Create a contiguous tensor and then make it non-contiguous by a transpose.
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    non_contiguous = input_tensor.transpose(1, 2)  # now non-contiguous in memory
    
    # Use float tensor; the kernel does not check contiguity.
    try:
        output_kernel = my_module.forward(non_contiguous, kernel_size, stride, padding, dilation)
    except RuntimeError as e:
        pytest.skip("Kernel raised an error on non-contiguous input, which is one expected behavior.")
    
    # Compute a reference using PyTorch (which supports non-contiguous inputs).
    output_ref = F.max_pool2d(non_contiguous, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    
    # The results may differ if the kernel does not handle non-contiguous input correctly.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
        "Test failed: Kernel output with non-contiguous input unexpectedly matches reference; "
        "the kernel assumed a contiguous layout."
    )

# Test 3:
# Trigger the issue that the kernel only supports floating point types.
# Passing an integer tensor should fail because AT_DISPATCH_FLOATING_TYPES does not cover int.
def test_issue_non_floating_input():
    device = "cuda"
    my_module = build_kernel()
    
    # Create an integer tensor.
    input_tensor = torch.randint(0, 10, (batch_size, channels, height, width), device=device, dtype=torch.int32)
    
    with pytest.raises(RuntimeError):
        _ = my_module.forward(input_tensor, kernel_size, stride, padding, dilation)

# Test 4:
# Trigger the lack of kernel error checking.
# Provide parameters that lead to an incorrect or unexpected output shape.
# For example, using a dilation greater than expected can lead to an output shape
# that does not match PyTorch's computed shape.
def test_issue_invalid_pool_params():
    device = "cuda"
    my_module = build_kernel()
    
    # Set dilation too high such that the effective kernel size is larger than the input.
    invalid_dilation = height  # very high dilation
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    
    # Compute reference output using PyTorch.
    output_ref = F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride, 
                              padding=padding, dilation=invalid_dilation)
    
    # Call our kernel.
    output_kernel = my_module.forward(input_tensor, kernel_size, stride, padding, invalid_dilation)
    
    # If the kernel did proper error checking or computed dimensions safely,
    # we would expect an error or a matching output shape.
    # Here we assert that the outputs differ, revealing that error checking is missing.
    if output_kernel.numel() > 0 and output_ref.numel() > 0:
        assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
            "Test failed: Kernel output with invalid dilation parameters unexpectedly matches reference output. "
            "Lack of proper error checking may be the cause."
        )
    else:
        pytest.skip("One of the outputs is empty due to invalid parameters; this triggers the issue of missing error checks.")

