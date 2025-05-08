
import os
import pytest
import torch
from torch.utils.cpp_extension import load
import tempfile

# Helper function for building the CUDA extension from the kernel.cu file.
def build_kernel():
    # Write the kernel code to a temporary file if needed
    kernel_filename = os.path.join(tempfile.gettempdir(), "kernel.cu")
    # Assume kernel.cu is already in the working directory. Otherwise, copy it.
    if not os.path.exists(kernel_filename):
        # For testing purposes, we assume the user has provided kernel.cu.
        # In a real-world test, you might copy from a known location.
        os.system(f'cp kernel.cu {kernel_filename}')
    
    cuda_module = load(
        name="custom_maxpool1d",
        sources=[kernel_filename],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32.
def test_dtype_support():
    cuda_module = build_kernel()
    # Create a double tensor (float64) input
    batch_size, channels, length = 2, 4, 16
    x = torch.randn(batch_size, channels, length, dtype=torch.float64, device="cuda")
    
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False

    with pytest.raises(RuntimeError):
        # Expect a runtime error due to type dispatch failure.
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)

# Issue 2: ReturnIndices behavior does not conform to expected API (concatenating output and indices).
def test_return_indices_format():
    cuda_module = build_kernel()
    batch_size, channels, length = 2, 4, 16
    # Use float32 input.
    x = torch.randn(batch_size, channels, length, device="cuda", dtype=torch.float32)
    
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = True

    result = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    # The expected behavior for maxpool with indices is to return a tuple (output, indices)
    # Instead, our kernel returns a concatenated tensor along the last dimension.
    # We check that the last dimension is twice as large as expected.
    output_length = ((length + 2*padding - dilation*(kernel_size-1) - 1) // stride) + 1
    expected_last_dim = output_length  # if it were separate output and indices
    if result.shape[-1] == expected_last_dim:
        pytest.fail("Kernel returned only output without indices; expected concatenated tensor (output, indices).")
    if result.shape[-1] != 2 * output_length:
        pytest.fail("Kernel returned a tensor with an unexpected final dimension size for concatenated output and indices.")

# Issue 3: Out-of-bound access in extreme conditions.
def test_out_of_bound_window():
    cuda_module = build_kernel()
    # Construct an input tensor where the pooling window will fall mostly out-of-bound.
    # For example, use a very small input_length with large dilation and padding.
    batch_size, channels, length = 1, 1, 4
    x = torch.randn(batch_size, channels, length, device="cuda", dtype=torch.float32)

    # Set parameters such that the pooling window starts far out-of-bound.
    kernel_size = 5
    stride = 1
    padding = 10  # large padding makes input_start largely negative
    dilation = 3
    return_indices = False

    # We expect that the kernel may try to access out-of-bound memory.
    # In a correctly defended kernel this would be caught or would produce a meaningful result.
    # Here, we trigger and catch a potential error.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
        
# Issue 4: Non-contiguous input.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size, channels, length = 2, 4, 16
    # Create a float32 tensor and then make it non-contiguous by transposing.
    x = torch.randn(batch_size, channels, length, device="cuda", dtype=torch.float32)
    x = x.transpose(1,2)  # Now shape is (batch_size, length, channels) and non-contiguous
    
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False

    with pytest.raises(RuntimeError):
        # Forward should check for contiguity and raise an error.
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
