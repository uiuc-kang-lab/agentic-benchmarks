
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_maxpool1d",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Unused shared memory. (We can’t directly test unused memory,
# but we can trigger a kernel launch ensuring that it doesn't affect the results.)
def test_unused_shared_memory():
    # Prepare a simple input so that kernel runs.
    batch_size = 1
    num_channels = 1
    input_length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False

    x = torch.randn(batch_size, num_channels, input_length, device="cuda", dtype=torch.float32)
    # The expected behavior is max pooling similar to PyTorch's MaxPool1d.
    expected_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    custom_module = build_kernel()
    y = custom_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    # Because shared memory is unused, output should be computed correctly.
    assert y.shape == (batch_size, num_channels, expected_length), \
        "Output shape does not match expected shape despite unused shared memory."

# Issue 2: Only supports float32. Testing by passing a float64 tensor.
def test_input_tensor_type():
    batch_size = 1
    num_channels = 1
    input_length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False

    # Create a float64 tensor
    x = torch.randn(batch_size, num_channels, input_length, device="cuda", dtype=torch.float64)
    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting the kernel to fail or raise an error due to type mismatch.
        custom_module.forward(x, kernel_size, stride, padding, dilation, return_indices)

# Issue 3: Inconsistent return_indices behavior.
def test_return_indices_concatenation():
    batch_size = 1
    num_channels = 1
    input_length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = True

    x = torch.randn(batch_size, num_channels, input_length, device="cuda", dtype=torch.float32)
    custom_module = build_kernel()
    out = custom_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    
    # The kernel returns a concatenated tensor of output and indices along the last dimension.
    # Compute expected dimensions:
    expected_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    # The concatenated dimension is twice the expected_length (first half max values, second half indices).
    expected_shape = (batch_size, num_channels, expected_length * 2)
    assert out.shape == expected_shape, \
        "Return indices concatenation issue: output shape does not match expected concatenated shape."

# Issue 4: Lack of kernel launch error checking.
# We simulate a scenario that is likely to cause a kernel launch error (e.g. invalid input dimensions).
def test_invalid_input_dimension():
    # Create a 2D tensor (invalid since the kernel expects a 3D tensor)
    x = torch.randn(16, 64, device="cuda", dtype=torch.float32)
    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        custom_module.forward(x, 3, 1, 1, 1, False)
        
if __name__ == "__main__":
    pytest.main([__file__])
