
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="my_pooling_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue #1 by providing an input tensor with different height and width.
# Since the kernel mistakenly uses the height as the width,
# the average pooling result will be incorrect when in_h != in_w.
def test_incorrect_dimension_usage():
    # Create an input tensor where height and width differ.
    batch_size = 2
    channels = 3
    depth = 8
    height = 20  # height
    width = 25   # width (different from height to expose the bug)
    kernel_size = 3
    stride = 2
    padding = 1

    # Input tensor on CUDA.
    input_tensor = torch.randn(batch_size, channels, depth, height, width, device="cuda", dtype=torch.float32)
    
    # Expected output computed with PyTorch's built-in AvgPool3d (which is correct).
    avg_pool = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    expected_output = avg_pool(input_tensor)
    
    # Invoke our custom CUDA kernel.
    my_module = build_kernel()
    output = my_module.forward(input_tensor, kernel_size, stride, padding)
    
    # The outputs should match but due to the bug, they will not.
    # We use a loose tolerance so that if they accidentally match (unlikely) the test fails.
    assert not torch.allclose(output, expected_output, atol=1e-4), \
        "Kernel output matches expected output despite different height and width. Bug in dimension handling might be masked."

# Test 2: Trigger issue #2 by simulating an error in copying constant memory.
# We force a wrong size in constant memory parameters by passing an obviously invalid kernel_size.
def test_constant_memory_copy_error():
    batch_size = 1
    channels = 1
    depth = 8
    height = 8
    width = 8
    
    # An invalid kernel_size (e.g., negative value) to see if the kernel or the host code
    # catches errors from cudaMemcpyToSymbol (note: this may also be caught on the PyTorch side via TORCH_CHECK)
    kernel_size = -3
    stride = 1
    padding = 0

    input_tensor = torch.randn(batch_size, channels, depth, height, width, device="cuda", dtype=torch.float32)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The invalid kernel_size should cause an error (or produce invalid results, which we treat as failure).
        # The error might occur due to a failed constant memory copy or later in the kernel.
        _ = my_module.forward(input_tensor, kernel_size, stride, padding)
