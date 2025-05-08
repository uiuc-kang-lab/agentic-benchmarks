
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Build the extension from the kernel.cu source file.
    # Assumes that kernel.cu is in the same directory as this test file.
    sources = [os.path.join(os.path.dirname(__file__), "kernel.cu")]
    cuda_module = load(
        name="test_kernel",
        sources=sources,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 4: Test to demonstrate that the kernel does not use the shared memory as advertised.
# While we cannot directly inspect shared memory usage from Python,
# we trigger the kernel with an input where coalescing would normally benefit from shared memory usage.
# We then compare against PyTorch's native AvgPool3d output.
def test_shared_memory_unused():
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = 2
    channels = 3
    depth = 10
    height = 10
    width = 10

    # Create input tensor filled with random values.
    input_tensor = torch.randn(batch_size, channels, depth, height, width, device="cuda", dtype=torch.float32)
    
    # Native PyTorch implementation
    ref_output = torch.nn.functional.avg_pool3d(input_tensor, kernel_size, stride=stride, padding=padding)
    
    # Our custom CUDA kernel
    cuda_mod = build_kernel()
    output = cuda_mod.forward(input_tensor, kernel_size, stride, padding)
    
    # They should match since nn.AvgPool3d with count_include_pad=True does full-volume division.
    assert torch.allclose(output, ref_output, atol=1e-5), f"Output mismatch indicating potential misuse of memory optimizations."

# Issue 2: Valid elements are computed but never used.
# In scenarios where the pooling window partially overlaps the input, the average should technically be valid_elements / pool_volume.
# The custom kernel always divides by full pool volume.
# This test creates an input tensor of ones and places the pooling window at the border.
def test_valid_elements_not_used():
    kernel_size = 3
    stride = 1
    padding = 1
    batch_size = 1
    channels = 1
    # Use a small 3D tensor so that border effects are prominent.
    depth = 3
    height = 3
    width = 3

    # Create an input tensor of ones so that the averaged result at border differs depending on treatment.
    input_tensor = torch.ones(batch_size, channels, depth, height, width, device="cuda", dtype=torch.float32)
    
    # Native PyTorch implementation using count_include_pad=False would compute a different average at borders,
    # but our kernel mimics count_include_pad=True.
    # Here we force the reference using count_include_pad=False by manually computing a different expected value
    # at the border. For the center voxel the window covers all ones so result is 1, while for a border voxel,
    # like (0,0,0), the native computed average (if excluding padding) would be less than 1.
    # Our kernel always divides by 27. We check for a known border element.
    cuda_mod = build_kernel()
    output = cuda_mod.forward(input_tensor, kernel_size, stride, padding)

    # For position (batch=0, channel=0, d=0, h=0, w=0), pooling window covers:
    # effective region indices d: 0-1, h: 0-1, w: 0-1 (8 ones) if excluding padding.
    # count_include_pad=True (used by our kernel) results in 8/27.
    expected_value = 8 / 27
    actual_value = output[0,0,0,0,0].item()
    assert abs(actual_value - expected_value) < 1e-5, f"Expected {expected_value} but got {actual_value}. Indicates that valid_elements is not used."

# Issue 3: The kernel only supports float32.
def test_input_tensor_type():
    kernel_size = 3
    stride = 2
    padding = 1
    batch_size = 2
    channels = 2
    depth = 8
    height = 8
    width = 8

    # Create an input tensor of type double.
    input_tensor = torch.randn(batch_size, channels, depth, height, width, device="cuda", dtype=torch.float64)
    
    cuda_mod = build_kernel()
    
    # Expect a runtime error or incorrect computation because the kernel
    # will interpret the underlying data as float.
    with pytest.raises(RuntimeError):
        output = cuda_mod.forward(input_tensor, kernel_size, stride, padding)
        # Force synchronization to catch async errors.
        torch.cuda.synchronize()
