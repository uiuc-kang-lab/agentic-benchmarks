
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Type assumption (only supports float32)
def test_input_tensor_type_issue():
    cuda_module = build_kernel()
    # Create tensors of type float64 instead of float32
    batch_size = 4
    in_channels = 8
    out_channels = 6
    input_length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    # Create inputs with wrong precision (float64)
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    with pytest.raises(RuntimeError):
        # The kernel expects float32 pointers; using float64 should trigger an error.
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 2: Shared memory size overflow (weight too large)
def test_shared_memory_overflow_issue():
    cuda_module = build_kernel()
    # Set a configuration where the weight tensor is huge.
    # Note: Actual shared memory limits vary per device; the following size is almost certainly
    # larger than what most GPUs can support.
    batch_size = 1
    in_channels = 512
    out_channels = 512
    input_length = 10
    kernel_size = 50  # This results in a huge shared memory requirement.
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError):
        # The kernel launch should fail because the shared memory allocation
        # exceeds the device's available shared memory.
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 3: Lack of kernel execution error checking (e.g. invalid parameters causing division by zero)
def test_stride_zero_issue():
    cuda_module = build_kernel()
    # Set stride to 0 to force a divide/modulo by zero error in the kernel.
    batch_size = 2
    in_channels = 4
    out_channels = 3
    input_length = 16
    kernel_size = 3
    stride = 0  # invalid stride, will cause modulo division by zero.
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    with pytest.raises(Exception):
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()
        
# Issue 4: Lack of error checking after kernel launch (this issue may be indirectly caught
# by tests above, but here we explicitly expect that an error in the kernel will raise an exception)
def test_kernel_launch_error_check():
    cuda_module = build_kernel()
    # Provide an input with wrong dimensions to trigger an error in the kernel launch.
    # For example, weight tensor with incorrect dimensions.
    batch_size = 2
    in_channels = 4
    out_channels = 3
    input_length = 16
    kernel_size = 3
    
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    # Wrong weight shape: e.g., 4 dimensions instead of 3.
    weight = torch.randn(in_channels, out_channels, kernel_size, 1, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError):
        # This should fail at the host check in forward_cuda.
        out = cuda_module.forward(x, weight, bias, 1, 1, 1)
        torch.cuda.synchronize()
