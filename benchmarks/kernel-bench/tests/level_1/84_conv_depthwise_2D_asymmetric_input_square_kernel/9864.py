
import pytest
import torch
from torch.nn.functional import conv2d
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    # Force rebuild by adding extra_cuda_cflags=...
    cuda_module = load(
        name="depthwise_conv2d_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Issue 1: Dtype support
def test_incorrect_dtype(kernel_module):
    # Create inputs with double precision (float64) which the kernel does not support.
    batch_size = 2
    in_channels = 3  # Depthwise: groups == in_channels
    kernel_size = 3
    height_in, width_in = 16, 16
    stride = 1
    padding = 0
    
    # Create double precision tensor inputs
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, dtype=torch.float64, device="cuda")
    # For depthwise conv, weight shape should be (in_channels, multiplier, kernel_size, kernel_size)
    # Here, use multiplier 1
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float64, device="cuda")
    bias_tensor = torch.randn(in_channels, dtype=torch.float64, device="cuda")
    
    # The kernel does not check for dtypes and will reinterpret the data as float32.
    # So the output will be almost surely different than the correct convolution.
    with pytest.raises(AssertionError):
        # We compare with PyTorch conv2d (after casting to float32) to detect inconsistencies.
        out_kernel = kernel_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding)
        torch.cuda.synchronize()
        # Convert input to float32 and compute PyTorch conv2d result with groups = in_channels.
        input32 = input_tensor.float()
        weight32 = weight_tensor.float()
        bias32 = bias_tensor.float()
        out_ref = conv2d(input32, weight32, bias32, stride=stride, padding=padding, groups=in_channels)
        # Since dtypes differ, the kernel output is expected to be wrong.
        assert torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel incorrectly processed non-float32 tensors."

# Issue 2: Loop unrolling with dynamic loop bounds.
def test_loop_unrolling_effect(kernel_module):
    # Use a kernel_size that is not a typical compile-time constant value (e.g., not 3).
    # Although the kernel is written for square kernels,
    # the usage of #pragma unroll may fail to unroll the loop effectively.
    # In this test, we simply check that the kernel still produces correct output.
    batch_size = 2
    in_channels = 3  # groups == in_channels
    kernel_size = 5  # dynamic value that may not be unrolled properly
    height_in, width_in = 20, 20
    stride = 1
    padding = 2  # to keep the output size same as input
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, dtype=torch.float32, device="cuda")
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float32, device="cuda")
    bias_tensor = torch.randn(in_channels, dtype=torch.float32, device="cuda")
    
    out_kernel = kernel_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding)
    torch.cuda.synchronize()
    
    # Compute reference output using PyTorch conv2d with groups set appropriately.
    out_ref = conv2d(input_tensor, weight_tensor, bias_tensor, stride=stride, padding=padding, groups=in_channels)
    
    assert torch.allclose(out_kernel, out_ref, atol=1e-4), "Kernel output differs with kernel_size not known at compile time."

# Issue 3: Lack of generality (only supports depthwise convolution).
def test_group_support(kernel_module):
    # In a general convolution, groups can be arbitrary.
    # This kernel is only for depthwise convolution (groups == in_channels).
    # Here, we simulate a scenario where groups != in_channels.
    batch_size = 2
    in_channels = 4
    groups = 2  # arbitrary group count not equal to in_channels (should be 4 for depthwise)
    kernel_size = 3
    height_in, width_in = 16, 16
    stride = 1
    padding = 1
    
    # For a general grouped conv2d, PyTorch expects weight of shape (out_channels, in_channels // groups, k, k)
    out_channels = 8  # arbitrary
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, dtype=torch.float32, device="cuda")
    weight_tensor = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, dtype=torch.float32, device="cuda")
    bias_tensor = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    
    # We expect the kernel to fail the internal assumptions when groups != in_channels.
    with pytest.raises(AssertionError):
        # The forward function of the kernel implementation computes:
        #   channels_per_group = weight.size(1)
        #   out_channels = in_channels * channels_per_group
        # So, if groups do not match in_channels then the bias size or output dimensions will be mismatched.
        kernel_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding)

# Issue 4: No error checking after kernel launch.
def test_no_post_launch_error_check(kernel_module):
    # We simulate a scenario that causes an illegal memory access in the kernel launch.
    # One way is to deliberately create an output tensor with insufficient size.
    # For example, use negative padding that makes the computed output dimensions negative.
    batch_size = 1
    in_channels = 3
    kernel_size = 3
    height_in, width_in = 8, 8
    stride = 1
    padding = -1  # invalid padding will lead to negative output dimensions, hence illegal index.
    
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, dtype=torch.float32, device="cuda")
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float32, device="cuda")
    bias_tensor = torch.randn(in_channels, dtype=torch.float32, device="cuda")
    
    # Although the host code does not perform post-launch error checking,
    # a negative padding will lead to invalid memory accesses inside the kernel.
    with pytest.raises(RuntimeError):
        out_kernel = kernel_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding)
        # Force synchronization to catch the error.
        torch.cuda.synchronize()
