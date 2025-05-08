
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build and load the CUDA extension from kernel.cu.
def build_kernel():
    module = load(
        name="custom_conv_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Test 1: Data type support issue.
# Pass double precision tensors to force the kernel (which expects float) to operate on wrong type.
# We expect that the computation will be wrong (or throw an error) because the kernel does not
# support float64.
def test_input_tensor_type():
    kernel_module = build_kernel()
    batch_size = 4
    in_channels = 3
    out_channels = 8
    height, width = 16, 16
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1

    # Create inputs in double precision (float64) instead of float32.
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float64, device='cuda')
    # Create weight in double precision.
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float64, device='cuda')
    # Optionally add bias.
    bias = torch.randn(out_channels, dtype=torch.float64, device='cuda')

    # Calling the kernel expecting float (32-bit) pointers. In many cases this mismatch will lead
    # to wrong results or an error. In this test we expect an error.
    with pytest.raises(RuntimeError):
        output = kernel_module.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)
        # Force synchronization to catch asynchronous kernel errors.
        torch.cuda.synchronize()

# Test 2: Groups divisibility issue.
# Provide tensors where the number of input channels is not evenly divisible by groups.
# The kernel computes C_in_per_group via integer division so the wrong weight slice will be used.
def test_groups_not_divisible():
    kernel_module = build_kernel()
    batch_size = 2
    in_channels = 3  # 3 is not divisible by 2.
    out_channels = 4
    kernel_size = (3, 3)
    height, width = 16, 16
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 2

    # Even though PyTorch normally enforces divisibility, we bypass this by calling our custom kernel directly.
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device='cuda')
    # The kernel expects weight shape: [C_out, C_in/groups, K_h, K_w]. Here C_in//groups = 3//2 = 1.
    # Thus, one input channel is effectively used per group.
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float32, device='cuda')
    bias = torch.randn(out_channels, dtype=torch.float32, device='cuda')

    # Run the custom kernel.
    output = kernel_module.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)
    torch.cuda.synchronize()
    # Compute a reference output using torch.nn.functional.conv2d (which performs proper error checking)
    ref_output = torch.nn.functional.conv2d(x, weight, bias, stride, padding, dilation, groups)
    # Because of the mismatch in channel splitting, the outputs are expected NOT to match.
    assert not torch.allclose(output, ref_output, atol=1e-4), \
        "The kernel output unexpectedly matches the reference despite groups not dividing channels!"

# Test 3: Shared memory overflow issue.
# Create a situation where the weight tile is so large that the shared memory requested exceeds the
# GPU’s limit; this should cause the kernel launch to fail.
def test_shared_memory_overflow():
    kernel_module = build_kernel()
    batch_size = 1
    # Choose a very high number of input channels to force a huge shared memory allocation.
    in_channels = 1024
    out_channels = 64
    # Use a relatively large kernel size.
    kernel_size = (11, 11)
    height, width = 32, 32
    stride = (1, 1)
    padding = (5, 5)
    dilation = (1, 1)
    groups = 1

    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device='cuda')
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float32, device='cuda')
    bias = torch.randn(out_channels, dtype=torch.float32, device='cuda')

    # In many GPUs the maximum shared memory per block is on the order of 48KB.
    # Here, shared memory needed = (in_channels * 11 * 11 * 4 bytes) which is far above that.
    with pytest.raises(RuntimeError):
        output = kernel_module.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)
        torch.cuda.synchronize()
