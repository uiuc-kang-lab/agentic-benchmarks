
import torch
import pytest
from torch import nn
from torch.utils.cpp_extension import load

def build_kernel():
    try:
        cuda_module = load(
            name="test_kernel",
            sources=["kernel.cu"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=True,
        )
    except Exception as e:
        pytest.skip("CUDA module build failed: " + str(e))
    return cuda_module

# Helper: Native PyTorch ConvTranspose2d operation for reference output
class NativeConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.ct = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # Set weights and bias to known constant values.
        nn.init.constant_(self.ct.weight, 1.0)
        if bias:
            nn.init.constant_(self.ct.bias, 1.0)
    def forward(self, x):
        return self.ct(x)

# Test case 1: Incomplete block-level reduction
#  For inputs that force more than one warp to perform reduction,
#  the kernel’s output will be incorrect compared to the reference.
def test_incomplete_reduction():
    # Use parameters that lead to many reduction iterations:
    # C_in * kH * kW will be much larger than warpSize.
    batch_size = 2
    in_channels = 32
    out_channels = 16
    kernel_size = (3, 5)  # => reduction iterations = 32 * 3 * 5 = 480 (> warpSize)
    stride = (1, 1)
    padding = (1, 2)
    height = 16
    width = 16

    # Create input tensor filled with ones so that expected sum is predictable.
    x = torch.ones(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    # Create weight tensor filled with ones.
    # Weight shape expected for conv_transpose2d: (C_in, C_out, kH, kW)
    weight = torch.ones(in_channels, out_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)

    # Create a dummy bias tensor (set to zero) to keep things simple.
    bias = torch.zeros(out_channels, device="cuda", dtype=torch.float32)

    # Build the custom kernel module.
    cuda_module = build_kernel()

    # Launch the custom CUDA kernel.
    # The kernel forward signature: forward(x, weight, bias, stride, padding)
    output_custom = cuda_module.forward(x, weight, bias, list(stride), list(padding))

    # Compute reference with native PyTorch ConvTranspose2d.
    native_op = NativeConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    # Set weights to ones.
    native_op.ct.weight.data.copy_(weight.transpose(0, 1))  # native weight shape: (C_out, C_in, kH, kW)
    # For bias, use zeros.
    native_op.ct.bias.data.copy_(bias)
    # Since our kernel uses a slightly different weight layout, we compute expected output via at::conv_transpose2d:
    output_ref = torch.nn.functional.conv_transpose2d(x, weight.transpose(0, 1), bias, stride, padding)

    # The expected per-element contribution is the sum of 1's over the valid reduction iterations.
    # Due to missing cross-warp reduction, output_custom is expected to be incorrect.
    assert not torch.allclose(output_custom, output_ref, atol=1e-4), \
        "Test failed: Custom kernel unexpectedly produced correct results despite incomplete reduction."

# Test case 2: Weight size exceeding constant memory size or mismatch buffer sizes
# For this test we force a weight tensor that exceeds the fixed constant memory size.
# The host function in the custom kernel should then bypass the custom kernel and call the fallback.
def test_weight_constant_memory_overflow():
    batch_size = 1
    in_channels = 128
    out_channels = 128
    # Create a weight tensor larger than 16384 floats.
    # For example, shape (128, 128, 5, 5) = 128*128*25 = 409600 floats, which is > 16384.
    kernel_size = (5, 5)
    stride = (1, 1)
    padding = (2, 2)
    height = 32
    width = 32

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Weight should be created with the layout (C_in, C_out, kH, kW).
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    cuda_module = build_kernel()

    # When weight tensor size exceeds constant memory, the fallback should be invoked
    # which means the kernel module will return the result of at::conv_transpose2d.
    output_custom = cuda_module.forward(x, weight, bias, list(stride), list(padding))
    output_fallback = torch.nn.functional.conv_transpose2d(x, weight.transpose(0, 1), bias, stride, padding)

    # We expect the fallback output to be correct.
    assert torch.allclose(output_custom, output_fallback, atol=1e-4), \
        "Test failed: Fallback path did not produce the expected result when weight size exceeds constant memory."

if __name__ == "__main__":
    pytest.main([__file__])
