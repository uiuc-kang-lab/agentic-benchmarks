
import torch
import torch.nn.functional as F
import pytest
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

# Helper function to run our custom CUDA operator.
def run_custom_conv(x, weight, bias, stride, padding, dilation, groups):
    my_module = build_kernel()
    # Call the custom CUDA function; note that our implementation is for depthwise convolution.
    # We assume that weight is a 4D tensor of shape (channels, 1, kernel_h, kernel_w)
    # even though the implemented kernel only uses kernel_h.
    out = my_module.forward(x, weight.squeeze(1), bias, stride, padding, dilation, groups)
    return out

# Test case 1:
# Trigger issue by using a non-standard kernel width (kernel_w != 1).
# Our custom kernel ignores any kernel width other than 1, so if we supply a weight with kernel_w > 1,
# the output dimensions (and values) will be computed incorrectly.
def test_nonstandard_kernel_width():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 3
    in_h = 32
    in_w = 32
    # Use a kernel with height 3 and width 3 (kernel_w should be 1 for our custom implementation)
    kernel_h = 3
    kernel_w = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = in_channels

    # Create an input tensor and a weight tensor with 4 dimensions (depthwise conv weight shape is (channels, 1, kh, kw)).
    x = torch.randn(batch_size, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Reference output using PyTorch's conv2d.
    ref_out = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # Run custom CUDA kernel. Note: Our kernel expects a weight tensor essentially of shape (channels, kernel_h)
    # so we pass weight.squeeze(1) which removes the 1 in the second dimension.
    custom_out = run_custom_conv(x, weight, bias, stride, padding, dilation, groups)
    
    # Because the custom kernel ignores the kernel width (only uses kernel_h),
    # the output will differ from the reference.
    # We expect the mismatch to be significant.
    assert not torch.allclose(custom_out, ref_out, atol=1e-5), (
        "Test failed: Custom kernel output matches reference output even when kernel width != 1."
    )

# Test case 2:
# Trigger issue by using dilation with a kernel width not equal to 1.
# Since the output width formula in the forward function does not incorporate dilation * (kernel_w -1),
# the output width will be computed incorrectly.
def test_nonstandard_kernel_width_with_dilation():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 4
    in_h = 40
    in_w = 40
    kernel_h = 3
    kernel_w = 3  # Non-unary kernel width; our kernel still only uses kernel_h.
    stride = 1
    padding = 2
    dilation = 2  # This dilation matters for a proper 2D convolution.
    groups = in_channels

    x = torch.randn(batch_size, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    ref_out = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    custom_out = run_custom_conv(x, weight, bias, stride, padding, dilation, groups)

    # Because the width dimension is not handled correctly with dilation when kernel_w > 1,
    # the custom kernel's output will differ from the expected one.
    assert not torch.allclose(custom_out, ref_out, atol=1e-5), (
        "Test failed: Custom kernel output matches reference output even with dilation > 1 and kernel width != 1."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
