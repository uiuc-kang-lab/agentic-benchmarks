
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="depthwise_conv_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to invoke the forward kernel
def run_forward(x, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups):
    mod = build_kernel()
    if bias is not None:
        return mod.forward(x, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups)
    else:
        from c10 import optional  # if needed, else pass bias as None in the extension
        return mod.forward(x, weight, None, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups)

# Issue 1: Kernel only accepts float32
def test_kernel_with_double_precision():
    # Create inputs with double precision.
    batch_size = 2
    in_channels = 3
    height = 8
    width = 8
    kernel_h = 3
    kernel_w = 3
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.double, device="cuda")
    # For a depthwise conv, weight shape is [in_channels, 1, kernel_h, kernel_w]
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, dtype=torch.double, device="cuda")
    bias_tensor = torch.randn(in_channels, dtype=torch.double, device="cuda")
    
    with pytest.raises(RuntimeError):
        # Expect a failure (or incorrect dispatch) because the kernel only supports float32.
        _ = build_kernel().forward(x, weight, bias_tensor, 1, 1, 0, 0, 1, 1, in_channels)

# Issue 2: The kernel assumes that out_channels divides evenly by groups.
def test_channels_per_group_not_integer():
    # In this test, we deliberately choose weight dimensions that lead to channels_per_group 
    # not being an integer.
    batch_size = 1
    # Let groups be 2 but set in_channels=3 which is common in practice for depthwise conv
    # but then weight shape is expected to be [groups, channels_per_group, kH, kW] with channels_per_group=?
    # In our custom kernel, out_channels is computed as groups * weight.size(1).
    # We set weight.size(1) such that groups*weight.size(1) != in_channels.
    groups = 2
    # Using in_channels=3 will force channels_per_group = (groups*weight.size(1)) / groups = weight.size(1)
    # which will then be used to index input as if channel = c_out/groups. 
    # Here, we intentionally set weight.size(1)=2 so that out_channels=4,
    # but the input channels remain 3 so that indexing into the input based on group becomes invalid.
    in_channels = 3  # input tensor always has 3 channels
    kernel_h = 3
    kernel_w = 3
    height = 8
    width = 8

    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    # Create a weight with out_channels=groups*2 = 4, even though input has only 3 channels.
    weight = torch.randn(groups, 2, kernel_h, kernel_w, dtype=torch.float32, device="cuda")
    bias_tensor = torch.randn(groups * 2, dtype=torch.float32, device="cuda")
    
    # This is an invalid configuration since the kernel will calculate the input channel index 
    # based on groups and will try to access input at channel index >= in_channels.
    with pytest.raises(RuntimeError):
        _ = build_kernel().forward(x, weight, bias_tensor, 1, 1, 0, 0, 1, 1, groups)

# Issue 3: The kernel does not check that the bias tensor matches the number of output channels.
def test_bias_shape_mismatch():
    # Set up a correct depthwise convolution configuration but with a bias tensor of the wrong shape.
    batch_size = 2
    in_channels = 4  # For depthwise conv, groups == in_channels
    kernel_h = 3
    kernel_w = 3
    height = 8
    width = 8
    groups = in_channels  # correct depthwise
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    # weight shape is [in_channels, 1, kernel_h, kernel_w] in a typical depthwise conv.
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, dtype=torch.float32, device="cuda")
    # Provide a bias of incorrect shape: e.g., length != in_channels.
    bias_tensor = torch.randn(in_channels + 1, dtype=torch.float32, device="cuda")
    
    # Expect the kernel to trigger an error or cause an out-of-bound read.
    with pytest.raises(RuntimeError):
        _ = build_kernel().forward(x, weight, bias_tensor, 1, 1, 0, 0, 1, 1, groups)
