
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Helper to build our CUDA extension (assuming the kernel is defined in kernel.cu)
def build_kernel():
    cuda_module = load(
        name="custom_depthwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Data type mismatch (only float32 is supported).
def test_input_tensor_type():
    # Build the extension
    module = build_kernel()
    # Create input and weight tensors of type float64 (double)
    # Note: In proper use the extension is expecting float32,
    # so here we trigger misinterpretation of memory.
    batch_size = 2
    in_channels = 3
    height = 16
    width = 16
    kernel_h = 3
    kernel_w = 3
    stride = 1; padding = 0; dilation = 1; groups = in_channels

    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float64, device='cuda')
    # For depthwise conv, weight shape expected is (in_channels, 1, kernel_h, kernel_w)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, dtype=torch.float64, device='cuda')
    bias = torch.randn(in_channels, dtype=torch.float64, device='cuda')
    
    # Call the extension function. It will use data_ptr<float>()
    out_cust = module.forward(
        x, weight, bias,
        stride, stride, padding, padding, dilation, dilation, groups
    )
    
    # Build a reference convolution with proper type conversion.
    conv = nn.Conv2d(in_channels, in_channels, (kernel_h, kernel_w),
                     stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True).cuda()
    # Copy our weight into conv.
    # Note: Because torch.nn.Conv2d parameters are float32 by default (if not provided as double),
    # we perform a conversion.
    conv.weight.data = weight.float()
    conv.bias.data = bias.float()
    out_ref = conv(x.float())
    
    # Since the kernel wrongly interprets double as float, the outputs will be very different.
    # We expect the outputs to not be all-close.
    assert not torch.allclose(out_cust, out_ref, atol=1e-3), (
        "Test for wrong input tensor type did not trigger an error: "
        "the custom kernel appears to work with double tensors, "
        "but it is hardcoded for float32."
    )


# Issue 2: Incorrect input indexing for grouped convolution when channels_per_group > 1.
def test_grouped_convolution():
    module = build_kernel()
    # Here we create a grouped convolution where groups != in_channels.
    # Let in_channels = 4 and groups = 2 ==> channels per group = 2.
    batch_size = 2
    in_channels = 4
    groups = 2
    out_channels = groups * (in_channels // groups)
    height = 16
    width = 16
    kernel_h = 3
    kernel_w = 3
    stride = 1; padding = 1; dilation = 1

    # For a general grouped convolution, the weight shape is:
    # (out_channels, in_channels/groups, kernel_h, kernel_w)
    weight = torch.randn(out_channels, in_channels // groups, kernel_h, kernel_w, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)

    out_cust = module.forward(
        x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups
    )
    
    # Build a standard grouped convolution in PyTorch for reference.
    conv = nn.Conv2d(in_channels, out_channels, (kernel_h, kernel_w),
                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True).cuda()
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(bias)
    out_ref = conv(x)
    
    # Due to wrong channel input indexing inside the kernel, the custom kernel output
    #  will be different.
    assert not torch.allclose(out_cust, out_ref, atol=1e-4), (
        "Test for grouped convolution did not trigger an error: "
        "the custom kernel produced results matching the correct grouped conv, "
        "but it does not properly handle channels_per_group > 1."
    )


# Issue 3: Launch configuration using a 1D grid may exceed CUDA's grid dimension limits.
def test_grid_dimension_limit():
    module = build_kernel()
    # We will try to force the shared-memory kernel launch by choosing a larger kernel
    # Since the forward() switches based on kernel_h*kernel_w, we use a kernel size > 25.
    kernel_h = 6
    kernel_w = 6  # product = 36 > 25 -> shared kernel should be used.
    stride = 1; padding = 0; dilation = 1
    groups = 1
    
    # Set up huge spatial dimensions such that the total number of outputs is enormous.
    # Many GPUs support a maximum grid dim.x of around 2^31-1.
    # We choose an output size that forces gridSize = batch_size*out_channels*out_h*out_w to be huge.
    batch_size = 1
    in_channels = 1
    # Use huge height and width so that out_h and out_w are enormous.
    # (These numbers may be tuned to exceed typical grid x limits – the test expects a RuntimeError.)
    in_h = 70000
    in_w = 70000

    # Create input and weight tensors accordingly.
    x = torch.randn(batch_size, in_channels, in_h, in_w, device='cuda', dtype=torch.float32)
    # For a depthwise convolution, weight shape: (in_channels, 1, kernel_h, kernel_w)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device='cuda', dtype=torch.float32)
    bias = torch.randn(in_channels, device='cuda', dtype=torch.float32)

    with pytest.raises(RuntimeError):
        # This call should trigger a CUDA RuntimeError due to too huge grid dimensions.
        out = module.forward(
            x, weight, bias, stride, stride, 0, 0, dilation, dilation, groups
        )
        # Also force a device sync to provoke the runtime error.
        torch.cuda.synchronize()


# Issue 4: Kernel selection based solely on kernel product may misclassify asymmetric kernels.
def test_asymmetric_kernel():
    module = build_kernel()
    # Choose an asymmetric kernel that has a product that is small but one dimension is larger than expected.
    # For example, kernel_h=3, kernel_w=7 gives product = 21 which is <= 25.
    # The forward() will dispatch the "small" kernel (depthwise_conv2d_kernel_2d),
    # but the implementation may be suboptimal or even incorrect for such asymmetry.
    batch_size = 2
    in_channels = 3  # depthwise conv: groups=in_channels.
    kernel_h = 3
    kernel_w = 7
    stride = 1; padding = 1; dilation = 1; groups = in_channels
    height = 32
    width = 32

    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    # Weight shape, for depthwise conv (each channel has a single kernel):
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, device='cuda', dtype=torch.float32)
    bias = torch.randn(in_channels, device='cuda', dtype=torch.float32)

    out_cust = module.forward(
        x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups
    )

    # Build a reference convolution using torch.nn.Conv2d.
    conv = nn.Conv2d(in_channels, in_channels, (kernel_h, kernel_w),
                     stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True).cuda()
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(bias)
    out_ref = conv(x)

    # If the wrong kernel is selected or implemented for asymmetric kernels,
    # the output will differ from the correct result.
    assert not torch.allclose(out_cust, out_ref, atol=1e-4), (
        "Test for asymmetric kernel did not trigger an error: "
        "the custom kernel produced correct results even with an asymmetric kernel, "
        "but it was expected to mis-handle such cases."
    )
