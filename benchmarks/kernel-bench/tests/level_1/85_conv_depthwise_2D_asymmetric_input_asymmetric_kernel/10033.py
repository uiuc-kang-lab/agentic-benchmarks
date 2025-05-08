
import torch
import pytest
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="cuda_depthwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_grouped_convolution_issue():
    # This test triggers Issue 1:
    # The kernel is not summing over multiple channels in each group.
    #
    # Setting up a grouped convolution where channels_per_group > 1:
    # For a correct grouped convolution:
    #   - Let in_channels = 4, groups = 2, so channels_per_group = 2.
    #   - For usual conv2d, weight shape should be (out_channels, in_channels//groups, kh, kw)
    # Here, our kernel expects weight with shape [groups * channels_per_group, kh, kw] in a depthwise scenario,
    # but when channels_per_group > 1 the proper summation over the two input channels is missing.
    
    batch_size = 2
    in_channels = 4
    groups = 2  # Change non-depthwise: channels_per_group should be 2 (i.e. 4/2)
    stride = 1
    padding = 1
    dilation = 1
    kh, kw = 3, 3
    H, W = 8, 8

    # Create a random input tensor.
    x = torch.randn(batch_size, in_channels, H, W, device="cuda", dtype=torch.float32)

    # For grouped convolution in F.conv2d, weight shape is (out_channels, in_channels//groups, kh, kw)
    out_channels = in_channels  # e.g. output channels = 4 (can be any, but must equal groups*(in_channels//groups) for a valid conv)
    weight = torch.randn(out_channels, in_channels // groups, kh, kw, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Get reference output using F.conv2d.
    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    # Our kernel expects weight data pointer to be in a layout corresponding to depthwise conv.
    # To trick the kernel, we flatten the weight to shape (out_channels, kh, kw)
    # Note: In a correct general case the kernel would need to perform summation over the channel dimension.
    weight_flat = weight.flatten(1)  # Fake a "depthwise" weight by taking only one channel per group.
    # Here, simply take the first channel of every kernel filter to mimic the expected layout.
    weight_depth = weight[:, 0:1, :, :].clone().detach().contiguous()
    
    # Invoke the CUDA kernel.
    kernel_mod = build_kernel()
    kernel_out = kernel_mod.forward(
        x, weight_depth, bias,
        stride, stride, padding, padding,
        dilation, dilation, groups
    )
    torch.cuda.synchronize()

    # Since the kernel does not sum over multiple channels in a group,
    # the outputs will differ from the reference computed via F.conv2d.
    # We want to see that the difference is significant.
    diff = (kernel_out - ref).abs().max().item()
    assert diff > 1e-3, (
        f"Test did not trigger the grouped convolution issue (max difference {diff}). "
        "Expected significant discrepancy due to missing summation over channels in each group."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dilation_valid_range_issue():
    # This test triggers Issue 2:
    # The valid kernel height range (valid_kh_start, valid_kh_end) is computed using integer division,
    # which may be off under non-unit dilation and certain padding/stride conditions.
    
    batch_size = 2
    in_channels = 3  # Standard case for depthwise convolution (groups == in_channels)
    groups = in_channels  # depthwise conv so each channel is separate
    stride = 1
    padding = 2
    dilation = 2  # Non-unit dilation
    kh, kw = 3, 5  # asymmetric kernel as in the example
    H, W = 16, 16

    x = torch.randn(batch_size, in_channels, H, W, device="cuda", dtype=torch.float32)
    # For depthwise, weight shape: (in_channels, 1, kh, kw)
    weight = torch.randn(in_channels, 1, kh, kw, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Reference output via F.conv2d.
    ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    kernel_mod = build_kernel()
    kernel_out = kernel_mod.forward(
        x, weight, bias,
        stride, stride, padding, padding,
        dilation, dilation, groups
    )
    torch.cuda.synchronize()

    # Due to likely incorrect boundary computations for the kernel window,
    # a significant difference is expected.
    diff = (kernel_out - ref).abs().max().item()
    assert diff > 1e-3, (
        f"Test did not trigger the dilation valid-range issue (max difference {diff}). "
        "Expected discrepancy due to incorrect computation of valid kernel region boundaries with dilation."
    )
