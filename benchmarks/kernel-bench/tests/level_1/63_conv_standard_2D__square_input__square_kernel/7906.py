
import torch
import torch.nn as nn
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to emulate the intended convolution using the custom CUDA function.
# Warning: This function has the known issues (ignoring dilation and groups) that we will test.
def custom_conv_forward(x, weight, bias, stride, padding, dilation, groups, cuda_module):
    return cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dilation_issue():
    # Use dilation != 1 to trigger the issue
    batch_size = 1
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    dilation = 2  # Non-unit dilation must be handled correctly, but our kernel ignores it.
    stride = 1
    padding = 1

    # Create input and weight
    x = torch.randn(batch_size, in_channels, 16, 16, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    # Use bias is optional, here we set it to None.
    bias = None

    # Build the cuda module
    cuda_module = build_kernel()

    # Run custom conv
    custom_out = custom_conv_forward(x, weight, bias, stride, padding, dilation, 1, cuda_module)
    
    # Use PyTorch's own conv2d as reference with dilation
    conv2d_ref = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False).cuda()
    # Set conv2d_ref weights to the same weights for fair comparison.
    conv2d_ref.weight.data.copy_(weight)
    ref_out = conv2d_ref(x)
    
    # Since our kernel ignores dilation, the outputs should not match.
    if torch.allclose(custom_out, ref_out, atol=1e-4):
        pytest.fail("Custom CUDA kernel incorrectly handled dilation. Expected mismatch because dilation was ignored.")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_groups_issue():
    # Use groups != 1 to trigger the groups issue.
    batch_size = 1
    groups = 2
    # Let in_channels and out_channels be divisible by groups.
    in_channels = 4
    out_channels = 4  # For a grouped conv, each group has 2 in and 2 out channels
    kernel_size = 3
    dilation = 1
    stride = 1
    padding = 1

    # Create input and weight for a grouped conv (note: weight shape for groups: [out_channels, in_channels/groups, k, k])
    x = torch.randn(batch_size, in_channels, 16, 16, device='cuda', dtype=torch.float32)
    # Create weight with shape (out_channels, in_channels/groups, kernel_size, kernel_size)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    
    bias = None

    # Build CUDA module
    cuda_module = build_kernel()

    # Run custom conv
    custom_out = custom_conv_forward(x, weight, bias, stride, padding, dilation, groups, cuda_module)

    # Use PyTorch's own conv2d as reference with groups
    conv2d_ref = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False).cuda()
    conv2d_ref.weight.data.copy_(weight)
    ref_out = conv2d_ref(x)

    # Since our kernel ignores the groups parameter, the outputs should not match.
    if torch.allclose(custom_out, ref_out, atol=1e-4):
        pytest.fail("Custom CUDA kernel incorrectly handled groups. Expected mismatch because groups were ignored.")
