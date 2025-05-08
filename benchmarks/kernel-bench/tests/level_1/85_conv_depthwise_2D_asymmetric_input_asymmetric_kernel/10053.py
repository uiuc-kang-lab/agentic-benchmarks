
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Make sure the absolute path is used in case of relative path issues.
    cu_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="test_module",
        sources=[cu_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Data type mismatch (kernel only supports float32)
def test_dtype_issue():
    # Create input tensors with double precision.
    batch_size = 1
    in_channels = 3
    height = 10
    width = 10
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.double, device="cuda")
    # Create a dummy weight tensor: for depthwise convolution, weight shape: [groups, channels_per_group, kernel_h, kernel_w]
    kernel_h, kernel_w = 3, 3
    # For depthwise conv with groups==in_channels, channels_per_group is usually 1.
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, dtype=torch.double, device="cuda")
    # Bias tensor (optional)
    bias = torch.randn(in_channels, dtype=torch.double, device="cuda")
    
    mod = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The kernel expects float32 tensors. This should trigger an error or misbehavior.
        mod.forward(x, weight, bias, 1, 1, 0, 0, 1, 1, in_channels)

# Issue 2: Warp-level reduction with a partial warp.
def test_incomplete_warp_issue():
    # Create an input which, after convolution, produces only one output pixel (thus less than one full warp)
    # This will force the kernel to compute reduction within a warp that is not fully active.
    batch_size = 1
    in_channels = 1  # Also groups==in_channels for this simple test.
    height = 3
    width = 3
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    kernel_h, kernel_w = 3, 3
    # Generate weight so that the output is a single pixel (full coverage)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, dtype=torch.float32, device="cuda")
    # No bias in this test.
    
    mod = build_kernel()
    
    # With valid parameters, the output spatial dimensions become 1x1.
    # Because there is only one output pixel per channel, the warp-level reduction will be performed with only one active lane,
    # but the fixed mask in __shfl_down_sync can produce an incorrect result.
    out = mod.forward(x, weight, None, 1, 1, 0, 0, 1, 1, in_channels)
    
    # We compute a reference result using PyTorch's convolution for comparison.
    conv_ref = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=in_channels)
    
    # We do not expect the results to match because of the warp reduction bug.
    assert not torch.allclose(out, conv_ref, atol=1e-4), "Output unexpectedly matches reference, warp reduction issue not triggered"

# Issue 3: Wrong indexing when groups != in_channels.
def test_groups_issue():
    # For a group convolution that is not depthwise (i.e. groups != in_channels),
    # the kernel’s assumption about input channel indexing will be wrong.
    batch_size = 1
    in_channels = 4  # Let there be 4 input channels.
    groups = 2     # But use 2 groups.
    # For a standard conv2d with groups != in_channels, weight shape is [out_channels, in_channels/groups, kh, kw].
    # However, our custom kernel expects a depthwise layout where groups==in_channels.
    height = 8
    width = 8
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    kernel_h, kernel_w = 3, 3
    # For the misguided use-case, out_channels = groups * channels_per_group.
    # We set channels_per_group arbitrarily, e.g., 2, so out_channels=4.
    channels_per_group = 2
    out_channels = groups * channels_per_group
    weight = torch.randn(out_channels // groups, kernel_h, kernel_w, dtype=torch.float32, device="cuda")
    # Rearrange weight to a 4D tensor as expected by our kernel: (groups, channels_per_group, kh, kw)
    weight = weight.view(groups, channels_per_group, kernel_h, kernel_w)
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    
    mod = build_kernel()
    
    # Launch the kernel with groups != in_channels.
    out = mod.forward(x, weight, bias, 1, 1, 1, 1, 1, 1, groups)
    
    # Since our kernel incorrectly assumes depthwise convolution (in_channels==groups) the output will differ
    # from what PyTorch's group convolution returns.
    conv_ref = torch.nn.functional.conv2d(x, weight, bias=bias, stride=1, padding=1, dilation=1, groups=groups)
    
    assert not torch.allclose(out, conv_ref, atol=1e-4), "Output unexpectedly matches reference, groups indexing issue not triggered"

# Issue 4: Incorrect handling of non-contiguous input tensors.
def test_non_contiguous_input():
    # Create a contiguous tensor and then create a non-contiguous view using .transpose()
    batch_size = 2
    in_channels = 3
    height = 10
    width = 10
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    # Create a non-contiguous tensor by transposing spatial dimensions (or channel and spatial dims)
    x_nc = x.transpose(2, 3)  # Now shape is (batch, in_channels, width, height) and non-contiguous.
    
    kernel_h, kernel_w = 3, 3
    # The kernel expects a tensor with layout (batch, channels, height, width)
    weight = torch.randn(in_channels, 1, kernel_h, kernel_w, dtype=torch.float32, device="cuda")
    bias = torch.randn(in_channels, dtype=torch.float32, device="cuda")
    
    mod = build_kernel()
    
    # Running the kernel with non-contiguous input will compute incorrect flat indices.
    out = mod.forward(x_nc, weight, bias, 1, 1, 1, 1, 1, 1, in_channels)
    
    # For reference, we expect PyTorch's conv2d to work fine even with non-contiguous inputs.
    conv_ref = torch.nn.functional.conv2d(x_nc, weight, bias=bias, stride=1, padding=1, dilation=1, groups=in_channels)
    
    assert not torch.allclose(out, conv_ref, atol=1e-4), "Output unexpectedly matches reference, non-contiguous input issue not triggered"

if __name__ == "__main__":
    pytest.main([__file__])
