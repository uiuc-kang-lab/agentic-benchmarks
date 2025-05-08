
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel assumes float32 input.
# This test passes half precision tensors to the extension.
def test_invalid_dtype():
    cuda_module = build_kernel()
    # Build a minimal configuration for a transposed convolution.
    batch = 2
    in_channels = 4
    out_channels = 4
    in_h, in_w = 8, 8
    kernel_h, kernel_w = 3, 3
    stride = [2, 2]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    # Create inputs in half precision.
    x = torch.randn(batch, in_channels, in_h, in_w, device="cuda", dtype=torch.float16)
    # Expected weight shape: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    weight = torch.randn(in_channels, out_channels // groups, kernel_h, kernel_w, device="cuda", dtype=torch.float16)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float16)
    
    # We expect the kernel to misbehave when using fp16 (or at least not produce comparable results).
    with pytest.raises(RuntimeError):
        # If the kernel silently runs and returns wrong results this test may fail.
        output = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()
    
# Issue 2: Kernel assumes in_channels is divisible by groups.
def test_in_channels_not_divisible_by_groups():
    cuda_module = build_kernel()
    # Choose in_channels not divisible by groups, e.g. 10 channels with groups=3.
    batch = 2
    in_channels = 10  # Not divisible by groups.
    out_channels = 6  # Use out_channels such that weight shape is consistent.
    # For the weight the expected shape is [in_channels, out_channels_per_group, kernel_h, kernel_w].
    # Let out_channels_per_group = out_channels // groups; so use groups = 3 and out_channels = 6 to get 2 per group.
    groups = 3
    kernel_h, kernel_w = 3, 3
    in_h, in_w = 8, 8
    stride = [2, 2]
    padding = [1, 1]
    dilation = [1, 1]

    x = torch.randn(batch, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    # Weight: the kernel expects shape [in_channels, out_channels//groups, kernel_h, kernel_w].
    weight = torch.randn(in_channels, out_channels // groups, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Run the extension and also compare with PyTorch's reference.
    # PyTorch's nn.ConvTranspose2d requires that in_channels % groups == 0 so we wrap in a try/except.
    with pytest.raises(Exception):
        # Our custom kernel will follow the truncated division (10//3 == 3) and hence produce an output 
        # that does not match the expected result. Therefore, we expect either an error or a mismatch.
        output = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()
        # In case no error is raised, we compare with PyTorch (which should error on invalid group configuration)
        ref_conv = torch.nn.ConvTranspose2d(in_channels, out_channels, (kernel_h, kernel_w), 
                                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True).cuda()
        # Force same weight and bias (if possible)
        ref_conv.weight.data = weight
        ref_conv.bias.data = bias
        ref_output = ref_conv(x)
        # The outputs should differ.
        assert not torch.allclose(output, ref_output, atol=1e-3), "Output unexpectedly matches even with channels/groups configuration error."

# Issue 3: Kernel does not verify bias shape.
def test_invalid_bias_shape():
    cuda_module = build_kernel()
    # Create valid input configuration.
    batch = 2
    in_channels = 4
    out_channels = 4
    kernel_h, kernel_w = 3, 3
    in_h, in_w = 8, 8
    stride = [2, 2]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    x = torch.randn(batch, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    # Provide a bias tensor with an incorrect shape.
    bias = torch.randn(out_channels - 1, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        output = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()
