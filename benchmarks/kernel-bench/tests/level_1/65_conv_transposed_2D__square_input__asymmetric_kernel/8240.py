
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper to build the module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Trigger issue with constant memory bias overflow
def test_constant_bias_overflow():
    # Set parameters so that out_channels > 1024, thus triggering the bias constant array size issue.
    batch_size = 1
    in_channels = 4
    out_channels = 1025  # >1024: channel 1024 will not have bias added.
    kernel_size = (3, 3)
    stride = 2
    padding = 1
    output_padding = 1
    groups = 1
    dilation = 1

    # Input tensor
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    
    # Weight: shape = (in_channels, out_channels/groups, kernel_h, kernel_w)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    
    # Bias: known pattern so we can check its effect.
    bias = torch.arange(0, out_channels, device="cuda", dtype=torch.float32)
    
    # Run custom kernel
    custom_module = build_kernel()
    out_custom = custom_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
    
    # Run PyTorch reference: using nn.ConvTranspose2d. Note that PyTorch ConvTranspose2d
    # applies the bias to every channel. Since our kernel only adds bias for channels <1024,
    # we expect a mismatch especially for channel index 1024.
    conv_trans = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding,
                                          output_padding=output_padding, groups=groups, bias=True).to("cuda")
    # Manually set weights and bias to the same values.
    with torch.no_grad():
        conv_trans.weight.copy_(weight)
        conv_trans.bias.copy_(bias)
    out_reference = conv_trans(x)
    
    # Check that the last channel’s bias is not applied in the custom kernel output.
    # We compare one spatial location for channel index 1024.
    b, oc, h, w = out_custom.shape
    # Using the center element for simplicity.
    center = (h // 2, w // 2)
    custom_val = out_custom[0, 1024, center[0], center[1]]
    ref_val = out_reference[0, 1024, center[0], center[1]]
    # Because the bias for channel 1024 is not added in the custom kernel, the difference should be close to bias[1024]
    diff = abs(custom_val - ref_val)
    expected_diff = abs(bias[1024])
    assert diff > expected_diff * 0.5, "Bias was unexpectedly applied to channel 1024 (should be missing due to constant memory size)."

# Test 2: Trigger issue with unsupported bias data type
def test_bias_dtype_incompatibility():
    batch_size = 1
    in_channels = 4
    out_channels = 8
    kernel_size = (3, 3)
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, 10, 10, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    # Provide bias as double instead of float; the kernel expects float.
    bias = torch.randn(out_channels, device="cuda", dtype=torch.double)

    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        # We expect a runtime error or misbehaviour when passing bias with incompatible type.
        custom_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
    
# Test 3: Trigger issue with missing kernel flip in spatial dimensions
def test_missing_kernel_flip():
    # Use an asymmetric (non-symmetric) kernel so that flipping matters.
    batch_size = 1
    in_channels = 3
    out_channels = 6
    kernel_size = (3, 5)  # non-symmetric kernel: flipping changes the kernel.
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, 12, 12, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    custom_module = build_kernel()
    out_custom = custom_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
    
    # For a reference, we compute the output of a properly implemented conv_transpose2d that flips the spatial kernel.
    # Manually flip the spatial dimensions of the weight.
    weight_flipped = weight.flip(dims=[2, 3])
    # Build a reference model.
    conv_trans = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding,
                                          output_padding=output_padding, groups=groups, bias=True).to("cuda")
    with torch.no_grad():
        conv_trans.weight.copy_(weight_flipped)
        conv_trans.bias.copy_(bias)
    out_reference = conv_trans(x)

    # Because the custom kernel did not flip the kernel weights, the outputs should differ noticeably.
    max_diff = (out_custom - out_reference).abs().max().item()
    assert max_diff > 1e-3, f"Custom kernel output unexpectedly matches reference (max diff {max_diff})."

if __name__ == "__main__":
    pytest.main([__file__])
