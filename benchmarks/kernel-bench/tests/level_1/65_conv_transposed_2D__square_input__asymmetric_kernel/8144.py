
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to compile and load the custom CUDA kernel.
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue with load imbalance by using a large spatial output.
def test_large_spatial_output():
    # Use parameters that make the spatial output very large.
    batch_size = 2
    in_channels = 8
    out_channels = 8
    in_height = in_width = 16
    kernel_h, kernel_w = 3, 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    
    # Build custom weight of shape [in_channels, out_channels/groups, kernel_h, kernel_w]
    weight = torch.randn(in_channels, out_channels // groups, kernel_h, kernel_w, device='cuda').float()
    bias = torch.randn(out_channels, device='cuda').float()
    
    # Create a large input (spatially)
    x = torch.randn(batch_size, in_channels, in_height, in_width, device='cuda').float()
    # Increase spatial size artificially by upsampling input for testing (simulate stress case).
    x = torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
    
    # Compute expected output using torch.nn.functional.conv_transpose2d
    expected = torch.nn.functional.conv_transpose2d(
        x, weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups
    )
    
    # Compute output from our kernel (dilation set to 1)
    cuda_module = build_kernel()
    output = cuda_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation=1)
    torch.cuda.synchronize()
    
    # The two outputs may differ if load imbalance causes numerical issues.
    assert torch.allclose(output, expected, atol=1e-4), "Mismatch in output for large spatial input (load imbalance issue)."

# Test 2: Trigger branch divergence and index arithmetic issues using non‐unit stride and dilation.
def test_stride_dilation_edge_case():
    batch_size = 2
    in_channels = 4
    out_channels = 4
    in_height = in_width = 10
    kernel_h, kernel_w = 3, 5  # asymmetric kernel
    stride = 2
    padding = 1
    output_padding = 1
    groups = 1
    dilation = 2  # non-unit dilation
    
    weight = torch.randn(in_channels, out_channels // groups, kernel_h, kernel_w, device='cuda').float()
    bias = torch.randn(out_channels, device='cuda').float()
    x = torch.randn(batch_size, in_channels, in_height, in_width, device='cuda').float()
    
    expected = torch.nn.functional.conv_transpose2d(
        x, weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, dilation=dilation
    )
    
    cuda_module = build_kernel()
    output = cuda_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation=dilation)
    torch.cuda.synchronize()
    
    assert torch.allclose(output, expected, atol=1e-4), "Mismatch in output for non-unit stride/dilation (index arithmetic issue)."

# Test 3: Trigger issue with bias handling by providing a bias vector with correct and incorrect size.
def test_bias_handling():
    batch_size = 2
    in_channels = 4
    out_channels = 6  # deliberately choose non-standard ratio when groups=1.
    in_height = in_width = 8
    kernel_h, kernel_w = 3, 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1
    
    weight = torch.randn(in_channels, out_channels // groups, kernel_h, kernel_w, device='cuda').float()
    # Correct bias size
    bias = torch.randn(out_channels, device='cuda').float()
    x = torch.randn(batch_size, in_channels, in_height, in_width, device='cuda').float()
    
    expected = torch.nn.functional.conv_transpose2d(
        x, weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, dilation=dilation
    )
    
    cuda_module = build_kernel()
    output = cuda_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation=dilation)
    torch.cuda.synchronize()
    
    assert torch.allclose(output, expected, atol=1e-4), "Kernel failed to handle bias correctly (bias handling issue)."
    
    # Now test with a bias of incorrect size (should trigger an error check)
    wrong_bias = torch.randn(out_channels + 1, device='cuda').float()
    with pytest.raises(RuntimeError):
        # Our kernel checks the bias size, so this launch is expected to fail.
        _ = cuda_module.forward(x, weight, wrong_bias, stride, padding, output_padding, groups, dilation=dilation)

# Test 4: Trigger lack of error checking by forcing a kernel configuration error.
def test_no_error_checking():
    # Here, we purposefully set up an invalid configuration (e.g., stride==0) 
    # to see if the kernel misbehaves or silently produces wrong output.
    # (A proper implementation would check for stride > 0.)
    batch_size = 1
    in_channels = 2
    out_channels = 2
    in_height = in_width = 5
    kernel_h, kernel_w = 3, 3
    stride = 0  # invalid stride
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1
    
    weight = torch.randn(in_channels, out_channels // groups, kernel_h, kernel_w, device='cuda').float()
    bias = torch.randn(out_channels, device='cuda').float()
    x = torch.randn(batch_size, in_channels, in_height, in_width, device='cuda').float()
    
    cuda_module = build_kernel()
    # Without error checking, the kernel launch might happen and produce an invalid output.
    output = cuda_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation=dilation)
    torch.cuda.synchronize()
    
    # If the invalid configuration is not caught, we compare against a dummy expected value.
    # This test is designed to fail if the kernel does not report an error.
    expected = torch.empty(0, device='cuda')
    assert not output.numel(), "Kernel did not error out for invalid configuration (lack of error checking)."
