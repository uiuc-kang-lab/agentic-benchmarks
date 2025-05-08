
import pytest
import torch
from torch.utils.cpp_extension import load
import torch.nn as nn

# Helper function to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_conv3d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper to compute reference convolution using PyTorch's nn.Conv3d
def ref_conv3d(input, weight, bias, stride, padding, dilation, groups):
    conv = nn.Conv3d(
        in_channels=input.size(1),
        out_channels=weight.size(0),
        kernel_size=(weight.size(2), weight.size(3), weight.size(4)),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=(bias is not None)
    ).cuda()
    # Set conv weights to our test weights and bias for a fair comparison.
    with torch.no_grad():
        conv.weight.copy_(weight)
        if bias is not None:
            conv.bias.copy_(bias)
    return conv(input)

# Test case 1: Trigger unsupported data types (e.g. float64)
def test_unsupported_dtype():
    my_module = build_kernel()
    # Create input and weight in double precision
    batch_size, in_channels, D, H, W = 2, 3, 10, 10, 10
    out_channels = 4
    kernel_size = (3, 3, 3)
    
    input = torch.randn(batch_size, in_channels, D, H, W, device='cuda', dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float64)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float64)
    
    stride = 1; padding = 1; dilation = 1; groups = 1

    # Expect the kernel build (using float conversion) to fail or produce wrong results.
    with pytest.raises(Exception):
        # The kernel is only compiled for float; the following call should raise an error or crash.
        _ = my_module.forward(input, weight, bias, stride, padding, dilation, groups)
        

# Test case 2: Trigger issue with non-contiguous inputs
def test_non_contiguous_input():
    my_module = build_kernel()
    batch_size, in_channels, D, H, W = 2, 3, 10, 10, 10
    out_channels = 4
    kernel_size = (3, 3, 3)
    
    # Create contiguous tensors first then make them non-contiguous via transpose.
    input = torch.randn(batch_size, in_channels, D, H, W, device='cuda', dtype=torch.float32).transpose(1,2)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    stride = 1; padding = 1; dilation = 1; groups = 1

    # The kernel does not verify contiguity and may lead to incorrect computations.
    output_kernel = my_module.forward(input, weight, bias, stride, padding, dilation, groups)
    output_ref = ref_conv3d(input.contiguous(), weight, bias, stride, padding, dilation, groups)
    # Since input is non-contiguous, the kernel may produce incorrect result.
    with pytest.raises(AssertionError):
        assert torch.allclose(output_kernel, output_ref, atol=1e-4), "Output should differ due to non-contiguous input"


# Test case 3: Trigger issue with asymmetric convolution parameters.
# The kernel only accepts a single int for stride/padding/dilation. 
# In a more general setting one may wish to use a tuple for asymmetric behavior.
def test_asymmetric_convolution_parameters():
    my_module = build_kernel()
    batch_size, in_channels, D, H, W = 2, 3, 12, 12, 12
    out_channels = 4
    kernel_size = (3, 5, 7)  # Asymmetric kernel as in the provided PyTorch code
    
    input = torch.randn(batch_size, in_channels, D, H, W, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    # Let’s say the intended behavior is to have different stride per dimension. 
    # But our kernel accepts only one integer. We simulate the issue by comparing to a reference.
    stride = 2
    padding = 1
    dilation = 1
    groups = 1

    output_kernel = my_module.forward(input, weight, bias, stride, padding, dilation, groups)
    output_ref = ref_conv3d(input, weight, bias, stride, padding, dilation, groups)
    
    # They likely will differ because the kernel is not implemented to handle per-dim parameters.
    with pytest.raises(AssertionError):
        assert torch.allclose(output_kernel, output_ref, atol=1e-4), "Kernel does not support asymmetric convolution parameters correctly"


# Test case 4: Trigger potential issues with fixed block/warp configuration.
# We create a tensor size that forces a very low number of output elements 
# (e.g., a very small output volume) so that the rigid block/warp scheduling might lead to incorrect mapping.
def test_fixed_block_configuration():
    my_module = build_kernel()
    # Create an input that produces a minimal output spatial dimension.
    batch_size, in_channels = 1, 3
    D, H, W = 5, 5, 5
    out_channels = 2
    kernel_size = (3, 3, 3)
    
    input = torch.randn(batch_size, in_channels, D, H, W, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    
    # With stride=1, padding=0, dilation=1 the output will be very small.
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    output_kernel = my_module.forward(input, weight, bias, stride, padding, dilation, groups)
    output_ref = ref_conv3d(input, weight, bias, stride, padding, dilation, groups)
    
    with pytest.raises(AssertionError):
        assert torch.allclose(output_kernel, output_ref, atol=1e-4), "Kernel output differs due to rigid block/warp configuration assumptions"

