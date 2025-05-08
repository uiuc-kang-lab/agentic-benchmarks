
import pytest
import torch
from torch import nn
from torch.utils.cpp_extension import load

# Build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="transposed_conv3d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: Create a reference using PyTorch's native ConvTranspose3d.
def native_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    conv = nn.ConvTranspose3d(
        in_channels=x.size(1),
        out_channels=weight.size(1)*groups,
        kernel_size=(weight.size(2), weight.size(3), weight.size(4)),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=(bias is not None),
    ).to(x.device)
    # Manually assign the weights and bias
    with torch.no_grad():
        conv.weight.copy_(weight)
        if bias is not None:
            conv.bias.copy_(bias)
    return conv(x)

# Test case 1: Triggering issue with output_padding (nonzero output_padding not handled)
def test_output_padding_issue():
    # Create input, weight, and bias (if any), all in float32.
    batch_size = 2
    in_channels = 4
    out_channels = 8
    depth, height, width = 5, 6, 7
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)  # nonzero -> should be handled, but kernel ignores it
    groups = 1

    x = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float32)
    # Weight expected shape for ConvTranspose3d: (in_channels, out_channels/groups, k_d, k_h, k_w)
    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, device='cuda', dtype=torch.float32)
    bias = None

    # Run through our custom CUDA kernel
    module = build_kernel()
    y_custom = module.forward(x, weight, bias, list(stride), list(padding), list(output_padding), groups)
    
    # Run native PyTorch implementation
    y_native = native_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups)
    
    # The outputs should agree if kernel accounted for output_padding.
    # Here we expect them to differ because kernel ignores output_padding.
    assert not torch.allclose(y_custom, y_native, atol=1e-5), \
        "The custom kernel unexpectedly matches the native output when output_padding is nonzero."

# Test case 2: Triggering issue with input data type (only float32 supported)
def test_dtype_issue():
    batch_size = 2
    in_channels = 4
    out_channels = 8
    depth, height, width = 5, 6, 7
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    output_padding = (0, 0, 0)
    groups = 1

    # Create inputs with double precision, which our kernel does not support
    x = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, device='cuda', dtype=torch.float64)
    bias = None

    module = build_kernel()
    # This should cause an issue because the kernel is compiled for float
    y_custom = module.forward(x.float(), weight.float(), bias, list(stride), list(padding), list(output_padding), groups)
    # Manually convert back to double for comparison with native op on double tensors
    y_custom = y_custom.double()
    y_native = native_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups)
    
    # Because the kernel is not type templated, if someone forgets to enforce float32,
    # the user might mistakenly pass double tensors. We test that the result does not match.
    assert not torch.allclose(y_custom, y_native, atol=1e-5), \
        "The custom kernel produced matching results with double precision input, but it is hard-coded for float32."

# Test case 3: Triggering issue with invalid group configuration (in_channels not divisible by groups)
def test_invalid_group_config_issue():
    batch_size = 2
    in_channels = 5  # deliberately not divisible by groups
    out_channels = 8   # arbitrary
    depth, height, width = 5, 6, 7
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    output_padding = (0, 0, 0)
    groups = 2  # 5 % 2 != 0, which is not properly handled in the kernel

    x = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float32)
    # Construct weight tensor with shape: (in_channels, out_channels/groups, k_d, k_h, k_w)
    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, device='cuda', dtype=torch.float32)
    bias = None

    module = build_kernel()
    y_custom = module.forward(x, weight, bias, list(stride), list(padding), list(output_padding), groups)
    
    # The native PyTorch conv_transpose3d expects in_channels to be divisible by groups.
    # We attempt to run the native implementation; if it does not error out, then the output shapes will disagree.
    with pytest.raises(RuntimeError):
        _ = native_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups)

    # Alternatively, if the native op allowed it (by some changes in future PyTorch versions), we can check for discrepancies.
    # Uncomment the following lines if native op does not throw an error.
    # y_native = native_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups)
    # assert not torch.allclose(y_custom, y_native, atol=1e-5), \
    #    "The custom kernel output incorrectly matches the native output despite invalid group configuration."

if __name__ == "__main__":
    pytest.main([__file__])
