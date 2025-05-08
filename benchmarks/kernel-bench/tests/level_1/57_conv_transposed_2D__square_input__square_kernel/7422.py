
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Build the extension module from the kernel.cu file
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test for double bias addition.
def test_double_bias_addition():
    # Use parameters with bias
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    # Input and parameters
    x = torch.randn(batch_size, in_channels, 16, 16, device='cuda', dtype=torch.float32)
    # Weight shape for conv_transpose2d: (in_channels, out_channels/groups, kernel_size, kernel_size)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # Compute output using the extension. Note: The extension calls at::conv_transpose2d
    # with bias and then adds bias again in the custom kernel.
    my_module = build_kernel()
    ext_output = my_module.forward(x, weight, bias, stride, padding, output_padding, groups)

    # Compute expected output manually by disabling bias in conv_transpose2d and then
    # adding bias once.
    out_without_bias = F.conv_transpose2d(x, weight, bias=None, stride=stride, padding=padding,
                                          output_padding=output_padding, groups=groups)
    expected_output = out_without_bias + bias.view(1, -1, 1, 1)

    # Since the extension adds the bias twice, its output should equal expected_output + bias.
    double_bias_output = expected_output + bias.view(1, -1, 1, 1)

    # Verify that the extension output is closer to the double-biased result.
    assert torch.allclose(ext_output, double_bias_output, atol=1e-5), (
        "Double bias addition issue not detected. The output did not include an extra bias."
    )

# Issue 2: Test for data type limitations.
def test_dtype_not_float32():
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    # Create half precision tensor (float16), which is not supported by our kernel.
    x = torch.randn(batch_size, in_channels, 16, 16, device='cuda', dtype=torch.float16)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device='cuda', dtype=torch.float16)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float16)

    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Our kernel calls data_ptr<float>(), so using a non-float32 tensor should lead to an error.
        my_module.forward(x, weight, bias, stride, padding, output_padding, groups)

# Issue 3: Test for incorrect bias indexing in grouped convolutions.
def test_grouped_convolution_bias_indexing():
    batch_size = 2
    in_channels = 4
    out_channels = 8  # For groups > 1, out_channels must be divisible by groups.
    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 2  # More than one group

    x = torch.randn(batch_size, in_channels, 16, 16, device='cuda', dtype=torch.float32)
    # Weight shape for conv_transpose2d: (in_channels, out_channels/groups, kernel_size, kernel_size)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    my_module = build_kernel()
    ext_output = my_module.forward(x, weight, bias, stride, padding, output_padding, groups)

    # Compute expected result by separately applying bias correctly. Disable built-in bias in the conv_transpose2d,
    # then add bias once.
    out_without_bias = F.conv_transpose2d(x, weight, bias=None, stride=stride, padding=padding,
                                          output_padding=output_padding, groups=groups)
    expected_output = out_without_bias + bias.view(1, -1, 1, 1)

    # Now, note that the custom kernel will incorrectly compute the bias index when groups > 1.
    # Here we assert that the output from the extension is different from the correctly computed output.
    # We check that the maximum absolute difference is significant.
    max_diff = (ext_output - expected_output).abs().max().item()
    assert max_diff > 1e-3, (
        f"Grouped convolution bias indexing issue not detected. Max difference: {max_diff}"
    )
