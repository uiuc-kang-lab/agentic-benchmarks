
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Helper function to compile and load the extension
def build_kernel():
    cuda_module = load(
        name="test_transposed_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Trigger issue with non float32 type
def test_dtype_issue():
    # Create inputs with double precision, which our kernel does not support.
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    height, width = 8, 8
    stride = (2, 2)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1

    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float64, device="cuda")
    # Weight layout as assumed: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float64, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda")

    kernel_mod = build_kernel()
    # Expect the kernel to silently use float32 (by reinterpretation or conversion error)
    # Since our kernel only works on float32, the output will be wrong.
    with pytest.raises(Exception):
        # This might throw an error or produce a wrong result.
        kernel_mod.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)

# Test 2: Trigger issue with non-contiguous weight tensor
def test_non_contiguous_weight_issue():
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    height, width = 8, 8
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1

    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    # Create a weight tensor and make it non-contiguous by transposing two kernel dimensions.
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float32, device="cuda")
    weight_non_contig = weight.transpose(2, 3)  # Now the layout is {in_channels, out_channels_per_group, kernel_w, kernel_h}
    # Although the values are the same, the kernel indexing assumes a particular contiguous layout.
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")

    kernel_mod = build_kernel()
    # Run our kernel. The output is expected to be wrong because the weight layout is not as assumed.
    output = kernel_mod.forward(x, weight_non_contig, bias, list(stride), list(padding), list(dilation), groups)
    
    # Use PyTorch's built-in ConvTranspose2d as reference (which internally makes weights contiguous)
    conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation,
                                    groups=groups, bias=True).cuda()
    # Force conv_trans weights to be the original contiguous weight so that we know the reference value.
    with torch.no_grad():
        conv_trans.weight.copy_(weight)
        conv_trans.bias.copy_(bias)
    ref_output = conv_trans(x)
    # The outputs should not match due to mis-indexing in the custom kernel.
    assert not torch.allclose(output, ref_output, atol=1e-4), "Kernel unexpectedly produced the correct output with non-contiguous weight."

# Test 3: Trigger issue with in_channels not divisible by groups
def test_groups_issue():
    batch_size = 2
    in_channels = 5  # not divisible by groups
    out_channels = 4  # arbitrary
    kernel_size = (3, 3)
    height, width = 8, 8
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 2  # 5 % 2 != 0

    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    # Weight shape based on assumption: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    # Here, out_channels_per_group = out_channels / groups. We simulate a broken scenario.
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")

    kernel_mod = build_kernel()
    # This case leads to an inconsistent in_channels per group value.
    # We expect that the results will be incorrect.
    output = kernel_mod.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)
    
    # Using PyTorch's ConvTranspose2d will raise an error in its own check.
    with pytest.raises(RuntimeError):
        conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation,
                                        groups=groups, bias=True).cuda()
        _ = conv_trans(x)

# Test 4: Trigger potential index overflow by simulating very large dimensions
def test_index_overflow_issue():
    # Instead of using extremely large tensors that could crash the machine,
    # we simulate the failure by providing dimensions that when multiplied,
    # approach the limit of 32-bit int arithmetic.
    #
    # Note: This test is conceptual; in practice the hardware might not permit
    # such large allocations, but we can check that our kernel (which uses int)
    # would be at risk.
    batch_size = 1
    in_channels = 1024  # large number of channels
    out_channels = 1024
    kernel_size = (3, 3)
    # Choose spatial dims such that in_h*in_w becomes large
    height, width = 500, 500  # 250000 pixels, multiplied by channels leads to potential overflow in 32-bit int.
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1

    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")

    kernel_mod = build_kernel()
    # Although the kernel may run without an immediate error,
    # numerical overflow in the index computation might produce garbage output.
    output = kernel_mod.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)
    
    # Compare with PyTorch's ConvTranspose2d which uses proper 64-bit indexing internally.
    conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation,
                                    groups=groups, bias=True).cuda()
    with torch.no_grad():
        conv_trans.weight.copy_(weight)
        conv_trans.bias.copy_(bias)
    ref_output = conv_trans(x)
    # The outputs are expected to differ significantly due to indexing issues.
    assert not torch.allclose(output, ref_output, atol=1e-4), "Kernel output unexpectedly matches reference output despite potential index overflow issues."
