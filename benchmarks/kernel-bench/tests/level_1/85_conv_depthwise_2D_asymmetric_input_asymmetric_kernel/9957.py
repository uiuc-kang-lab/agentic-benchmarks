
import torch
import pytest
from torch import nn
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Grouped convolution input indexing for channels_per_group > 1.
def test_grouped_input_indexing():
    # Create a grouped convolution where each group has 2 channels.
    # This will trigger the incorrect input indexing.
    batch_size = 2
    in_channels = 4  # e.g. 2 groups of 2 channels each
    groups = 2
    channels_per_group = in_channels // groups  # =2
    kernel_size_h = 3
    kernel_size_w = 3
    height = 10
    width = 10
    stride = 1
    padding = 1
    dilation = 1

    # Create input tensor
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Weight tensor in PyTorch for grouped conv is normally 4D: [groups, channels_per_group, kernel_h, kernel_w]
    weight = torch.randn(groups, channels_per_group, kernel_size_h, kernel_size_w, device="cuda", dtype=torch.float32)
    # Bias tensor with one bias per output channel.
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Create reference convolution using PyTorch's Conv2d with groups>1
    conv = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w),
                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
    # Manually set weights and bias to the same as the tensors above.
    # Because PyTorch expects weight shape [in_channels, 1, kernel_h, kernel_w] for depthwise
    # and [groups, channels_per_group, kernel_h, kernel_w] for grouped convolution,
    # we assign weight and bias.
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    ref_out = conv(x)

    # Run the custom CUDA kernel.
    cuda_module = build_kernel()
    out = cuda_module.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)

    # This test is expected to fail (i.e. output differs from torch.nn.Conv2d) because of input indexing error.
    assert not torch.allclose(out, ref_out, atol=1e-3), "The kernel output unexpectedly matches the reference output (input indexing issue not triggered)"

# Test 2: Weight tensor layout assumption.
def test_weight_tensor_layout():
    # Here we intentionally pass a 4D weight tensor in the conventional PyTorch layout
    # to trigger the issue in the weight indexing logic.
    batch_size = 2
    in_channels = 3  # using depthwise setting
    groups = in_channels  # depthwise conv
    kernel_size_h = 3
    kernel_size_w = 5
    height = 16
    width = 16
    stride = 1
    padding = 0
    dilation = 1

    # Generate input.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Weight in conventional PyTorch depthwise conv shape: [in_channels, 1, kernel_h, kernel_w]
    weight = torch.randn(in_channels, 1, kernel_size_h, kernel_size_w, device="cuda", dtype=torch.float32)
    bias = None

    # Reference convolution.
    conv = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w),
                     stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
    with torch.no_grad():
        conv.weight.copy_(weight)
    ref_out = conv(x)

    # Run our kernel: our kernel expects weight to be in a flattened 3D layout.
    cuda_module = build_kernel()
    out = cuda_module.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)

    # The output is expected to differ because of the wrong weight indexing.
    assert not torch.allclose(out, ref_out, atol=1e-3), "The kernel output unexpectedly matches the reference output (weight layout issue not triggered)"

# Test 3: Data type handling (kernel only supports float32).
def test_data_type_handling():
    batch_size = 2
    in_channels = 3
    groups = in_channels  # depthwise conv, channels_per_group=1
    kernel_size_h = 3
    kernel_size_w = 3
    height = 16
    width = 16
    stride = 1
    padding = 0
    dilation = 1

    # Create input with double precision.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.double)
    # Weight as double, following PyTorch's 4D layout.
    weight = torch.randn(in_channels, 1, kernel_size_h, kernel_size_w, device="cuda", dtype=torch.double)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.double)

    # Run the custom kernel; since the kernel calls data_ptr<float>(),
    # interpreting double data as float, the results should be incorrect.
    cuda_module = build_kernel()
    out = cuda_module.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)
    
    # Use a reference convolution with proper type conversion.
    conv = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w),
                     stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True)
    # Convert weight and bias to float32 for the reference conv.
    with torch.no_grad():
        conv.weight.copy_(weight.float())
        conv.bias.copy_(bias.float())
    ref_out = conv(x.float())
    
    # The outputs are likely to be different due to type misinterpretation.
    assert not torch.allclose(out, ref_out, atol=1e-3), "The kernel output unexpectedly matches the reference output (data type issue not triggered)"

# Test 4: Shared memory reduction assumptions.
def test_shared_memory_reduction():
    # To trigger the shared memory reduction path, we need a "large" kernel.
    # We choose kernel dimensions that exceed the SMALL_KERNEL_THRESHOLD.
    batch_size = 1
    in_channels = 3  # depthwise conv
    groups = in_channels
    kernel_size_h = 8
    kernel_size_w = 8  # 64 elements in kernel; typically this forces the shared memory kernel branch.
    height = 32
    width = 32
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # For depthwise conv, PyTorch weight shape: [in_channels, 1, kernel_h, kernel_w]
    weight = torch.randn(in_channels, 1, kernel_size_h, kernel_size_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Reference convolution using nn.Conv2d.
    conv = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w),
                     stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=True)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    ref_out = conv(x)
    
    cuda_module = build_kernel()
    out = cuda_module.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)
    
    # Because of mismatches in shared memory reduction assumptions, we expect a discrepancy.
    assert not torch.allclose(out, ref_out, atol=1e-3), "The kernel output unexpectedly matches the reference output (shared memory reduction issue not triggered)"
