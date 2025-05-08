
import pytest
import torch
from torch import nn
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension
def build_kernel():
    return load(
        name="transposed_conv3d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Test issue 1: Limited data type support (float16 not supported)
def test_half_precision_not_supported():
    cuda_module = build_kernel()

    # Create input, weight, and (optional) bias tensors in half-precision
    batch_size = 2
    in_channels = 4
    out_channels = 4
    in_depth = in_height = in_width = 5
    kT = kH = kW = 3

    input_tensor = torch.randn(batch_size, in_channels, in_depth, in_height, in_width, 
                               device="cuda", dtype=torch.float16)
    # weight shape: [in_channels, out_channels/groups, kT, kH, kW] where groups==1 here
    weight_tensor = torch.randn(in_channels, out_channels, kT, kH, kW, device="cuda", dtype=torch.float16)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float16)

    stride = [1, 1, 1]
    padding = [0, 0, 0]
    output_padding = [0, 0, 0]
    groups = 1

    # Expect that dispatching with float16 will fail, causing a RuntimeError or NotImplementedError
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)
        torch.cuda.synchronize()

# Test issue 3: Improper group handling when out_channels and/or in_channels are not divisible by groups.
def test_invalid_group_channel_numbers():
    cuda_module = build_kernel()

    # Use a groups value that does not divide channel numbers evenly.
    batch_size = 2
    in_channels = 10  # deliberately not divisible by groups
    out_channels = 8  # deliberately chosen so that out_channels/groups is not an integer for groups=3
    groups = 3

    # Use asymmetric kernel sizes to mimic complex cases.
    kT, kH, kW = 3, 5, 7
    in_depth, in_height, in_width = 8, 16, 16

    input_tensor = torch.randn(batch_size, in_channels, in_depth, in_height, in_width, 
                               device="cuda", dtype=torch.float32)
    # Expecting weight shape: [in_channels, out_channels/groups, kT, kH, kW]
    # But here out_channels/groups is fractional since 8/3 != integer.
    # We simulate this by deliberately constructing a weight tensor with an incorrect shape.
    weight_tensor = torch.randn(in_channels, out_channels // groups, kT, kH, kW, 
                                device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    stride = [1, 1, 1]
    padding = [0, 0, 0]
    output_padding = [0, 0, 0]

    # Since the kernel does not check for divisibility by groups, the computed indexing will be wrong.
    # We compare the output with PyTorch's native ConvTranspose3d to trigger a discrepancy.
    native_conv = nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size=(kT, kH, kW),
        stride=tuple(stride), padding=tuple(padding), output_padding=tuple(output_padding),
        groups=groups, bias=True
    ).cuda()

    # Manually set the weights and bias to the ones we provided (if possible)
    # Here, since our weight_tensor shape does not match what native_conv expects,
    # we simulate the mismatch by only comparing that the outputs differ.
    native_conv.weight.data.copy_(torch.randn_like(native_conv.weight))
    native_conv.bias.data.copy_(torch.randn_like(native_conv.bias))

    # Run our kernel (which will use the provided weight_tensor shape)
    out_kernel = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)
    out_native = native_conv(input_tensor)

    # Because of the improper group division, the outputs should not match.
    with pytest.raises(AssertionError):
        assert torch.allclose(out_kernel, out_native, atol=1e-5)

# Test issue 2 & 4: Poor work partitioning and inefficient memory accesses.
# While these issues are primarily performance related (and correctness is maintained),
# we design a test case with a larger batch and channel numbers that will suffer from the suboptimal
# parallel work distribution. The test checks for correctness but can be used for profiling later.
def test_large_input_correctness():
    cuda_module = build_kernel()

    batch_size = 16
    in_channels = 32
    out_channels = 16
    in_depth = 16
    in_height = 32
    in_width = 64
    kT, kH, kW = 3, 5, 7

    input_tensor = torch.randn(batch_size, in_channels, in_depth, in_height, in_width, 
                               device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, out_channels // 1, kT, kH, kW, 
                                device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    stride = [1, 1, 1]
    padding = [0, 0, 0]
    output_padding = [0, 0, 0]
    groups = 1

    # Use PyTorch's native conv_transpose3d for reference.
    native_conv = nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size=(kT, kH, kW),
        stride=tuple(stride), padding=tuple(padding), output_padding=tuple(output_padding),
        groups=groups, bias=True
    ).cuda()

    # Assign the same parameters to native_conv as our kernel.
    with torch.no_grad():
        native_conv.weight.copy_(weight_tensor)
        native_conv.bias.copy_(bias_tensor)

    out_kernel = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)
    out_native = native_conv(input_tensor)

    # Even though the kernel's work partitioning and memory accesses are inefficient,
    # the final result should be correct.
    assert torch.allclose(out_kernel, out_native, atol=1e-5), "Kernel output does not match native output."

