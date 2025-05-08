
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

def test_non_default_stream():
    # Issue 1: Kernel uses cudaStreamDefault instead of the current stream.
    # Create a custom CUDA stream and run the kernel within its context.
    stream = torch.cuda.Stream()
    mod = build_kernel()
    # Set some reasonable parameters for a transposed conv with bias.
    batch, in_channels, out_channels = 2, 4, 8
    kernel_size, stride, padding, output_padding, groups = 3, 2, 1, 1, 1
    x = torch.randn(batch, in_channels, 10, 10, device="cuda", dtype=torch.float32).contiguous()
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32).contiguous()
    # Create a PyTorch ConvTranspose2d module for expected output.
    conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, bias=True
    ).cuda()
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(bias)
    expected = conv(x)
    
    # Launch our kernel within a non-default stream.
    with torch.cuda.stream(stream):
        output = mod.forward(x, weight, bias, stride, padding, output_padding, groups)
    # Force synchronization of our custom stream.
    stream.synchronize()
    # Because the CUDA kernel always launches on cudaStreamDefault, using a non-default stream
    # may lead to unexpected ordering. In this contrived test we expect the mismatch.
    assert not torch.allclose(output, expected, atol=1e-4), (
        "Kernel appears to respect the non-default stream context, but it should always use cudaStreamDefault."
    )

def test_runtime_kernel_size_unroll():
    # Issue 2: The use of #pragma unroll with a runtime kernel_size can cause suboptimal or incorrect behavior.
    # Use a kernel_size that is not a compile time constant (e.g. 5) and compare the result
    # with PyTorch’s own conv_transpose2d implementation.
    mod = build_kernel()
    batch, in_channels, out_channels = 1, 3, 3
    kernel_size, stride, padding, output_padding, groups = 5, 1, 2, 0, 1
    x = torch.randn(batch, in_channels, 20, 20, device="cuda", dtype=torch.float32).contiguous()
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32).contiguous()
    # No bias in this test.
    expected = torch.nn.functional.conv_transpose2d(
        x, weight, bias=None, stride=stride, padding=padding,
        output_padding=output_padding, groups=groups
    )
    output = mod.forward(x, weight, None, stride, padding, output_padding, groups)
    torch.cuda.synchronize()
    # If the unroll pragma causes issues for runtime kernel sizes, the result may differ.
    assert torch.allclose(output, expected, atol=1e-3), (
        "Kernel output mismatches expected output when using a runtime kernel_size."
    )

def test_input_double_not_supported():
    # Issue 3: The kernel only supports float32. When a double tensor is passed, it should raise an error.
    mod = build_kernel()
    batch, in_channels, out_channels = 1, 3, 3
    kernel_size, stride, padding, output_padding, groups = 3, 1, 0, 0, 1
    x = torch.randn(batch, in_channels, 10, 10, device="cuda", dtype=torch.float64).contiguous()
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float64).contiguous()
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64).contiguous()
    with pytest.raises(RuntimeError):
        _ = mod.forward(x, weight, bias, stride, padding, output_padding, groups)
