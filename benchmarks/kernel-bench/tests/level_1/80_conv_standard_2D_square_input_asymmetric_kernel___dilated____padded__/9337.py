
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import math

def build_kernel():
    # This builds the CUDA extension from kernel.cu.
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Data type support only float32.
# Test: Pass double tensors. The kernel calls data_ptr<float>(),
#       so using double tensors should trigger an error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dtype_float32():
    kernel_ext = build_kernel()

    batch_size = 2
    in_channels = 3
    out_channels = 4
    H, W = 16, 16
    kernel_h, kernel_w = 3, 3
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)
    
    # Create tensors of dtype double instead of float32.
    x = torch.randn(batch_size, in_channels, H, W, dtype=torch.double, device='cuda')
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.double, device='cuda')
    bias = torch.randn(out_channels, dtype=torch.double, device='cuda')

    # Expect a RuntimeError or a misinterpreted computation due to dtype mismatch.
    with pytest.raises(RuntimeError):
        _ = kernel_ext.forward(x, weight, bias, stride, padding, dilation)

# Issue 2: Lack of kernel launch error checking.
# Test: Passing an invalid (zero) stride forces a division by zero when computing
#       the output dimensions, which should cause an error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_stride_launch_error():
    kernel_ext = build_kernel()

    batch_size = 2
    in_channels = 3
    out_channels = 4
    H, W = 16, 16
    kernel_h, kernel_w = 3, 3
    stride = 0  # invalid stride, division by zero expected in output dimension computation
    padding = (1, 1)
    dilation = (1, 1)
    
    x = torch.randn(batch_size, in_channels, H, W, dtype=torch.float32, device='cuda')
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.float32, device='cuda')
    bias = torch.randn(out_channels, dtype=torch.float32, device='cuda')

    with pytest.raises(ZeroDivisionError):
        _ = kernel_ext.forward(x, weight, bias, stride, padding, dilation)

# Issue 3: Only a single stride parameter is supported.
# Test: When using asymmetric stride (vertical stride different than horizontal stride)
#       the CUDA kernel cannot match the expected output of torch.nn.functional.conv2d.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asymmetric_stride_limitation():
    kernel_ext = build_kernel()

    batch_size = 1
    in_channels = 1
    out_channels = 1
    H, W = 20, 20
    kernel_h, kernel_w = 3, 5  # asymmetric kernel
    # We want different strides for height and width, but our kernel only accepts one stride.
    stride_given = 2  # single stride value used inside the kernel
    vertical_stride = 2
    horizontal_stride = 3  # desired different stride for width
    padding = (1, 2)
    dilation = (2, 1)

    x = torch.randn(batch_size, in_channels, H, W, dtype=torch.float32, device='cuda')
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.float32, device='cuda')
    bias = None

    # Use the kernel extension with the given single stride value:
    output_kernel = kernel_ext.forward(x, weight, bias, stride_given, padding, dilation)

    # Compute reference output using torch.nn.functional.conv2d where we can supply a tuple for stride.
    # Here we use the intended different vertical/horizontal strides.
    output_ref = F.conv2d(
        x,
        weight,
        bias,
        stride=(vertical_stride, horizontal_stride),
        padding=padding,
        dilation=dilation
    )

    # Because the extension uses stride_given for both dimensions,
    # the results will differ from the reference output.
    # We check that the maximum absolute difference is large to flag the issue.
    diff = (output_kernel - output_ref).abs().max().item()
    # For this test we expect the difference not to be near zero.
    assert diff > 1e-3, f"Expected significant difference due to asymmetric stride limitation, but got diff={diff}"

