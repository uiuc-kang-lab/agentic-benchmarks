
import pytest
import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    cuda_module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Limited support for groups != 1.
def test_groups_not_supported():
    # Create input and weight with groups != 1.
    # For simplicity, we use a random bias of shape (out_channels,)
    batch_size = 4
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    height = 32
    width = 32
    groups = 2   # groups != 1, not supported by the kernel.
    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Weight shape: [out_channels, in_channels/groups, k, k]
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    my_module = build_kernel()
    # Since groups != 1, the forward function should fall back to torch::conv2d.
    out_cuda = my_module.forward(x, weight, bias, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # They should match as the fallback uses torch::conv2d.
    assert torch.allclose(out_cuda, out_ref, atol=1e-5), "Fallback for groups != 1 did not produce the expected output."

# Issue 1 (variant): Limited support for dilation != 1.
def test_dilation_not_supported():
    batch_size = 4
    in_channels = 3
    out_channels = 3
    kernel_size = 3
    height = 32
    width = 32
    groups = 1  # valid for kernel but dilation != 1 is not supported
    stride = 1
    padding = 0
    dilation = 2  # dilation not supported by our kernel

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    my_module = build_kernel()
    # Should fall back to torch::conv2d.
    out_cuda = my_module.forward(x, weight, bias, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    assert torch.allclose(out_cuda, out_ref, atol=1e-5), "Fallback for dilation != 1 did not produce the expected output."

# Issue 2: Assumption of a square kernel.
def test_non_square_kernel():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    # Create a non-square kernel manually.
    kernel_h = 3
    kernel_w = 4  # non-square kernel
    height = 20
    width = 20
    groups = 1
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Intentionally create weight with non-square dimensions.
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    my_module = build_kernel()
    # Our kernel uses weight.size(2) as kernel_size and ignores weight.size(3).
    out_cuda = my_module.forward(x, weight, bias, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # The outputs are expected to differ since the custom kernel does not support non-square kernels.
    with pytest.raises(AssertionError):
        assert torch.allclose(out_cuda, out_ref, atol=1e-5), "The custom kernel should not produce the same output for non-square kernels."

# Issue 3: Data type limitation (only supports float32).
def test_dtype_not_supported():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    height = 16
    width = 16
    groups = 1
    stride = 1
    padding = 1
    dilation = 1

    # Create input tensors in double precision
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    my_module = build_kernel()
    # Expect the kernel to fail because it casts data_ptr<float>() on double tensors.
    with pytest.raises(RuntimeError):
        my_module.forward(x, weight, bias, stride, padding, dilation, groups)
