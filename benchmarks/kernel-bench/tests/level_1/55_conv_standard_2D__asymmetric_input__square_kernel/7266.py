
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_kernel_size_mismatch():
    # This test is designed to trigger issue #1: the kernel
    # is hardcoded to a 3x3 kernel. When a weight with a different
    # kernel size (e.g. 5x5) is provided, the forward function should
    # raise an error.
    batch = 1
    in_channels = 3
    out_channels = 4
    height, width = 10, 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Create a weight with kernel size 5 instead of the expected 3.
    weight = torch.randn(out_channels, in_channels, 5, 5, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    mod = build_kernel()
    with pytest.raises(RuntimeError, match="Kernel size mismatch with defined KERNEL_SIZE"):
        mod.forward(x, weight, bias, stride, padding, dilation, groups)

def test_invalid_groups():
    # This test is designed to trigger issue #2: the kernel does not validate
    # that in_channels is evenly divisible by groups. For example, with in_channels=3
    # and groups=2 the proper behavior is to flag an error, but the kernel silently uses
    # integer division (3//2 == 1) and computes an output.
    batch = 1
    in_channels = 3  # not divisible by groups
    out_channels = 4
    height, width = 10, 10
    stride = 1
    padding = 0
    dilation = 1
    groups = 2

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Prepare weight with shape: [out_channels, in_channels//groups, kernel_size, kernel_size]
    # With integer division, 3//2 == 1. Note that torch.nn.functional.conv2d expects
    # in_channels % groups == 0 and will raise an error.
    weight = torch.randn(out_channels, in_channels // groups, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    mod = build_kernel()
    # The custom kernel does not check for this mismatch and will run,
    # even though the configuration is invalid.
    out_custom = mod.forward(x, weight, bias, stride, padding, dilation, groups)

    # Let’s confirm that PyTorch’s own conv2d flags the issue.
    with pytest.raises(RuntimeError):
        torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
