
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_3x3_kernel():
    # Issue 1: Using a non 3x3 kernel (e.g. 5x5) should trigger a mismatch.
    batch_size = 4
    in_channels = 3
    out_channels = 8
    in_height, in_width = 32, 32
    kernel_size = 5  # different than the hardcoded 3

    # Create input and weight with kernel size 5
    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda")
    bias = None

    # Compute reference output using PyTorch functional conv2d.
    ref_out = F.conv2d(x, weight, bias=bias, stride=1, padding=0)

    # Call the custom CUDA kernel (which assumes a 3x3 kernel) via the forward function
    module = build_kernel()
    out = module.forward(x, weight, None, stride=1, padding=0, dilation=1, groups=1)

    # We expect the outputs to differ because the kernel always uses a 3x3 filter.
    assert not torch.allclose(out, ref_out, atol=1e-5), (
        "Test failed: non-3x3 kernel inputs produced matching outputs even though the kernel is hardcoded for 3x3!"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dilation_parameter():
    # Issue 2: Using a dilation parameter != 1 should cause disagreement.
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    in_height, in_width = 32, 32
    dilation = 2  # non default dilation
    padding = 0
    stride = 1

    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda")
    bias = None

    ref_out = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)

    module = build_kernel()
    out = module.forward(x, weight, None, stride=stride, padding=padding, dilation=dilation, groups=1)

    # Because the kernel ignores dilation, the output will be different.
    assert not torch.allclose(out, ref_out, atol=1e-5), (
        "Test failed: dilation parameter is not being ignored by the kernel as expected!"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_groups_parameter():
    # Issue 3: Using groups != 1 should cause incorrect results.
    batch_size = 4
    in_channels = 4  # must be divisible by groups
    out_channels = 4
    groups = 2
    kernel_size = 3
    in_height, in_width = 32, 32
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda")
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device="cuda")
    bias = None

    ref_out = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    module = build_kernel()
    out = module.forward(x, weight, None, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # Because groups are not handled in the kernel, the output is expected to differ.
    assert not torch.allclose(out, ref_out, atol=1e-5), (
        "Test failed: groups parameter handling is broken: the kernel should not match the reference output when groups != 1!"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_stride_parameter():
    # Issue 4: When stride != 1, the shared memory indexing and tiling computations will be off.
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    in_height, in_width = 32, 32
    stride = 2  # non-unit stride
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda")
    bias = None

    ref_out = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)

    module = build_kernel()
    out = module.forward(x, weight, None, stride=stride, padding=padding, dilation=dilation, groups=1)

    # The kernel's custom tiling with non-unit stride is likely to yield incorrect outputs.
    assert not torch.allclose(out, ref_out, atol=1e-5), (
        "Test failed: stride handling in kernel appears to work correctly, but it is expected to be incorrect with stride != 1!"
    )
