
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Kernel supports only float32 tensors.
def test_dtype_incompatibility():
    cuda_module = build_kernel()
    # Create input tensors with double (float64) type.
    batch_size = 2
    in_channels = 3
    height = 16
    width = 16
    out_channels = 4
    kernel_height = 3
    kernel_width = 3
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.double, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_height, kernel_width, dtype=torch.double, device="cuda")
    # Optional bias tensor
    bias = torch.randn(out_channels, dtype=torch.double, device="cuda")
    # The parameters are scalar ints.
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    with pytest.raises(RuntimeError):
        # Expect an error because the kernel will treat the underlying data as float32.
        _ = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Issue 2: Kernel does not accept tuple parameters for stride, padding, or dilation.
def test_tuple_parameter_incompatibility():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 3
    height = 16
    width = 16
    out_channels = 4
    kernel_height = 3
    kernel_width = 3
    # Create float32 tensors so that dtype is not an issue.
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_height, kernel_width, dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    # Pass tuple values for padding (and/or stride), expecting a type conversion or runtime error.
    stride = (1, 1)
    padding = (1, 2)  # asymmetric padding
    dilation = 1
    groups = 1
    with pytest.raises((TypeError, RuntimeError)):
        # This should raise an error because the kernel expects an int, not a tuple.
        _ = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Issue 3: Kernel requires contiguous tensors.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 3
    height = 16
    width = 16
    out_channels = 4
    kernel_height = 3
    kernel_width = 3
    # Create a contiguous tensor then make it non-contiguous by transposing a couple of axes.
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device="cuda")
    x_noncontig = x.transpose(2, 3)  # now non-contiguous
    weight = torch.randn(out_channels, in_channels, kernel_height, kernel_width, dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    with pytest.raises(RuntimeError):
        # Expect an error because x_noncontig is not contiguous.
        _ = cuda_module.forward(x_noncontig, weight, bias, stride, padding, dilation, groups)
