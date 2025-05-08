
import torch
import pytest
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

# Issue 1: The kernel only supports groups==1.
def test_groups_not_supported():
    mod = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 2  # Not supported
    input_tensor = torch.randn(batch_size, in_channels, 10, 10, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = None
    with pytest.raises(RuntimeError, match="groups == 1"):
        mod.forward(input_tensor, weight, bias, stride, padding, output_padding, groups)

# Issue 2: The kernel only supports output_padding==0.
def test_output_padding_not_supported():
    mod = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 1  # Not supported
    groups = 1
    input_tensor = torch.randn(batch_size, in_channels, 10, 10, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = None
    with pytest.raises(RuntimeError, match="output_padding == 0"):
        mod.forward(input_tensor, weight, bias, stride, padding, output_padding, groups)

# Issue 3: The kernel only supports float32 tensors.
def test_non_float32_input():
    mod = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1
    # Use double precision instead of float32.
    input_tensor = torch.randn(batch_size, in_channels, 10, 10, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float64)
    bias = None
    with pytest.raises(RuntimeError):
        mod.forward(input_tensor, weight, bias, stride, padding, output_padding, groups)

# Issue 4: Incorrect kernel index computation for stride > 1.
def test_stride_greater_than_one():
    mod = build_kernel()
    # Use sizes large enough to force usage of the custom kernel (total_outputs > 1e6)
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    stride = 2  # Stride > 1, where the index calculation may be wrong.
    padding = 1
    output_padding = 0
    groups = 1
    H_in = 64
    W_in = 64
    input_tensor = torch.randn(batch_size, in_channels, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    output_custom = mod.forward(input_tensor, weight, bias, stride, padding, output_padding, groups)
    # Use PyTorch's built-in conv_transpose2d as a reference.
    output_reference = torch.nn.functional.conv_transpose2d(
        input_tensor, weight, bias,
        stride=stride, padding=padding, output_padding=output_padding, groups=groups
    )
    # Expect significant difference due to the potential bug in kernel index computation.
    diff = (output_custom - output_reference).abs().mean().item()
    assert diff > 1e-3, f"Expected significant difference for stride>1, but got diff={diff}"

# Issue 5: Lack of dimension sanity checks may lead to out-of-bound accesses.
def test_mismatched_dimensions():
    mod = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1
    # Create an input tensor whose dimensions do not match the expected weight dimensions.
    input_tensor = torch.randn(batch_size, in_channels, 10, 10, device="cuda", dtype=torch.float32)
    # Deliberately create a weight tensor with an incorrect kernel spatial dimension.
    weight = torch.randn(in_channels, out_channels, kernel_size + 1, kernel_size, device="cuda", dtype=torch.float32)
    bias = None
    with pytest.raises(RuntimeError):
        mod.forward(input_tensor, weight, bias, stride, padding, output_padding, groups)
