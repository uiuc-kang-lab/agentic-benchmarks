
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to compile the CUDA extension
def build_kernel():
    module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return module

# Issue 1: Fixed warp size assumption
# There is no simple way in Python to change the physical warp size.
# Instead, we trigger the limitation indirectly by using a non‐multiple of WARP_SIZE
# in the reduction length (e.g. using a kernel that has an odd reduction length) and then
# checking that the result is not consistent with PyTorch’s conv2d.
def test_fixed_warp_size_assumption():
    # Use parameters that lead to a reduction_length that is not a multiple of 32.
    batch_size = 2
    in_channels = 3
    out_channels = 8
    kernel_size = (3, 5)  # reduction_length = 3*5=15, which is not a multiple of 32.
    stride = 1
    padding = (1, 2)
    dilation = (1, 1)
    height, width = 16, 16

    # Create input, weight and bias (None) in float32
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Weight shape: (out_channels, in_channels, kh, kw)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)

    # Using PyTorch's conv2d as a reference.
    conv_ref = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride,
                                            padding=padding, dilation=dilation)

    module = build_kernel()
    out = module.forward(x, weight, torch.tensor([]) if False else None, stride, padding, dilation)

    # Since the kernel assumes a specific reduction work‐partition,
    # a non‐multiple reduction length may lead to numerical differences.
    # We test that the maximum absolute difference is large enough to suspect the limitation.
    diff = (out - conv_ref).abs().max().item()
    assert diff > 1e-3, f"Reduction with non-multiple warp size did not trigger expected error (diff={diff})."


# Issue 2: Grid dimension limitation
# We force the kernel launch to use more blocks in the z-dimension than allowed.
def test_grid_dimension_limit():
    # Most CUDA devices limit gridDim.z to 65535.
    # We try to exceed that by setting batch_size high enough.
    max_grid_z = 65535
    # Use a small out_channels so that groups_per_batch is minimal.
    batch_size = max_grid_z + 1000  # exceed gridDim.z limit
    in_channels = 1
    out_channels = 4  # groups_per_batch = ceil(4/4) = 1 -> grid z = batch_size
    kernel_size = (1, 1)
    stride = 1
    padding = (0, 0)
    dilation = (1, 1)
    height, width = 8, 8

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)

    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect a CUDA launch error because gridDim.z will exceed its maximum.
        _ = module.forward(x, weight, None, stride, padding, dilation)
        

# Issue 3: Only supports float32 type.
# Test by providing a double tensor input for x and weight.
def test_wrong_dtype():
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)
    height, width = 16, 16

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float64)

    module = build_kernel()
    # This will misinterpret the double data as float data.
    out = module.forward(x, weight, None, stride, padding, dilation)

    # Use PyTorch's conv2d with proper dtype conversion for reference.
    conv_ref = torch.nn.functional.conv2d(x.to(torch.float32), weight.to(torch.float32), bias=None,
                                            stride=stride, padding=padding, dilation=dilation)

    # The outputs are likely to differ significantly.
    diff = (out - conv_ref).abs().max().item()
    assert diff > 1e-3, f"Kernel accepted double input but produced nearly identical output (diff={diff})."


# Issue 4: Lack of input contiguity check enforcement.
# Test by providing a non-contiguous input tensor (the kernel expects contiguous inputs).
def test_non_contiguous_input():
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)
    height, width = 16, 16

    # Create a contiguous tensor and then make it non-contiguous by a transpose.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    x = x.transpose(1, 2)  # now non-contiguous
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)

    module = build_kernel()

    with pytest.raises(RuntimeError, match="x must be contiguous"):
        _ = module.forward(x, weight, None, stride, padding, dilation)
