
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="custom_depthwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Incorrect warp-level reduction due to fixed mask.
# We design a test case with a borderline output count (which leads to a partially populated warp)
# and compare the kernel’s output with PyTorch’s built-in depthwise convolution.
def test_incorrect_warp_mask():
    # Use a size that produces one or few output elements per kernel launch.
    batch_size = 1
    in_channels = 2
    kernel_size = 3
    stride = 1
    padding = 1  # so output size equals input size
    height_in = 3
    width_in = 3

    # Create input tensor and weight tensor that match depthwise convolution layout.
    # For torch.nn.Conv2d with groups=in_channels, weight shape is (out_channels, 1, kernel_size, kernel_size)
    x = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    # Let channels_per_group be 1 so that out_channels = in_channels.
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = None  # No bias

    # Run PyTorch’s own depthwise convolution
    ref = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, groups=in_channels)

    # Use the CUDA kernel (our custom extension) forward function.
    mod = build_kernel()
    # The kernel expects weight layout with dimensions: [in_channels, channels_per_group, kernel_size, kernel_size]
    # so we use the same weight.
    out = mod.forward(x, weight, bias, stride, padding)

    # If the warp mask issue is present, the numerical result may differ.
    assert torch.allclose(out, ref, atol=1e-4), \
        f"Output ({out}) differs from reference ({ref}). Likely due to warp-level reduction issues."

# Issue 2: Kernel supports only float32.
def test_non_float_input():
    batch_size = 2
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 0
    height_in = 8
    width_in = 8

    # Create input and weight tensors of type double.
    x = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.double)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.double)
    bias = None

    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel’s TORCH_CHECKs should trigger an error because the tensors are not float32.
        mod.forward(x, weight, bias, stride, padding)

# Issue 3: Non-contiguous tensors.
def test_non_contiguous_input():
    batch_size = 2
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1
    height_in = 16
    width_in = 16

    # Create a contiguous tensor then make it non-contiguous by transposing spatial dims.
    x = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    x_non_contig = x.transpose(2, 3)  # This makes tensor non-contiguous.
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = None
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the input is non-contiguous.
        mod.forward(x_non_contig, weight, bias, stride, padding)

# Issue 4: Lack of CUDA error checking.
# Here we trigger an error by providing an invalid weight tensor dimension.
def test_invalid_weight_dim():
    batch_size = 2
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 0
    height_in = 10
    width_in = 10

    x = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)

    # Create a weight tensor with incorrect dimensions (3D instead of 4D).
    weight_invalid = torch.randn(in_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    mod = build_kernel()
    with pytest.raises(RuntimeError):
        mod.forward(x, weight_invalid, bias, stride, padding)
        
# Issue 5: Assumption of warp size 32.
# There is no direct way to simulate a different warp size from Python.
# However, we at least document this limitation and check that our expected warp size constant is used.
def test_assumed_warp_size():
    # This test just checks that a known input produces a valid output.
    # It indirectly confirms that our kernel’s internal assumption (warp size = 32)
    # does not crash for a standard configuration.
    batch_size = 1
    in_channels = 1
    kernel_size = 3
    stride = 1
    padding = 1
    height_in = 5
    width_in = 5

    x = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    mod = build_kernel()
    out = mod.forward(x, weight, bias, stride, padding)
    # Basic sanity check: output shape must match
    expected_h = (height_in + 2 * padding - kernel_size) // stride + 1
    expected_w = (width_in + 2 * padding - kernel_size) // stride + 1
    assert out.shape == (batch_size, in_channels, expected_h, expected_w), \
        f"Output shape {out.shape} does not match expected shape {(batch_size, in_channels, expected_h, expected_w)}."

if __name__ == "__main__":
    pytest.main([__file__])
