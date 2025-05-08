
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    return load(
        name="conv_transpose2d_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Test 1: Trigger issue with non-contiguous input tensor.
def test_non_contiguous_input():
    conv_ext = build_kernel()
    # Setup parameters with small kernel size to force use of the custom kernel.
    batch, in_channels, H, W = 2, 3, 8, 8
    out_channels = 4
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    
    # Create a contiguous input tensor then make it non contiguous via a transpose.
    x = torch.randn(batch, in_channels, H, W, device="cuda", dtype=torch.float32)
    x_noncont = x.transpose(1, 2)  # Now non contiguous.
    assert not x_noncont.is_contiguous(), "x_noncont should be non-contiguous to trigger the issue."
    
    # Weight is created contiguous.
    weight = torch.randn(in_channels, out_channels, *kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    # Call the extension using non-contiguous input.
    output_ext = conv_ext.forward(x_noncont, weight, bias, list(stride), list(padding))
    # For reference, compute using ATen's fallback on a contiguous tensor.
    output_ref = torch.nn.functional.conv_transpose2d(x_noncont.contiguous(), weight, bias, stride, padding)
    
    # Expect a difference because the custom kernel was used on non-contiguous input.
    assert not torch.allclose(output_ext, output_ref, atol=1e-4), \
        "Custom kernel should not work correctly with non-contiguous input."

# Test 2: Trigger issue due to runtime kernel size and unrolling.
def test_runtime_kernel_size_unroll():
    conv_ext = build_kernel()
    # Setup parameters with a non-constant kernel shape that still meets the custom kernel threshold.
    # Although in this example the sizes are within the threshold, the kernel's use of loop unrolling with runtime values
    # may lead to incorrect behavior in scenarios where the kernel dimensions are not compile-time constants.
    batch, in_channels, H, W = 2, 3, 10, 10
    out_channels = 4
    kernel_size = (3, 5)  # Both dimensions are <= 5 so the custom kernel is used.
    stride = (1, 1)
    padding = (1, 2)
    
    # Create contiguous tensors (avoiding the first issue) so that we can isolate problems caused by the runtime unrolling.
    x = torch.randn(batch, in_channels, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, *kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    output_ext = conv_ext.forward(x, weight, bias, list(stride), list(padding))
    output_ref = torch.nn.functional.conv_transpose2d(x, weight, bias, stride, padding)
    
    # Although the custom kernel has the same intent as the fallback, if the unrolling issue is present,
    # the output may differ from the highly optimized ATen op.
    assert not torch.allclose(output_ext, output_ref, atol=1e-4), \
        "Custom kernel output differs from the reference due to runtime unrolling issues."

if __name__ == "__main__":
    pytest.main([__file__])
