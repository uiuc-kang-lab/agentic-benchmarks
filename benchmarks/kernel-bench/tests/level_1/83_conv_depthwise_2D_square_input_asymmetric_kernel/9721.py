
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Force rebuild if necessary.
    module = load(
        name="custom_depthwise_conv",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Test that using a non-float32 dtype (e.g., float64) causes incorrect results.
def test_input_dtype_must_be_float32():
    # Prepare input tensors in double.
    batch, channels, height, width = 2, 3, 32, 32
    x = torch.randn(batch, channels, height, width, device="cuda", dtype=torch.float64)
    
    # Create a weight tensor corresponding to (channels, 1, kernel_h, 1).
    kernel_h = 3
    weight = torch.randn(channels, 1, kernel_h, 1, device="cuda", dtype=torch.float64)
    bias = torch.randn(channels, device="cuda", dtype=torch.float64)
    
    module = build_kernel()

    # The custom kernel expects float32 pointers; calling it with float64
    # will result in erroneous computation or silent failure.
    with pytest.raises(RuntimeError):
        # We force a crash or a runtime error by calling the kernel
        _ = module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=channels)
        
# Issue 2: Test that providing a kernel with non-unit width (i.e. not 1) will lead to wrong or mismatched output.
def test_kernel_assumes_kernel_width_is_one():
    batch, channels, height, width = 2, 3, 32, 32
    x = torch.randn(batch, channels, height, width, device="cuda", dtype=torch.float32)
    
    # Create a weight tensor with kernel width > 1. The kernel code only indexes kernel height,
    # so if kernel width != 1, the behavior is not general.
    kernel_h, kernel_w = 3, 2  # non-standard: width is 2 instead of 1
    weight = torch.randn(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    
    module = build_kernel()
    # Even though the module.forward will use weight.size(2) as kernel_h and ignore kernel width,
    # we expect the output to be wrong compared to a reference depthwise convolution.
    out_custom = module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=channels)
    
    # Use PyTorch's native depthwise convolution for a reference,
    # but note that the native Conv2d expects the entire kernel. Here the width differs.
    conv = torch.nn.Conv2d(channels, channels, kernel_size=(kernel_h, kernel_w),
                           stride=1, padding=0, dilation=1, groups=channels, bias=True).cuda()
    # Manually set conv weights and bias to the same values
    conv.weight.data = weight
    conv.bias.data = bias
    
    out_ref = conv(x)
    
    # The outputs will differ because the CUDA kernel incorrectly assumes kernel width == 1.
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(out_custom, out_ref, atol=1e-5)

# Issue 3: Test that a very large kernel height (exceeding thread block capacity for shared memory load)
# results in an error (e.g. illegal memory access).
def test_large_kernel_height_triggers_shared_memory_issue():
    batch, channels, height, width = 1, 1, 64, 64
    x = torch.randn(batch, channels, height, width, device="cuda", dtype=torch.float32)
    
    # Set a very large kernel height (e.g., 300) which likely exceeds 16x16 = 256 thread count for shared loading.
    kernel_h = 300
    weight = torch.randn(channels, 1, kernel_h, 1, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    
    module = build_kernel()
    
    # Expect a runtime error due to out-of-bound shared memory access.
    with pytest.raises(RuntimeError):
        _ = module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=channels)

# Issue 4: Test that the use of __ldg (read-only loads) may not catch non-read-only pointer issues.
# We simulate this by passing non-contiguous weights. The forward function makes them contiguous,
# so to force a problem we break that assumption after the contiguous call.
def test_non_readonly_memory_access():
    batch, channels, height, width = 2, 3, 32, 32
    x = torch.randn(batch, channels, height, width, device="cuda", dtype=torch.float32)
    kernel_h = 3
    weight = torch.randn(channels, 1, kernel_h, 1, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    
    # Create non-contiguous weight by transposing a dimension after contiguous.
    weight = weight.transpose(1,2)
    # Note: The forward function in our kernel code calls .contiguous() on weight, so
    # this test may pass silently. However, if for any reason the pointer is non-readonly,
    # __ldg may not be optimal. We report a warning if outputs differ.
    module = build_kernel()
    out_custom = module.forward(x, weight, bias, stride=1, padding=0, dilation=1, groups=channels)
    
    # Use PyTorch native depthwise convolution for reference.
    # Rearrange weight to match expected shape.
    weight_ref = weight.contiguous().transpose(1,2)
    conv = torch.nn.Conv2d(channels, channels, kernel_size=(kernel_h, 1),
                           stride=1, padding=0, dilation=1, groups=channels, bias=True).cuda()
    conv.weight.data = weight_ref
    conv.bias.data = bias
    out_ref = conv(x)
    
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(out_custom, out_ref, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
