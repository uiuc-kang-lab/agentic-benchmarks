
import pytest
import torch
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test to trigger the unqualified min issue.
#    (This test attempts to compile a kernel with a huge workload.
#     If the build environment requires qualified min, compilation will fail.)
def test_min_qualification_issue():
    # We create inputs with very large dimensions to force a high total element count.
    batch = 1
    in_channels = 4
    in_h = 256
    in_w = 256
    # Set kernel, stride, padding, dilation such that output is huge.
    kernel_h = 3
    kernel_w = 3
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    # Create input and weight tensors.
    x = torch.randn(batch, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    # For ConvTranspose2d in PyTorch, weight shape: (in_channels, out_channels_per_group, kH, kW)
    out_channels_per_group = 8
    weight = torch.randn(in_channels, out_channels_per_group, kernel_h, kernel_w,
                         device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels_per_group * groups, device="cuda", dtype=torch.float32)
    # Large output size:
    # out_h = (in_h - 1) * stride_h - 2*pad_h + dilation_h*(kernel_h-1) + 1
    # Here it is still in the 256 range but total_elements becomes large as we tile batches.
    # We simulate a case where (total_elements + threads - 1) / threads > 65535
    # by virtually increasing batch size.
    x = torch.randn(700, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    
    kernel_module = build_kernel()
    # This call should compile/run; if "min" is not defined, the compilation will have failed.
    output = kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)
    # Basic sanity check (even though the kernel result is not verified in detail)
    assert output.numel() > 0

# 2. Test to check that non-float32 input types are not handled correctly.
def test_input_tensor_type():
    batch = 2
    in_channels = 4
    in_h = 32
    in_w = 32
    kernel_h = 3
    kernel_w = 3
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1
    
    # Create double type input which the kernel does not support.
    x = torch.randn(batch, in_channels, in_h, in_w, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, 8, kernel_h, kernel_w, device="cuda", dtype=torch.float64)
    bias = torch.randn(8 * groups, device="cuda", dtype=torch.float64)
    
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # It is expected that the kernel will crash or throw an error because it assumes float.
        kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)

# 3. Test to trigger the issue when in_channels is not divisible by groups.
def test_in_channels_not_divisible_by_groups():
    batch = 2
    in_channels = 32  # 32 is not divisible by 3.
    in_h = 64
    in_w = 64
    kernel_h = 3
    kernel_w = 3
    stride = [2, 2]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 3   # 32/3 is not an integer.
    
    x = torch.randn(batch, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    
    # For weight, the shape is (in_channels, out_channels_per_group, kernel_h, kernel_w).
    # If in_channels=32 and groups=3 then the intended in_channels_per_group would be 32//3 = 10,
    # but then 2 channels (32 - 3*10) are effectively ignored.
    out_channels_per_group = 8
    weight = torch.randn(in_channels, out_channels_per_group, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels_per_group * groups, device="cuda", dtype=torch.float32)
    
    kernel_module = build_kernel()
    output = kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)
    
    # Compare with PyTorch's native ConvTranspose2d which validates this condition.
    # We expect a difference because PyTorch throws an error whereas our kernel silently computes a result.
    conv_t = torch.nn.ConvTranspose2d(in_channels, out_channels_per_group*groups,
                                      kernel_size=(kernel_h, kernel_w),
                                      stride=tuple(stride),
                                      padding=tuple(padding),
                                      dilation=tuple(dilation),
                                      groups=groups,
                                      bias=True)
    # Copy the weight & bias for a fair comparison.
    with torch.no_grad():
        conv_t.weight.copy_(weight)
        conv_t.bias.copy_(bias)
    ref_output = conv_t(x)
    
    # The outputs will differ because our kernel does not process the last 2 channels.
    assert not torch.allclose(output, ref_output, atol=1e-4), \
        "Kernel output unexpectedly matches reference even though in_channels is not divisible by groups."

# 4. Test to trigger issues with edge-case convolution parameters.
def test_edge_case_parameters():
    batch = 2
    in_channels = 8
    in_h = 10
    in_w = 10
    kernel_h = 4
    kernel_w = 5
    stride = [3, 4]
    padding = [2, 3]
    dilation = [2, 2]
    groups = 1
    
    x = torch.randn(batch, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 6, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(6, device="cuda", dtype=torch.float32)
    
    kernel_module = build_kernel()
    output = kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)
    
    # Using PyTorch's standard ConvTranspose2d as reference.
    conv_t = torch.nn.ConvTranspose2d(in_channels, 6,
                                      kernel_size=(kernel_h, kernel_w),
                                      stride=tuple(stride),
                                      padding=tuple(padding),
                                      dilation=tuple(dilation),
                                      groups=groups,
                                      bias=True)
    with torch.no_grad():
        conv_t.weight.copy_(weight)
        conv_t.bias.copy_(bias)
    ref_output = conv_t(x)
    
    # Owing to potential edge-case index elimination or mis‐alignment in our kernel,
    # the outputs may not match.
    assert not torch.allclose(output, ref_output, atol=1e-4), \
        "Kernel output unexpectedly matches reference on edge-case parameters; there may be missing index checks."

if __name__ == "__main__":
    pytest.main([__file__])
