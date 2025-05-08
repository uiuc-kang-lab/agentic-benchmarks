
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure the build uses the current file kernel.cu in the working directory.
    module = load(
        name="custom_conv2d",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dtype_not_float32():
    # Issue 1: Use a tensor with dtype float64 (double) to trigger type misuse.
    cuda_module = build_kernel()
    
    # Create a simple convolution setup
    N, C_in, H, W = 1, 3, 8, 8
    C_out = 2
    K_h, K_w = 3, 3
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1
    
    # Create input and weight with float64 instead of float32.
    input = torch.randn(N, C_in, H, W, dtype=torch.float64, device="cuda")
    weight = torch.randn(C_out, C_in // groups, K_h, K_w, dtype=torch.float64, device="cuda")
    
    # Create a valid bias
    bias = torch.randn(C_out, dtype=torch.float64, device="cuda")
    
    with pytest.raises(RuntimeError):
        # The kernel expects float32 pointers. This call should fail or produce a runtime error.
        cuda_module.forward(input, weight, bias, stride, padding, dilation, groups)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_constant_kernel_dims():
    # Issue 2: Use kernel dimensions that are not compile-time constants.
    # Although this kernel works for any run-time kernel size,
    # the use of #pragma unroll may cause warnings or suboptimal unrolling with larger kernels.
    cuda_module = build_kernel()
    
    N, C_in, H, W = 1, 3, 16, 16
    C_out = 2
    # Use a larger kernel so that unrolling is not trivial.
    K_h, K_w = 7, 7
    stride = [1, 1]
    padding = [3, 3]
    dilation = [1, 1]
    groups = 1
    
    input = torch.randn(N, C_in, H, W, dtype=torch.float32, device="cuda")
    weight = torch.randn(C_out, C_in // groups, K_h, K_w, dtype=torch.float32, device="cuda")
    bias = torch.randn(C_out, dtype=torch.float32, device="cuda")
    
    # This run may compile but can trigger suboptimal performance warnings related to #pragma unroll.
    output = cuda_module.forward(input, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()
    # There is no reference here; we just ensure that a tensor is returned and has the expected shape.
    assert output.shape == (N, C_out, H, W), f"Expected shape {(N, C_out, H, W)}, got {output.shape}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bias_shape_mismatch():
    # Issue 3: Provide a bias tensor with a shape that does not match C_out.
    cuda_module = build_kernel()
    
    N, C_in, H, W = 1, 4, 10, 10
    C_out = 3
    K_h, K_w = 3, 3
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1
    
    input = torch.randn(N, C_in, H, W, dtype=torch.float32, device="cuda")
    weight = torch.randn(C_out, C_in // groups, K_h, K_w, dtype=torch.float32, device="cuda")
    # Create a bias tensor with incorrect shape (size not equal to C_out)
    bias = torch.randn(C_out + 1, dtype=torch.float32, device="cuda")
    
    with pytest.raises(RuntimeError):
        cuda_module.forward(input, weight, bias, stride, padding, dilation, groups)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_invalid_groups_division():
    # Issue 4: Pass a groups parameter that does not evenly divide
    # C_in or C_out which is not allowed in the kernel's logic.
    cuda_module = build_kernel()
    
    # Define parameters where in_channels is not divisible by groups.
    N, C_in, H, W = 1, 5, 8, 8  # 5 is not divisible by 2
    C_out = 4  # For simplicity, let C_out be divisible by groups, but groups still invalid for C_in.
    K_h, K_w = 3, 3
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 2  # 5 / 2 is not an integer
    
    input = torch.randn(N, C_in, H, W, dtype=torch.float32, device="cuda")
    # The weight tensor shape for conv2d should be (C_out, C_in/groups, K_h, K_w),
    # but with invalid groups, C_in/groups is not an integer.
    # We manually force a malformed weight shape.
    with pytest.raises(RuntimeError):
        # This should trigger an error either from our kernel extension or from the caller.
        weight = torch.randn(C_out, C_in // groups, K_h, K_w, dtype=torch.float32, device="cuda")
        bias = torch.randn(C_out, dtype=torch.float32, device="cuda")
        cuda_module.forward(input, weight, bias, stride, padding, dilation, groups)
