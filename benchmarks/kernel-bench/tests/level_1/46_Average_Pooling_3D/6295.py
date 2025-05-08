
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="avgpool3d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True
    )
    return cuda_module

# Helper: PyTorch average pooling using built-in operator (assumes count_include_pad=True)
def pytorch_avgpool3d(x, kernel_size, stride, padding):
    avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    return avg_pool(x)

# Test case for Issue 1: Kernel only handles float32 input.
def test_input_tensor_type():
    kernel_size = 3
    stride = 2
    padding = 1
    # Create a tensor with double precision.
    x = torch.randn(16, 32, 64, 64, 64, device='cuda', dtype=torch.double)
    
    # Fix the built-in pooling result (convert x to float before pooling)
    x_float = x.float()
    expected = pytorch_avgpool3d(x_float, kernel_size, stride, padding)
    
    module = build_kernel()
    # The kernel blindly calls data_ptr<float>(), so it will interpret the bits incorrectly.
    output = module.forward(x, kernel_size, stride, padding)
    
    # The outputs should not match since the kernel misinterprets the underlying data.
    assert not torch.allclose(output, expected, rtol=1e-3, atol=1e-3), (
        "Kernel unexpectedly produced correct results for non-float32 input; "
        "expected misinterpretation due to hard-coded float type."
    )

# Test case for Issue 2: Grid dimension overflow when (batch * channels * out_depth) is large.
def test_grid_dimension_overflow():
    kernel_size = 3
    stride = 2
    padding = 1
    # Choose dimensions such that: fused_size = out_d * batch * channels exceeds 65,535.
    # For example, with batch=16 and channels=32, we need out_d > 65535 / (16*32) ≈ 129.
    # Using the pooling formula for depth (out_d = (in_d + 2*padding - kernel_size) / stride + 1),
    # set in_d such that out_d becomes 130.
    # Solve: (in_d + 2 - 3)/2 + 1 = 130 => (in_d - 1)/2 + 1 = 130 => (in_d - 1)/2 = 129, so in_d = 259.
    in_d = 259
    in_h = 64
    in_w = 64
    x = torch.randn(16, 32, in_d, in_h, in_w, device='cuda', dtype=torch.float32)
    module = build_kernel()
    
    # When grid.z exceeds the maximum allowed value, CUDA should report an error.
    with pytest.raises(RuntimeError):
        _ = module.forward(x, kernel_size, stride, padding)
    # Note: Depending on the GPU and CUDA driver, the kernel launch may or may not immediately
    # trigger an error. This test is designed to catch cases where the grid configuration overshoots
    # allowed limits.

# Test case for Issue 3: Kernel assumes contiguous input.
def test_non_contiguous_input():
    kernel_size = 3
    stride = 2
    padding = 1
    # Create a contiguous tensor first.
    x_contig = torch.randn(16, 32, 64, 64, 64, device='cuda', dtype=torch.float32)
    # Make a non-contiguous tensor by transposing two dimensions (e.g., channels and depth)
    x_non_contig = x_contig.transpose(1, 2)  # Now shape is [16, 64, 32, 64, 64] and non-contiguous
    # Ensure it's non contiguous
    assert not x_non_contig.is_contiguous(), "Input is unexpectedly contiguous."
    
    # Built-in average pooling requires contiguous input, so we force a contiguous version for reference.
    expected = pytorch_avgpool3d(x_non_contig.contiguous(), kernel_size, stride, padding)
    
    module = build_kernel()
    # Passing the non-contiguous tensor directly to the kernel.
    output = module.forward(x_non_contig, kernel_size, stride, padding)
    
    # Expect the result to differ from the expected outcome due to incorrect index computation.
    assert not torch.allclose(output, expected, rtol=1e-3, atol=1e-3), (
        "Kernel unexpectedly produced correct results for non-contiguous input; "
        "it assumes that the input tensor is contiguous."
    )
