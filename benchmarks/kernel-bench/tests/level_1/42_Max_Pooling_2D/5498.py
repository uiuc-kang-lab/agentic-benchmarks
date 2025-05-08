
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="pool_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    return cuda_module

# A reference implementation using PyTorch's own MaxPool2d layer.
def reference_maxpool(input, kernel_size, stride, padding, dilation):
    pool = torch.nn.MaxPool2d(kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
    return pool(input)

# Test case 1:
# Using non-compile-time constant kernel_size may trigger issues with #pragma unroll.
def test_non_compile_time_kernel_size():
    # Use a kernel size that is not the common small compile-time constant
    batch_size, channels, height, width = 2, 3, 16, 16
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 2
    input = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float32)
    
    # Build the extension and run the kernel.
    mod = build_kernel()
    output_kernel = mod.forward(input, kernel_size, stride, padding, dilation)
    output_ref = reference_maxpool(input, kernel_size, stride, padding, dilation)
    
    # We expect the outputs to be close. If the unroll pragma causes issues with non-constant iteration count,
    # the results may diverge.
    assert torch.allclose(output_kernel, output_ref, atol=1e-5), \
        f"Output from kernel (non compile-time kernel_size) deviates from reference!"

# Test case 2:
# Passing an input tensor of non-floating type (e.g. int32) should trigger issues due to use of infinity and max.
def test_non_floating_input_type():
    batch_size, channels, height, width = 2, 3, 8, 8
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    # Creating an int tensor.
    input = torch.randint(low=-100, high=100, size=(batch_size, channels, height, width), device='cuda', dtype=torch.int32)
    
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # We expect the kernel to fail ("AT_DISPATCH_FLOATING_TYPES" will not dispatch int types),
        # so a RuntimeError should be thrown.
        mod.forward(input, kernel_size, stride, padding, dilation)

# Test case 3:
# Passing a non-contiguous input tensor should reveal that the kernel's assumption on memory layout is too strict.
def test_non_contiguous_input():
    batch_size, channels, height, width = 2, 3, 16, 16
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1
    # Create a contiguous tensor and then make it non-contiguous via transpose
    input = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float32)
    non_contig_input = input.transpose(1, 2)  # Now not in plain NCHW order

    mod = build_kernel()
    output_kernel = mod.forward(non_contig_input, kernel_size, stride, padding, dilation)
    
    # Compute reference after making the input contiguous with expected dimensions; we simulate reordering.
    # First, convert non_contiguous input back to NCHW using contiguous().
    input_reordered = non_contig_input.contiguous()
    output_ref = reference_maxpool(input_reordered, kernel_size, stride, padding, dilation)
    
    # The outputs will likely differ if the kernel incorrectly assumes contiguous NCHW layout.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), \
        "Kernel incorrectly handled non-contiguous input as if it were contiguous."
