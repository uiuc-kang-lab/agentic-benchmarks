
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    cuda_module = load(
        name="custom_pooling",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: kernel_size > 32 to trigger the fixed iteration loop issue.
def test_kernel_size_over_max_unroll():
    # Create an input tensor.
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    kernel_size = 33  # bigger than 32 to trigger incomplete pooling window accumulation.
    stride = 1
    padding = 0
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    
    # Get a reference result from PyTorch's F.avg_pool2d.
    ref = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    
    # Run the custom CUDA kernel.
    custom_pool = build_kernel()
    # The kernel forward function expects: tensor, kernel_size, stride, padding.
    out = custom_pool.forward(x, kernel_size, stride, padding)
    
    # Since the kernel does not properly handle kernel_size > 32, the outputs should differ.
    # We explicitly check that they are not nearly equal.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Test failed: Kernel output matches PyTorch reference with kernel_size > 32. "
        "This indicates the fixed unroll loops may not be triggering the issue."
    )

# Test case 2: using a tensor with non-float type (double) to trigger wrong shared memory allocation.
def test_shared_memory_allocation_for_double():
    # Create an input tensor of type double.
    batch_size = 2
    channels = 3
    height = 32
    width = 32
    kernel_size = 3
    stride = 1
    padding = 1
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.double)
    
    # Get a reference result from PyTorch's F.avg_pool2d.
    ref = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    
    # Run the custom CUDA kernel.
    custom_pool = build_kernel()
    out = custom_pool.forward(x, kernel_size, stride, padding)
    
    # Because the shared memory allocation uses sizeof(float) instead of sizeof(scalar_t),
    # the kernel will produce incorrect results when operating on double.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Test failed: Kernel output matches PyTorch reference for double, "
        "which indicates the shared memory allocation might be incorrectly sized."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
