
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build and load our CUDA extension.
def build_kernel():
    cuda_module = load(
        name="optim_sigmoid",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue with non float32 (e.g., double) input.
# Expect the result to be incorrect (or at least not close to torch.sigmoid) due to use of float-specific operations.
def test_incompatible_dtype():
    my_kernel = build_kernel()
    # Create a double tensor. The kernel is not adapted for double.
    x = torch.randn(1024, dtype=torch.float64, device="cuda")
    output = my_kernel.forward(x)
    # Expected result computed by torch.sigmoid (using float64)
    expected = torch.sigmoid(x)
    # Since the kernel incorrectly uses float operations, the output is not expected to be close.
    assert not torch.allclose(output, expected, atol=1e-5), (
        "Kernel unexpectedly produced correct results with double input."
    )

# Test 2: Trigger potential misaligned memory access by creating a tensor with an offset slice.
# Although this is hard to guarantee without low-level control, slicing a tensor can produce non-16-byte aligned storage.
def test_non_aligned_memory():
    my_kernel = build_kernel()
    # Create a tensor that is likely properly aligned, then slice it to possibly break alignment
    x_full = torch.randn(1025, device="cuda", dtype=torch.float32)
    x = x_full[1:]  # this slice may not be 16-byte aligned
    output = my_kernel.forward(x)
    expected = torch.sigmoid(x)
    # We use a loose tolerance because numerical differences might appear.
    assert not torch.allclose(output, expected, atol=1e-5), (
        "Kernel did not trigger misalignment issue with non-aligned tensor input."
    )

# Test 3: Trigger improper behavior when the total number of elements is not a multiple of 4.
# The vectorized processing might mishandle the tail elements.
def test_non_multiple_vectorized_elements():
    my_kernel = build_kernel()
    # Create a tensor with a size not divisible by 4.
    # For example: 1023 elements (not divisible by 4).
    x = torch.randn(1023, device="cuda", dtype=torch.float32)
    output = my_kernel.forward(x)
    expected = torch.sigmoid(x)
    # Check if the outputs differ significantly.
    assert not torch.allclose(output, expected, atol=1e-5), (
        "Kernel incorrectly processed tail elements when total number of elements is not a multiple of 4."
    )
    
if __name__ == '__main__':
    pytest.main([__file__])
