
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Kernel is not type-generic (only works for float32)
def test_double_input():
    # Create a double input tensor
    x = torch.randn(1024, dtype=torch.float64, device='cuda')
    kernel_module = build_kernel()
    # The kernel is designed for float, so using double should trigger an issue.
    output = kernel_module.forward(x)
    expected = torch.sigmoid(x)
    # The results likely differ significantly (or the kernel might even crash)
    # Here we assert that the computed output is NOT close to the expected double-precision sigmoid.
    # (If the kernel were correct for doubles, they would be close.)
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Kernel unexpectedly handled double dtype correctly."

# Issue 2: Kernel assumes contiguous memory
def test_non_contiguous_input():
    # Create a contiguous tensor and then create a non-contiguous view via slicing.
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    # Create a non-contiguous tensor by slicing columns with a step.
    x_noncontig = x[:, ::2]
    kernel_module = build_kernel()
    # The kernel expects contiguous input, so the output will be incorrect.
    output = kernel_module.forward(x_noncontig)
    expected = torch.sigmoid(x_noncontig)
    # We expect the results to differ because the kernel does not support non-contiguous tensors.
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Kernel unexpectedly handled non-contiguous input correctly."

# Issue 3: Kernel assumes aligned memory (vectorized float4 loads/stores)
def test_misaligned_input():
    # Allocate a contiguous tensor then create a misaligned view by offsetting the data pointer.
    # One way is to create a larger tensor and then take a slice starting from an offset element.
    base = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Create misaligned tensor by skipping the first element so that the underlying pointer is offset.
    x_misaligned = base[1:]
    kernel_module = build_kernel()
    output = kernel_module.forward(x_misaligned)
    expected = torch.sigmoid(x_misaligned)
    # The vectorized memory accesses may fail or yield incorrect results if the pointer is misaligned.
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Kernel unexpectedly handled misaligned input correctly."
        
if __name__ == "__main__":
    pytest.main([__file__])
