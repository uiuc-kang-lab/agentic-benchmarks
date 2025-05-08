
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1: Non-contiguous tensor input
# This test creates a non-contiguous tensor and checks that the kernel output does NOT match the expected ReLU output.
# Because the kernel does not handle strides, a non-contiguous tensor will be misinterpreted.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create a contiguous tensor and then create a non-contiguous view
    x = torch.randn(64, 32, device='cuda', dtype=torch.float32)
    x_t = x.t()  # Transpose makes it non-contiguous
    # Expected result using PyTorch's built-in ReLU for the non-contiguous tensor
    expected = torch.relu(x_t)
    # Kernel call: Note that our kernel expects contiguous memory.
    # We deliberately pass a non-contiguous tensor to trigger the issue.
    output = cuda_module.forward(x_t)
    torch.cuda.synchronize()
    # The outputs should differ because the kernel does not respect the tensor strides.
    assert not torch.allclose(output, expected), "Test failed: Non-contiguous input produced correct output even though it was expected to fail."

# Issue 2: Lack of support for half precision.
# This test checks that providing a half precision tensor (torch.half) leads to an error 
# since the kernel only dispatches for float/double.
def test_half_precision_input():
    cuda_module = build_kernel()
    x = torch.randn(128, device='cuda', dtype=torch.half)
    with pytest.raises(RuntimeError):
        # The AT_DISPATCH_FLOATING_TYPES macro does not include half, so we expect a runtime error.
        output = cuda_module.forward(x)
        torch.cuda.synchronize()

# Issue 3: Inconsistent warp-level optimization when total number of elements is not divisible by 32.
# This test verifies that for input sizes not divisible by 32 the kernel falls back to per-thread writes.
# Though mathematically correct, the performance (and the inconsistent path) makes it an issue for a general kernel.
def test_partial_warp_input():
    cuda_module = build_kernel()
    # Choose a number of elements that is not divisible by 32.
    # For example, 1000 is not divisible by 32.
    numel = 1000
    x = torch.randn(numel, device='cuda', dtype=torch.float32)
    expected = torch.relu(x)
    output = cuda_module.forward(x)
    torch.cuda.synchronize()
    # Even though the output is mathematically correct, we flag this test as "failing" the warp-optimization consistency check.
    # Here, we check that the result is correct, but note internally that the kernel did not optimize the partial warp.
    assert torch.allclose(output, expected, atol=1e-5), "Kernel output does not match expected output for partial warp input."

if __name__ == '__main__':
    pytest.main([__file__])
