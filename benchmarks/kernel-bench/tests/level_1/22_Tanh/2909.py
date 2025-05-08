
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA kernel extension
def build_kernel():
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger issue with non-float32 input (e.g. float64)
def test_double_input():
    my_module = build_kernel()
    # Create a double (float64) tensor
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # Run the kernel: this is expected to produce incorrect results
    y = my_module.forward(x)
    # Compare against torch.tanh
    y_ref = torch.tanh(x)
    # We expect the results to diverge because of invalid reinterpret_cast 
    # for double type. The test fails if the kernel accidentally works.
    assert not torch.allclose(y, y_ref, atol=1e-5), (
        "Kernel should not correctly process float64 inputs due to incorrect vectorized casting."
    )

# Test case 2: Trigger issue with non-contiguous input causing misaligned accesses.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing a 2D view.
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    x_t = x.t()  # non-contiguous view
    # Ensure tensor is non-contiguous
    assert not x_t.is_contiguous(), "Tensor should be non-contiguous."
    try:
        y = my_module.forward(x_t)
        # If the kernel manages to run, compare with the reference, but we expect memcpy issues.
        y_ref = torch.tanh(x_t)
        # The outputs should be different due to misaligned vector loads.
        assert not torch.allclose(y, y_ref, atol=1e-5), (
            "Kernel output should be incorrect when given a non-contiguous input."
        )
    except RuntimeError:
        # The kernel might throw a runtime error due to misaligned memory.
        pytest.fail("Kernel raised an error on non-contiguous input, which indicates mishandling of memory alignment.")

# Test case 3: Trigger issue with input size smaller than 4 elements.
def test_small_input():
    my_module = build_kernel()
    # Create an input tensor with fewer than 4 elements, e.g., 3 elements.
    x = torch.randn(3, device="cuda", dtype=torch.float32)
    y = my_module.forward(x)
    y_ref = torch.tanh(x)
    # Because the block calculation might yield zero blocks, the output could be uninitialized.
    # We assert that the output does not match the expected result.
    assert not torch.allclose(y, y_ref, atol=1e-5), (
        "Kernel should not process input tensors with less than 4 elements correctly due to block count computation."
    )
