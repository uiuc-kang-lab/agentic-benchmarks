
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel module from kernel.cu.
def build_kernel():
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Helper: reference leaky_relu using PyTorch
def ref_leaky_relu(x, negative_slope):
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

# Issue 1: Misaligned memory.
# Create a tensor whose underlying storage is offset such that the pointer is not 16-byte aligned.
def test_misaligned_input():
    # Allocate a larger tensor and then take a slice with a non-zero storage_offset.
    # Even though the sliced tensor is contiguous, its data pointer may be misaligned for float4.
    base = torch.randn(33, dtype=torch.float32, device="cuda")
    # Slicing off one element creates a tensor sharing the storage but its pointer is base.data_ptr() + sizeof(float)
    misaligned = base[1:].clone()  # clone to ensure contiguous allocation with offset preserved in metadata
    # Make sure the total number of elements is divisible by 4 (e.g. 32 elements)
    assert misaligned.numel() % 4 == 0, "Test precondition failed: numel must be multiple of 4"
    module = build_kernel()
    negative_slope = 0.01
    # Even if no immediate exception is raised, the results may be wrong.
    out = module.forward(misaligned, negative_slope)
    torch.cuda.synchronize()
    expected = ref_leaky_relu(misaligned, negative_slope)
    # Because misaligned accesses may produce undefined behavior, we check that the outputs are not close.
    assert not torch.allclose(out, expected, atol=1e-6), "Misaligned memory access did not trigger an observable error."

# Issue 2: Remainder loop processing device memory on the host.
# Create an input whose total number of elements is not divisible by 4.
def test_remainder_processing():
    # Create a tensor with 10 elements (not divisible by 4)
    x = torch.randn(10, dtype=torch.float32, device="cuda")
    module = build_kernel()
    negative_slope = 0.01
    # The CUDA kernel processes the first 8 elements via the kernel and then
    # attempts to process the remaining 2 elements on the host (which is invalid for GPU memory).
    with pytest.raises(RuntimeError):
        # Expect a runtime error when the host code incorrectly touches GPU memory.
        out = module.forward(x, negative_slope)
        # Force synchronization so that any asynchronous error is raised.
        torch.cuda.synchronize()

# Issue 3: Missing data type check.
# Pass a tensor with a wrong dtype (e.g. float64) to the kernel.
def test_wrong_dtype():
    x = torch.randn(16, dtype=torch.float64, device="cuda")
    module = build_kernel()
    negative_slope = 0.01
    # Although the CHECK_INPUT macro does not verify the data type,
    # the reinterpret_cast to float4 assumes float32. This should eventually cause an error.
    with pytest.raises(RuntimeError):
        out = module.forward(x, negative_slope)
        torch.cuda.synchronize()

# Issue 4: Lack of proper error checking after kernel launch.
# Although the code does not explicitly call cudaGetLastError(),
# we can trigger an error by passing an input that violates the CHECK macros.
def test_non_contiguous_input():
    # Create a contiguous tensor first
    x = torch.randn(16, dtype=torch.float32, device="cuda")
    # Make a non-contiguous tensor by transposing a 2D tensor
    x = x.view(4, 4).t()
    assert not x.is_contiguous(), "Tensor should be non-contiguous for this test."
    module = build_kernel()
    negative_slope = 0.01
    with pytest.raises(RuntimeError) as excinfo:
        out = module.forward(x, negative_slope)
    # Check that the error message indicates the tensor must be contiguous.
    assert "must be contiguous" in str(excinfo.value)
