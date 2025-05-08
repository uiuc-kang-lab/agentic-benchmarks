
import os
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Issue 1: Misaligned memory for vectorized loads.
# We simulate misalignment by creating a tensor with a non-zero storage offset.
def test_misaligned_memory(kernel_module):
    # Create a tensor with one extra element so that slicing forces a misalignment.
    base_tensor = torch.randn(129, device="cuda", dtype=torch.float32)
    # This slice has storage_offset = 1, which makes its data pointer misaligned (4-byte offset -> misaligned for float2)
    preds = base_tensor.narrow(0, 1, 128)
    tgts = base_tensor.narrow(0, 1, 128)
    # Expected MSE should be 0 because preds and tgts are identical.
    result = kernel_module.forward(preds, tgts)
    torch.cuda.synchronize()
    expected = torch.tensor(0.0, device="cuda", dtype=preds.dtype)
    # We expect the misaligned load to produce an incorrect result (or even crash) because of alignment issues.
    # For testing purposes, we assert that the result is not equal to expected.
    assert not torch.allclose(result, expected, atol=1e-6), (
        "Test for misaligned memory did not trigger an error in the vectorized load branch."
    )

# Issue 2: Odd number of elements forces the scalar branch.
def test_odd_number_elements(kernel_module):
    # Create tensors with an odd number of total elements (e.g., 127)
    N = 127
    preds = torch.full((N,), 2.0, device="cuda", dtype=torch.float32)
    tgts = torch.full((N,), 1.0, device="cuda", dtype=torch.float32)
    # Expected MSE = mean((2-1)^2) = 1.0
    result = kernel_module.forward(preds, tgts)
    torch.cuda.synchronize()
    expected = torch.tensor(1.0, device="cuda", dtype=preds.dtype)
    # Since we expect the scalar loop to be used, we check correctness.
    assert torch.allclose(result, expected, atol=1e-6), (
        f"Scalar load loop did not correctly compute MSE for odd number of elements. "
        f"Got {result.item()}, expected {expected.item()}."
    )

# Issue 3: Non-contiguous input tensors.
def test_non_contiguous_inputs(kernel_module):
    # Create a 2D tensor and then take a transpose so that it is non-contiguous.
    preds_base = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    tgts_base = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    preds = preds_base.t()  # Transpose makes it non-contiguous.
    tgts = tgts_base.t()
    # Compute expected MSE using PyTorch's built-in operations.
    expected = torch.mean((preds - tgts) ** 2)
    result = kernel_module.forward(preds, tgts)
    torch.cuda.synchronize()
    # This tests that the kernel is not accounting for non-contiguity.
    assert not torch.allclose(result, expected, atol=1e-6), (
        "Kernel did not fail or produce an incorrect result when given non-contiguous inputs."
    )

# Issue 4: Hardware-specific grid configuration.
def test_hardware_assumptions(kernel_module):
    # Create a large input tensor to force the grid size to be capped by the hard-coded assumptions.
    # BLOCK_SIZE is 128, num_sms * blocks_per_sm is 216.
    # We choose a tensor larger than 216 * 128 to ensure the grid_stride is small relative to total elements.
    total_elements = 30000  # Arbitrary large number to trigger capped grid size.
    preds = torch.randn(total_elements, device="cuda", dtype=torch.float32)
    tgts = torch.randn(total_elements, device="cuda", dtype=torch.float32)
    expected = torch.mean((preds - tgts) ** 2)
    result = kernel_module.forward(preds, tgts)
    torch.cuda.synchronize()
    # While the reduction may compute the correct value mathematically due to the loop logic,
    # suboptimal hardware assumptions are an issue.
    # Here, we check if the result is correct; if it is, then only performance is affected,
    # but for our test we flag the kernel due to its hard-coded grid configuration.
    assert torch.allclose(result, expected, atol=1e-5), (
        f"Kernel produced incorrect result with large inputs under fixed grid size assumptions. "
        f"Got {result.item()}, expected {expected.item()}."
    )
    # We also print a warning message (using pytest.warns could be used in real scenarios) that the kernel
    # is using hard-coded hardware parameters.
    print("Warning: Kernel uses hard-coded hardware parameters which may not be optimal for all GPUs.")

