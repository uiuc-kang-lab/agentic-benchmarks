
import torch
import math
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def compute_hinge_loss(predictions, targets):
    # Computes hinge loss as defined in the PyTorch code:
    return torch.mean(torch.clamp(1 - predictions * targets, min=0))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_type():
    # Issue 1: Pass double (float64) tensors even though kernel expects float (float32).
    # Compute the expected result in float32 precision.
    N = 1024
    predictions = torch.randn(N, device="cuda", dtype=torch.float64)
    # Create targets as -1 or 1 but in double precision.
    targets = torch.randint(0, 2, (N,), device="cuda", dtype=torch.float64) * 2 - 1
    module = build_kernel()
    # Invoke kernel which will recast data as float, producing a result that does not match
    output = module.forward(predictions, targets)
    expected = compute_hinge_loss(predictions.float(), targets.float())
    # Because of the type misinterpretation, the computed loss will be very different.
    # We assert that they are not almost equal.
    assert not torch.allclose(output, expected, atol=1e-5), \
        f"Kernel unexpectedly produced correct result with double tensors!"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_empty_input():
    # Issue 2: Test empty input (n == 0). This should produce a division by zero (or NaN).
    predictions = torch.tensor([], device="cuda", dtype=torch.float32)
    targets = torch.tensor([], device="cuda", dtype=torch.float32)
    module = build_kernel()
    output = module.forward(predictions, targets)
    # Division by zero should produce NaN (or at least an invalid value).
    assert torch.isnan(output), \
        "Kernel did not produce NaN on empty input as expected due to division by zero."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nonstandard_block_dimension():
    # Issue 3: The kernel reduction uses a fixed shared memory array size assuming a blockDim of 256.
    # In a more general scenario one might launch the kernel with a different block size.
    # Here we simulate a scenario with a very small tensor so that the effective work per block is tiny.
    # Note: The forward() wrapper in the extension always uses 256 threads per block,
    # so to mimic a nonstandard configuration we assume that a future refactoring may launch with a different block size.
    # For now we choose a tensor size that is not a multiple of 256.
    N = 300  # Not a multiple of 256; grid = 2 blocks with 256 threads each.
    predictions = torch.randn(N, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, 2, (N,), device="cuda", dtype=torch.float32) * 2 - 1
    module = build_kernel()
    output = module.forward(predictions, targets)
    expected = compute_hinge_loss(predictions, targets)
    # Depending on the bug, the reduction may be off.
    assert torch.allclose(output, expected, atol=1e-5), \
        f"Kernel output {output.item()} differs from expected {expected.item()} in nonstandard block dimension test."

