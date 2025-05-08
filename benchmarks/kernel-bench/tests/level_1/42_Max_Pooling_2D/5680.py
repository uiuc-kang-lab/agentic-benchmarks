
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build and load our CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="maxpool_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return cuda_module

# Test 1: Invalid input dimensions (e.g. 3D tensor instead of expected 4D)
def test_invalid_input_dimensions():
    mod = build_kernel()
    # Create a 3D tensor (missing one dimension)
    x = torch.randn(16, 128, 128, device="cuda", dtype=torch.float32)
    # Expect an error due to incorrect input dimensions.
    with pytest.raises(Exception):
        # The kernel expects a 4D tensor; this should lead to incorrect index calculations.
        mod.forward(x, 2, 2, 1, 1)

# Test 2: Non-contiguous input tensor
def test_non_contiguous_input():
    mod = build_kernel()
    # Create a contiguous 4D tensor then make it non-contiguous via a permutation.
    x = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float32)
    x_noncontiguous = x.transpose(2, 3)  # still 4D but non-contiguous
    # Compute outputs via our kernel and via PyTorch's native max_pool2d.
    # Note: PyTorch's max_pool2d requires contiguous input. This test is designed to surface issues.
    with pytest.raises(Exception):
        # Likely the kernel will produce incorrect results if it silently reads bad memory;
        # here we assume that a failure (or miscomputed output) is acceptable to signal an issue.
        out_kernel = mod.forward(x_noncontiguous, 2, 2, 1, 1)
        # Optionally, if the kernel does not crash, we can check for discrepancies.
        out_ref = torch.nn.functional.max_pool2d(x_noncontiguous.contiguous(), kernel_size=2, stride=2, padding=1)
        # An assertion that they must be identical (this test expects a failure or discrepancy).
        assert not torch.allclose(out_kernel, out_ref), "Non-contiguous input did not trigger an error as expected."

# Test 3: Unsupported data type (e.g. half precision)
def test_unsupported_dtype():
    mod = build_kernel()
    # Create an input tensor with half precision. std::numeric_limits<scalar_t>::infinity() is not defined for half.
    x = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.half)
    with pytest.raises(Exception):
        # The dispatch macro AT_DISPATCH_FLOATING_TYPES in the kernel does not cover half precision,
        # so this should result in a runtime error.
        mod.forward(x, 2, 2, 1, 1)

# Test 4: Kernel launch error checking (simulate misuse: kernel_size <= 0)
def test_invalid_kernel_size():
    mod = build_kernel()
    # Create a valid contiguous 4D float32 tensor.
    x = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float32)
    # Pass an invalid kernel_size (0) which should yield an invalid output shape computation.
    with pytest.raises(Exception):
        # The computed output dimensions become nonsensical, potentially causing out-of-bound accesses.
        mod.forward(x, 0, 2, 1, 1)
