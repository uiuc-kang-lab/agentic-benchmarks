
import torch
import pytest
from torch.utils.cpp_extension import load
import os
import numpy as np

# Helper function to build and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="gelu_cuda_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Test case 1: Incorrect data type (e.g. torch.double)
def test_incorrect_dtype():
    my_module = build_kernel()
    # Create a double tensor on CUDA.
    x = torch.randn(1024, dtype=torch.double, device="cuda")
    # Since the kernel expects float32, the result will be computed on wrong data.
    # We compare against PyTorch's native function and expect the outputs not to match.
    y_kernel = my_module.forward(x)
    # Use the original GELU formula from PyTorch for reference (casting input to float to simulate correct behavior)
    y_ref = 0.5 * x.float() * (1.0 + torch.tanh(np.sqrt(2.0/np.pi) * (x.float() + 0.044715 * x.float().pow(3))))
    # The difference should be significant because of the misinterpretation.
    max_diff = (y_kernel - y_ref).abs().max().item()
    assert max_diff > 1e-3, f"Kernel should fail with double input, but max difference is {max_diff}"

# Test case 2: Misaligned input due to slicing
def test_misaligned_input():
    my_module = build_kernel()
    # Create a tensor large enough so that slicing may cause misalignment.
    # We allocate extra elements and then take a slice that is likely misaligned.
    base = torch.randn(1025, dtype=torch.float32, device="cuda")
    # Slicing: skip the first element so that the resulting underlying pointer may not be 16-byte aligned.
    x = base[1:]
    assert x.is_contiguous(), "Sliced tensor is expected to remain contiguous."
    y_kernel = my_module.forward(x)
    # Compare with PyTorch GELU calculation
    # Note: We use the same GELU formula as in the CUDA kernel for comparison.
    sqrt_2_over_pi = np.sqrt(2.0/np.pi)
    coeff = 0.044715
    y_ref = 0.5 * x * (1.0 + torch.tanh(sqrt_2_over_pi * (x + coeff * x.pow(3))))
    # If misalignment occurs, the results of vectorized loads/stores may be incorrect.
    max_diff = (y_kernel - y_ref).abs().max().item()
    # We expect a significant difference if misalignment is an issue.
    assert max_diff > 1e-3, f"Kernel output should differ for misaligned input; max diff = {max_diff}"

# Test case 3: Non-contiguous input tensor
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a tensor and then deliberately make it non-contiguous (e.g., via transpose)
    x = torch.randn(128, 128, dtype=torch.float32, device="cuda")
    x_non_contig = x.t()  # Transpose makes it non-contiguous
    with pytest.raises(RuntimeError) as excinfo:
        _ = my_module.forward(x_non_contig)
    assert "contiguous" in str(excinfo.value), f"Expected error about contiguity, got: {excinfo.value}"

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
