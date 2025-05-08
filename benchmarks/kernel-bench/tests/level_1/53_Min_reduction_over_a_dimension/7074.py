
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu.
def build_kernel():
    # Assumes kernel.cu is in the same directory as this test file.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(this_dir, "kernel.cu")
    cuda_module = load(
        name="test_kernel_module",
        sources=[source_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Test case to trigger issue 1:
# Create a half precision (float16) tensor and attempt to run the kernel.
# Expect the kernel dispatch to fail (or raise an error) because half precision is not supported.
def test_half_precision_unsupported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    my_module = build_kernel()
    # create a tensor with dtype float16 (half precision)
    x = torch.randn(8, 10, 12, device="cuda", dtype=torch.float16)
    # We use reduction along dimension 1, for instance.
    with pytest.raises(RuntimeError):
        # Calling the kernel should raise an error since half type is not supported.
        _ = my_module.forward(x, 1)

# Test case to trigger issue 2:
# The use of unqualified min may lead to a wrong (or inconsistent) kernel launch configuration.
# We test by passing in a tensor with a reduction dimension not divisible by 32
# and compare the kernel output with PyTorch's native torch.min.
def test_unqualified_min_effect():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    my_module = build_kernel()
    # Create a tensor with a reduction dimension (dim 1) whose size is not divisible by 32.
    # For example, size 10 is not divisible by 32.
    x = torch.randn(16, 10, 8, device="cuda", dtype=torch.float32)
    # Run our custom CUDA kernel
    out_kernel = my_module.forward(x, 1)
    # Compute reference using PyTorch native reduction
    out_ref = torch.min(x, dim=1)[0]
    # If the unqualified min in the launch code mis-computes block count, the results may be off.
    # Allow a small tolerance for floating point comparisons.
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), \
        f"Kernel output differs from reference output!. Max difference: {(out_kernel - out_ref).abs().max().item()}"
