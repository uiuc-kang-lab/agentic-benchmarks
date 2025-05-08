
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

# Utility function to build and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="rms_norm_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A reference implementation of RMS normalization in PyTorch for comparison.
def reference_rms_norm(x, eps):
    # Calculate the RMS along the feature dimension (dimension 1)
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms

# Test 1: Trigger type conversion issue with double precision
def test_double_precision():
    # Create a double precision tensor.
    batch_size = 8
    features = 32
    dim1, dim2 = 16, 16
    x = torch.randn(batch_size, features, dim1, dim2, dtype=torch.double, device='cuda')
    eps = 1e-5

    # Get reference output from PyTorch implementation.
    ref = reference_rms_norm(x, eps)

    # Build and run the CUDA kernel.
    module = build_kernel()
    # The kernel expects a tensor of type: double
    out = module.forward(x, eps)
    torch.cuda.synchronize()

    # The use of float literals may lead to precision differences.
    # We expect the output to differ from the reference normalization within a tight tolerance.
    assert not torch.allclose(
        out, ref, atol=1e-8
    ), "Double precision test did not trigger the type conversion issue. The kernel output is unexpectedly close to the reference."

# Test 2: Trigger half precision accumulation issues.
def test_half_precision():
    # Create a half precision tensor.
    batch_size = 8
    features = 32
    dim1, dim2 = 16, 16
    x = torch.randn(batch_size, features, dim1, dim2, dtype=torch.half, device='cuda')
    eps = 1e-5

    ref = reference_rms_norm(x, eps)
    module = build_kernel()
    out = module.forward(x, eps)
    torch.cuda.synchronize()

    # Due to low precision accumulation in fp16, expect noticeable errors.
    # We use a rather loose tolerance here and then assert that the error is larger than a small threshold.
    max_err = (out - ref).abs().max().item()
    assert max_err > 1e-2, f"Half-precision accumulation issue not triggered: max error {max_err} is too low."

# Test 3: Trigger non-contiguous tensor issue.
def test_noncontiguous_input():
    # Create a contiguous tensor then make it non-contiguous by transposing dimensions.
    batch_size = 8
    features = 32
    dim1, dim2 = 16, 16
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=torch.float32)
    # Transpose so that the memory layout is non-contiguous, while keeping same shape.
    # For example, swap dim1 and features.
    x_noncontig = x.transpose(1, 2)
    # Bring it back to original shape by transposing again.
    # However, note that view and stride could be affected.
    # We want that the kernel sees a non-contiguous layout.
    x_noncontig = x_noncontig.transpose(1, 2)
    assert not x_noncontig.is_contiguous(), "Test setup error: Tensor is contiguous."
    
    eps = 1e-5
    ref = reference_rms_norm(x_noncontig, eps)
    module = build_kernel()
    out = module.forward(x_noncontig, eps)
    torch.cuda.synchronize()

    # Since the kernel assumes contiguous memory, its results will likely differ from the reference.
    # We assert that the normalized outputs are not close.
    assert not torch.allclose(
        out, ref, atol=1e-5
    ), "Non-contiguous tensor test did not trigger an error: kernel output is unexpectedly close to reference."

if __name__ == '__main__':
    pytest.main([__file__])
