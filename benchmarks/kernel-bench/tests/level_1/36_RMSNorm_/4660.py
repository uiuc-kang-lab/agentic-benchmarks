
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="rms_norm_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference RMS normalization implemented in PyTorch (applied on feature dim)
def torch_rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Compute RMS along the feature dimension (dim=1) keeping dimensions.
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms

# Test case 1: Use half precision input.
# This should trigger issue 1 and issue 2 because the kernel code calls sqrt on __half
# without using proper half arithmetic intrinsics and may miscompute the shared memory size.
def test_half_precision():
    device = 'cuda'
    batch_size, num_features, dim1, dim2 = 4, 32, 16, 16
    eps = 1e-5
    x = torch.randn(batch_size, num_features, dim1, dim2, device=device, dtype=torch.half)
    # Run the extension kernel.
    module = build_kernel()
    y_cuda = module.forward(x, eps)
    # Compute reference using PyTorch (do the computations in half precision).
    y_ref = torch_rms_norm(x, eps)
    # They should be close; if the issues affect half computations the error will be large.
    assert torch.allclose(y_cuda, y_ref, atol=1e-3), (
        f"Half precision test failed: max diff = {(y_cuda - y_ref).abs().max().item()}"
    )

# Test case 2: Use non-power-of-two number of features.
# While the default launch uses 256 threads, if num_features is not a power of two the reduction loop 
# (which assumes a power-of-two blockDim) may be fragile in a more general setting.
def test_non_power_of_two_features():
    device = 'cuda'
    batch_size, num_features, dim1, dim2 = 4, 7, 16, 16  # 7 is not a power-of-two.
    eps = 1e-5
    x = torch.randn(batch_size, num_features, dim1, dim2, device=device, dtype=torch.float32)
    module = build_kernel()
    y_cuda = module.forward(x, eps)
    y_ref = torch_rms_norm(x, eps)
    assert torch.allclose(y_cuda, y_ref, atol=1e-5), (
        f"Non-power-of-two feature test failed: max diff = {(y_cuda - y_ref).abs().max().item()}"
    )

# Test case 3: Use a custom shape that stresses grid-stride looping.
# Although not a direct error, this test provides many samples (by having a large value for the product of dimensions beyond features)
# so that the grid-stride loop is exercised, exposing potential issues when many samples are processed.
def test_large_sample_count():
    device = 'cuda'
    batch_size, num_features, dim1, dim2 = 8, 64, 128, 128
    eps = 1e-5
    x = torch.randn(batch_size, num_features, dim1, dim2, device=device, dtype=torch.float32)
    module = build_kernel()
    y_cuda = module.forward(x, eps)
    y_ref = torch_rms_norm(x, eps)
    assert torch.allclose(y_cuda, y_ref, atol=1e-5), (
        f"Large sample count test failed: max diff = {(y_cuda - y_ref).abs().max().item()}"
    )
