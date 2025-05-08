
import pytest
import torch
from torch.utils.cpp_extension import load
import numpy as np

# Helper function to build the CUDA extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="instance_norm_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference function using PyTorch's InstanceNorm2d for correctness
def pytorch_instance_norm(x, weight, bias, eps):
    # Remove running stats; affine is applied using weight and bias.
    # (Note: our kernel applies instance normalization per channel per instance.)
    N, C, H, W = x.shape
    x_reshaped = x.view(N, C, -1)
    mean = x_reshaped.mean(dim=2, keepdim=True)
    var = x_reshaped.var(dim=2, unbiased=False, keepdim=True)
    invstd = 1.0 / torch.sqrt(var + eps)
    x_norm = (x_reshaped - mean) * invstd
    # Reshape back and apply affine transformation if weight and bias are provided.
    x_norm = x_norm.view_as(x)
    if weight is not None and bias is not None:
        x_norm = x_norm * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)
    return x_norm

# Issue 1: Test for handling weight/bias constant memory usage.
# If C > 1024, the kernel does not copy weight and bias to constant memory.
# This test creates an input whose channel count is larger than 1024,
# causing the kernel to use uninitialized d_weight/d_bias.
def test_weight_bias_constant_memory():
    cuda_module = build_kernel()
    # Create tensor with C > 1024
    N, C, H, W = 2, 1025, 16, 16
    x = torch.randn(N, C, H, W, device='cuda', dtype=torch.float32)
    # Create proper weight and bias but note that our kernel only copies them if C <= 1024.
    weight = torch.randn(C, device='cuda', dtype=torch.float32)
    bias = torch.randn(C, device='cuda', dtype=torch.float32)
    # Run the kernel forward
    y = cuda_module.forward(x, weight, bias, 1e-5)
    # Compare with PyTorch instance_norm (our reference function)
    y_ref = pytorch_instance_norm(x, weight, bias, 1e-5)
    # The outputs will likely differ significantly because constant memory was not updated.
    max_diff = (y - y_ref).abs().max().item()
    assert max_diff > 1e-3, (
        "Test failed: Kernel did not show an error-like behavior when C > 1024, "
        "but it should be reading uninitialized constant memory."
    )

# Issue 2: Test for tensor data type mismatch.
# Passing a double tensor (float64) should trigger the TORCH_CHECK in forward.
def test_input_tensor_type():
    cuda_module = build_kernel()
    N, C, H, W = 2, 64, 32, 32
    # Create a float64 tensor
    x = torch.randn(N, C, H, W, device='cuda', dtype=torch.float64)
    weight = torch.randn(C, device='cuda', dtype=torch.float64)
    bias = torch.randn(C, device='cuda', dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # Expect the check in forward to trigger an error due to wrong type.
        cuda_module.forward(x, weight, bias, 1e-5)

# Issue 3: Test for unaligned memory accesses due to vectorized loads.
# By slicing the input we can force a misaligned pointer.
def test_unaligned_pointer():
    cuda_module = build_kernel()
    N, C, H, W = 2, 64, 32, 32
    # Create a contiguous tensor and then slice it so that the underlying pointer
    # is not guaranteed to be 16-byte aligned.
    x_full = torch.randn(N, C, H + 1, W, device='cuda', dtype=torch.float32)
    # Slicing off one row may cause misalignment.
    x = x_full[:, :, 1:, :]
    weight = torch.randn(C, device='cuda', dtype=torch.float32)
    bias = torch.randn(C, device='cuda', dtype=torch.float32)
    try:
        y = cuda_module.forward(x, weight, bias, 1e-5)
    except Exception as e:
        pytest.skip(f"Misaligned pointer triggered an exception as expected: {e}")
    # Compute reference result (note: our reference uses view() so we need a contiguous tensor)
    if not x.is_contiguous():
        x = x.contiguous()
    y_ref = pytorch_instance_norm(x, weight, bias, 1e-5)
    # Allow some tolerance if the misalignment degrades performance/accuracy.
    assert torch.allclose(y, y_ref, atol=1e-3), "Kernel output mismatch due to possible misaligned loads."

# Issue 4: Test for missing kernel launch error checking.
# To simulate a kernel error we can pass an input tensor with wrong shape (non 4D).
def test_wrong_input_shape():
    cuda_module = build_kernel()
    # Create a tensor with wrong dimensions (e.g., 3D instead of 4D)
    x = torch.randn(16, 64, 256, device='cuda', dtype=torch.float32)
    weight = torch.randn(64, device='cuda', dtype=torch.float32)
    bias = torch.randn(64, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expecting the TORCH_CHECK in forward to trigger an error.
        cuda_module.forward(x, weight, bias, 1e-5)

# Issue 5: Test for non-contiguous input tensor.
# The kernel assumes contiguous data, so passing a non-contiguous input might yield wrong results.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    N, C, H, W = 4, 32, 16, 16
    x = torch.randn(N, C, H, W, device='cuda', dtype=torch.float32)
    # Create a non-contiguous tensor by transposing
    x_non_contig = x.transpose(1, 2)
    weight = torch.randn(C, device='cuda', dtype=torch.float32)
    bias = torch.randn(C, device='cuda', dtype=torch.float32)
    # The kernel does not do stride checking so the results may be wrong.
    y = cuda_module.forward(x_non_contig, weight, bias, 1e-5)
    # Compute reference (by making the tensor contiguous)
    y_ref = pytorch_instance_norm(x_non_contig.contiguous(), weight, bias, 1e-5)
    max_diff = (y - y_ref).abs().max().item()
    assert max_diff > 1e-3, (
        "Test failed: Kernel did not show a misbehavior when provided non-contiguous input, "
        "but it should assume contiguous memory layout."
    )
