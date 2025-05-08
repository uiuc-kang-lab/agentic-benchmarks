
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def build_kernel():
    # Compile the CUDA extension from kernel.cu.
    cuda_module = load(
        name="test_group_norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that non-float32 inputs trigger an error.
def test_dtype_issue():
    kernel = build_kernel()
    batch_size = 4
    features = 16
    num_groups = 4
    dim1 = 8
    dim2 = 8
    eps = 1e-5

    # Create input tensors of type double.
    x = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=torch.float64)
    weight = torch.ones(features, device='cuda', dtype=torch.float64)
    bias = torch.zeros(features, device='cuda', dtype=torch.float64)

    with pytest.raises(RuntimeError):
        # Expect a runtime error because the kernel expects float32.
        _ = kernel.forward(x, weight, bias, num_groups, eps)

# Issue 2: Test that using more than 1024 channels (exceeding constant memory size) leads to incorrect results.
def test_constant_memory_overflow():
    kernel = build_kernel()
    batch_size = 2
    features = 2048  # more than 1024 channels
    num_groups = 8
    dim1 = 16
    dim2 = 16
    eps = 1e-5

    x = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=torch.float32)
    weight = torch.randn(features, device='cuda', dtype=torch.float32)
    bias = torch.randn(features, device='cuda', dtype=torch.float32)

    # Run the fused kernel.
    y_kernel = kernel.forward(x, weight, bias, num_groups, eps)

    # Compute reference output using PyTorch GroupNorm.
    # Note: PyTorch's GroupNorm requires channels divisible by num_groups.
    group_norm_ref = nn.GroupNorm(num_groups=num_groups, num_channels=features, eps=eps).to(x.device)
    # Set weight and bias to the same values.
    group_norm_ref.weight.data.copy_(weight)
    group_norm_ref.bias.data.copy_(bias)
    y_reference = group_norm_ref(x)

    # With constant memory overflow the output is likely to deviate considerably.
    # Here we assert that the outputs are not close.
    assert not torch.allclose(y_kernel, y_reference, atol=1e-3), (
        "Kernel output is unexpectedly close to the reference output despite constant memory overflow issue."
    )

# Issue 3: Test that using a number of channels not divisible by num_groups leads to wrong behavior.
def test_invalid_channel_groups():
    kernel = build_kernel()
    batch_size = 2
    features = 62  # deliberately not divisible by num_groups
    num_groups = 8
    dim1 = 16
    dim2 = 16
    eps = 1e-5

    x = torch.randn(batch_size, features, dim1, dim2, device='cuda', dtype=torch.float32)
    weight = torch.randn(features, device='cuda', dtype=torch.float32)
    bias = torch.randn(features, device='cuda', dtype=torch.float32)

    # Since the kernel does not check that channels % num_groups == 0,
    # the indexing will be incorrect. We test for incorrect output by comparing to the PyTorch GroupNorm.
    with pytest.raises(AssertionError):
        # Compute reference output using PyTorch GroupNorm (which would not even be allowed if channels mismatch).
        group_norm_ref = nn.GroupNorm(num_groups=num_groups, num_channels=features, eps=eps).to(x.device)
        group_norm_ref.weight.data.copy_(weight)
        group_norm_ref.bias.data.copy_(bias)
        y_reference = group_norm_ref(x)

        # Run the fused kernel.
        y_kernel = kernel.forward(x, weight, bias, num_groups, eps)

        # If the kernel were correct, these outputs should be similar.
        # Here we assert that they are not, to highlight the issue.
        assert torch.allclose(y_kernel, y_reference, atol=1e-3), (
            "Kernel output unexpectedly matches the reference despite invalid channel grouping."
        )
