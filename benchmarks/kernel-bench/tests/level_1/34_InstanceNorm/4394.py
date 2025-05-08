
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

# Build the kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="instance_norm_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case for Issue 1: Incorrect grid dimension calculation
# The kernel should process each (N, C) slice separately.
# We prepare input data with nontrivial per-instance statistics so that 
# an unprocessed instance produces a wrong result.
def test_incorrect_grid_dimension():
    cuda_module = build_kernel()
    # Use a small grid so that (N * C) is much larger than the number of blocks actually launched.
    # Here, the bug will result in some instances not getting normalized correctly.
    batch_size = 8
    channels = 32
    H, W = 128, 128  # reasonably large number of elements per instance

    # Create input tensor with a unique constant value per instance channel
    x = torch.empty(batch_size, channels, H, W, device="cuda", dtype=torch.float32)
    # Fill each instance with a unique value based on its (n, c) index.
    for n in range(batch_size):
        for c in range(channels):
            x[n, c] = (n * channels + c) * 0.1  # constant value per instance channel

    # Create weight and bias (affine parameters)
    weight = torch.ones(channels, device="cuda", dtype=torch.float32)
    bias = torch.zeros(channels, device="cuda", dtype=torch.float32)
    eps = 1e-5

    # Use PyTorch's InstanceNorm2d as reference
    ref_inorm = torch.nn.InstanceNorm2d(channels, eps=eps, affine=True)
    # Copy the parameters to match our weight and bias
    with torch.no_grad():
        ref_inorm.weight.copy_(weight)
        ref_inorm.bias.copy_(bias)
    ref_out = ref_inorm(x)

    # Run our custom kernel forward (which has grid launch bug)
    out = cuda_module.forward(x, weight, bias, eps)
    torch.cuda.synchronize()

    # Because of the grid misconfiguration, some channels will be processed incorrectly.
    # This test should fail (i.e. the outputs should differ from the reference).
    # We use a loose tolerance to catch the difference.
    assert not torch.allclose(out, ref_out, atol=1e-4), \
        "Test failed to trigger the grid dimension issue: output unexpectedly matches reference."

# Test case for Issue 2: Lack of error checking after kernel launch
# We intentionally pass an input with an incorrect shape to provoke a CUDA launch error.
def test_missing_cuda_error_checking():
    cuda_module = build_kernel()
    # Providing a tensor with less than 4 dimensions.
    x = torch.randn(10, 10, device="cuda", dtype=torch.float32)
    weight = torch.ones(10, device="cuda", dtype=torch.float32)
    bias = torch.zeros(10, device="cuda", dtype=torch.float32)
    eps = 1e-5

    with pytest.raises(RuntimeError):
        # The forward function has a TORCH_CHECK for 4D input.
        cuda_module.forward(x, weight, bias, eps)

# Test case for Issue 3: Kernel only supports float (float32) data type.
# We try to pass a half-precision (float16) tensor and expect a runtime error.
def test_incorrect_dtype():
    cuda_module = build_kernel()
    batch_size = 4
    channels = 16
    H, W = 64, 64

    # Create half-precision input tensor.
    x = torch.randn(batch_size, channels, H, W, device="cuda", dtype=torch.float16)
    weight = torch.ones(channels, device="cuda", dtype=torch.float16)
    bias = torch.zeros(channels, device="cuda", dtype=torch.float16)
    eps = 1e-5

    with pytest.raises(RuntimeError):
        # Since our kernel only accepts float32, this should cause an error.
        cuda_module.forward(x, weight, bias, eps)
