
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="max_pool1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    # Build the kernel module from kernel.cu.
    return build_kernel()

# Issue 1: Test kernel with a non-float32 input type.
def test_input_type(kernel_module):
    # Create a double input tensor on CUDA.
    batch_size = 2
    channels = 3
    seq_len = 16
    x = torch.randn(batch_size, channels, seq_len, device="cuda", dtype=torch.double)
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    return_indices = False

    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel uses float*.
        _ = kernel_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    torch.cuda.synchronize()

# Issue 2: Test for unexpected API behavior when return_indices is True.
def test_return_indices_behavior(kernel_module):
    # Prepare a float32 input tensor on CUDA.
    batch_size = 2
    channels = 3
    seq_len = 16
    x = torch.randn(batch_size, channels, seq_len, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = True

    out = kernel_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    
    # Expected separate outputs for the pooling and the indices.
    # The concatenation done in the kernel leads to an output tensor with last dimension doubled.
    expected_output_length = ((seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    
    # Check if the last dimension is not equal to expected_output_length (should be equal if tuple were returned)
    assert out.size(-1) != expected_output_length, (
        "Expected concatenated output when return_indices is True. "
        f"Got last dimension size {out.size(-1)} equal to expected pooling output length {expected_output_length}."
    )
    torch.cuda.synchronize()

# Issue 3: Test absence of CUDA launch error checking.
def test_no_cuda_error_check(kernel_module):
    # Intentionally supply a kernel_size that results in an incorrect output length calculation.
    # For example, a too-large dilation might cause miscomputation.
    batch_size = 1
    channels = 1
    seq_len = 10
    x = torch.randn(batch_size, channels, seq_len, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 100  # Unrealistic dilation causing output_length to be computed as negative or zero.
    return_indices = False

    with pytest.raises(RuntimeError):
        # The kernel launch should fail or produce an error upon synchronization,
        # since the output_length computed inside the extension will be non-positive.
        _ = kernel_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    torch.cuda.synchronize()

# Issue 4: Test behavior with a runtime-determined kernel_size (loop unrolling concern).
def test_kernel_size_runtime(kernel_module):
    # Create input with a kernel_size that is not a compile-time constant.
    batch_size = 2
    channels = 3
    seq_len = 32
    x = torch.randn(batch_size, channels, seq_len, device="cuda", dtype=torch.float32)
    
    # Use a non-trivial kernel_size to examine the effects of #pragma unroll
    kernel_size = 7
    stride = 2
    padding = 3
    dilation = 1
    return_indices = False

    # Although this may run without error, the unroll pragma may not apply correctly.
    # We trigger the kernel and perform a basic shape validation.
    out = kernel_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    expected_output_length = ((seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    assert out.shape == (batch_size, channels, expected_output_length), (
        f"Unexpected output shape: got {out.shape}, expected ({batch_size}, {channels}, {expected_output_length})."
    )
    torch.cuda.synchronize()

# Issue 5: Test potential issues with using min() for grid configuration.
def test_grid_calculation(kernel_module):
    # Provide a scenario that demands a large number of blocks.
    batch_size = 128
    channels = 64
    seq_len = 512
    x = torch.randn(batch_size, channels, seq_len, device="cuda", dtype=torch.float32)
    
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False

    # The idea is to see if the kernel works when the calculated number of blocks is capped.
    # If using an ambiguous min() causes an error, this test should raise.
    out = kernel_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    expected_output_length = ((seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    assert out.shape == (batch_size, channels, expected_output_length), (
        f"Unexpected output shape: got {out.shape}, expected ({batch_size}, {channels}, {expected_output_length})."
    )
    torch.cuda.synchronize()
