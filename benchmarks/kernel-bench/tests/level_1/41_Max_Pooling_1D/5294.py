
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Utility function to compile and load the custom CUDA extension.
def build_kernel():
    return load(
        name="max_pool1d_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Test that feeding a non-float32 tensor (e.g. torch.float64) leads to incorrect behavior.
# We expect that the output of the custom kernel will not match the reference implementation.
def test_non_float32_input():
    cuda_module = build_kernel()
    # Prepare a float64 tensor.
    batch_size, channels, length = 2, 3, 20
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False

    # Create a double tensor.
    x = torch.randn(batch_size, channels, length, dtype=torch.float64, device="cuda")
    # Call the kernel extension.
    try:
        output_custom = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    except RuntimeError as err:
        # If the extension explicitly errors out (which it ideally should), then the test passes.
        pytest.skip("Kernel does not support non-float32 types as expected: " + str(err))
    # If no exception is raised, compare with reference after converting x to float.
    x_float = x.float()
    output_ref = F.max_pool1d(x_float, kernel_size, stride=stride, padding=padding, dilation=dilation)
    # Convert custom kernel result to float32 for comparison.
    output_custom_float = output_custom.to(torch.float32)
    # Check that the maximum difference is large, indicating a type-support issue.
    diff = (output_custom_float - output_ref).abs().max().item()
    assert diff > 1e-3, f"Kernel should not produce correct results for double tensors, but diff={diff}"

# Issue 2: Test that when return_indices is true, the kernel returns a concatenated tensor rather than a tuple.
def test_return_indices_concatenation():
    cuda_module = build_kernel()
    batch_size, channels, length = 2, 3, 20
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True

    x = torch.randn(batch_size, channels, length, device="cuda", dtype=torch.float32)
    result = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    # Compute expected output length for max_pool1d
    output_length = ((length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    # The custom kernel concatenates output and indices along the last dimension.
    expected_last_dim = output_length * 2
    assert result.shape == (batch_size, channels, expected_last_dim), \
        f"Expected concatenated tensor shape {(batch_size, channels, expected_last_dim)}, got {result.shape}"

# Issue 3: Test that the kernel errors out for non-contiguous inputs.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size, channels, length = 2, 3, 20
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False

    # Create a contiguous tensor and then make it non-contiguous via transposition.
    x = torch.randn(batch_size, channels, length, device="cuda", dtype=torch.float32)
    x_noncontiguous = x.transpose(1, 2)
    # Ensure that the tensor is indeed non-contiguous.
    assert not x_noncontiguous.is_contiguous(), "Test setup error: tensor is contiguous."

    with pytest.raises(RuntimeError, match="Input tensor must be contiguous."):
        _ = cuda_module.forward(x_noncontiguous, kernel_size, stride, padding, dilation, return_indices)

# Issue 4: Test for potential asynchronous kernel launch errors (simulate by providing an invalid kernel size).
def test_kernel_launch_error():
    cuda_module = build_kernel()
    batch_size, channels, length = 2, 3, 20
    # Provide a kernel_size that makes output_length non-positive.
    kernel_size = 30  # Likely larger than what input can support with valid parameters.
    stride = 1
    padding = 0
    dilation = 1
    return_indices = False

    x = torch.randn(batch_size, channels, length, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Output length must be positive."):
        _ = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
