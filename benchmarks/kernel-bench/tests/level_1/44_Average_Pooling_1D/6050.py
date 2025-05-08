
import pytest
import torch
from torch.utils.cpp_extension import load

# Rebuild the CUDA extension from kernel.cu
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Issue 1: Data type limitation (only supports float32)
def test_input_dtype():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 2
    input_length = 10
    kernel_size = 3
    stride = 1
    padding = 1

    # Create a tensor of type float64 (double) - not supported
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.double)
    with pytest.raises(RuntimeError):
        # The kernel expects the underlying data pointer to be float* so using double should trigger an error.
        cuda_module.forward(x, kernel_size, stride, padding)

# Issue 2: No error checking after kernel launch.
# We can simulate a kernel launch configuration error by providing an invalid parameter.
def test_invalid_kernel_parameters():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 2
    input_length = 10
    kernel_size = -3  # Invalid kernel_size (must be > 0)
    stride = 1
    padding = 1

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the kernel wrapper should raise an error
        cuda_module.forward(x, kernel_size, stride, padding)

# Issue 3: Hard-coded normalization (always divides by kernel_size)
# This test creates a situation where the pooling window overlaps the padded region.
# When count_include_pad is False the expected average should be computed with the effective number of elements.
# Since our kernel divides by kernel_size, its output will differ from the effective-average.
def test_normalization_behavior():
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 1
    input_length = 4
    kernel_size = 4
    stride = 1
    padding = 2  # This makes some pooling windows contain fewer than kernel_size valid elements

    # Create a simple input so that we can manually compute the expected sum.
    # We'll use 1's so that the sum equals the count of valid elements.
    x = torch.ones(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    # Forward compute using our kernel.
    out_cuda = cuda_module.forward(x, kernel_size, stride, padding)
    # Manually compute the expected output using effective counts (simulate count_include_pad=False)
    # For each pooling window, the valid part of the window is the overlap with [0, input_length).
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    expected = torch.empty(batch_size, in_channels, output_length, device="cuda", dtype=torch.float32)
    for b in range(batch_size):
        for c in range(in_channels):
            for idx in range(output_length):
                offset = idx * stride - padding
                valid_count = 0
                for k in range(kernel_size):
                    pos = offset + k
                    if pos >= 0 and pos < input_length:
                        valid_count += 1
                # Effective average if count_include_pad is False: sum/valid_count.
                # Since x is ones, the sum is equal to valid_count.
                # Note: Our kernel always divides by kernel_size so the output will be valid_count/kernel_size.
                expected[b, c, idx] = valid_count / kernel_size

    # The kernel uses division by kernel_size regardless
    assert not torch.allclose(out_cuda, expected, atol=1e-5), (
        "Kernel normalization unexpectedly matches effective count normalization. "
        "It appears the kernel does not support flexible normalization."
    )

# Issue 4: Assumption of contiguous input memory.
# This test deliberately creates a non-contiguous tensor and checks that the kernel output
# does not match the expected result computed with nn.AvgPool1d on the contiguous version.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    input_length = 16
    kernel_size = 3
    stride = 2
    padding = 1

    # Create a contiguous input tensor and then make a non-contiguous view (e.g. via transpose)
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    x_noncontiguous = x.transpose(1, 2)  # This is non-contiguous in its storage for the pooling dimension.

    # Use the nn.AvgPool1d (which requires contiguous input)
    avg_pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    # For a fair comparison, produce the expected output from the contiguous tensor.
    expected = avg_pool(x).clone()

    # Invoke our kernel using the non-contiguous input. Since our kernel assumes contiguous data,
    # the result will likely differ from the expected output.
    out_cuda = cuda_module.forward(x_noncontiguous, kernel_size, stride, padding)
    # Bring non-contiguous tensor to a contiguous version for nn.AvgPool1d for comparison.
    out_cuda_contig = out_cuda.contiguous()

    assert not torch.allclose(expected, out_cuda_contig, atol=1e-5), (
        "Kernel produced correct output for a non-contiguous input; it should assume contiguous memory."
    )
