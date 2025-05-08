
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="tuned_grid_maxpool2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Grid dimension limitation
def test_grid_dimension_limit():
    # Creating a tensor with a huge grid size: batch * channels exceeds 65535
    # For example: batch_size=700, channels=100 yields 70000 which exceeds the typical limit.
    batch_size = 700
    channels = 100
    height = 32
    width = 32
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    # Using a moderately sized spatial tensor so that the grid in z becomes problematic.
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting an error due to grid dimension overflow (or an incorrect kernel launch).
        out = kernel.forward(x, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 2: Device max function incorrect usage
def test_max_function():
    # A simple test that can detect if max() comparison is malfunctioning.
    # We create an input tensor with known values such that the location of the max is known.
    batch_size = 1
    channels = 1
    height = 4
    width = 4
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    # Create an input where the max in each pooling region is unique.
    # For the first pooling window we set 5 as the maximum, elsewhere lower values.
    x = torch.tensor([[[[1, 2, 1, 0],
                        [3, 5, 0, 1],
                        [1, 2, 1, 0],
                        [0, 1, 0, 1]]]], device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    out = kernel.forward(x, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # Expected result manually computed for non-overlapping pooling regions
    # Regions: [[1,2,3,5]] and so on.
    expected = torch.tensor([[[[5, 1],
                                [2, 1]]]], device="cuda", dtype=torch.float32)
    # If the device max function is not working, the results won't match.
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"

# Issue 3: Lack of support for half precision inputs
def test_half_precision():
    # Creating a tensor in half precision.
    batch_size = 1
    channels = 1
    height = 8
    width = 8
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float16)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel dispatch or compilation to fail because half is not supported.
        out = kernel.forward(x, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 4: Unrolled loops with fixed unroll factor may misbehave when kernel_size > 4
def test_unroll_loop():
    # Create a case where kernel_size is greater than 4.
    batch_size = 1
    channels = 1
    height = 10
    width = 10
    kernel_size = 5  # larger than the unroll factor of 4
    stride = 1
    padding = 0
    dilation = 1

    # Create an input tensor with incrementing values.
    x = torch.arange(batch_size * channels * height * width, device="cuda", dtype=torch.float32)
    x = x.reshape(batch_size, channels, height, width)
    kernel = build_kernel()
    out = kernel.forward(x, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # Compute expected output using PyTorch's built-in max pooling for verification.
    torch_maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    expected = torch_maxpool(x)
    # The results may differ if unrolling caused issues.
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"

# Issue 5: Non-contiguous input tensor
def test_non_contiguous_input():
    # Create a contiguous tensor and then make it non-contiguous.
    batch_size = 1
    channels = 1
    height = 8
    width = 8
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    # Make tensor non-contiguous by transposing spatial dimensions
    x_noncontig = x.transpose(2, 3)
    assert not x_noncontig.is_contiguous(), "Tensor should be non-contiguous for the test."

    kernel = build_kernel()
    out = kernel.forward(x_noncontig, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # For reference, compare against PyTorch's nn.MaxPool2d applied to the non-contiguous tensor.
    torch_maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    expected = torch_maxpool(x_noncontig)
    # If the kernel does not handle non-contiguous memory, the outputs may not match.
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"
