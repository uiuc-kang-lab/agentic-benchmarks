
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension module from the kernel file "kernel.cu"
def build_kernel():
    cuda_module = load(
        name="custom_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Non‐contiguous input via transposition.
# The kernel computes the reduction using flat pointer arithmetic assuming contiguity.
# Thus, when an input is transposed the output will differ from torch.sum.
def test_non_contiguous_input_transpose():
    cuda_module = build_kernel()

    # Generate a tensor and then transpose it.
    # Suppose original shape is (batch, height, width). We choose to reduce over dimension 1.
    # For a contiguous tensor, the reduction dimension (dim=1) is assumed to have stride equal to width.
    # Here, after transpose the real stride for the chosen reduction dimension
    # is not equal to the computed inner_size.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float32)
    x_t = x.transpose(1, 2).contiguous().transpose(1, 2)  # remove contiguity but keep logical order different
    # The above trick “undoes” the contiguity but leaves non-standard strides.
    reduce_dim = 1

    # Compute expected result using PyTorch’s built-in sum reduction
    expected = torch.sum(x_t, dim=reduce_dim, keepdim=True)

    # Call the CUDA kernel (which is not stride-aware) 
    kernel_out = cuda_module.forward(x_t, reduce_dim)
    torch.cuda.synchronize()

    # We expect that the kernel output is incorrect
    assert not torch.allclose(kernel_out, expected, atol=1e-3), \
        "Test should have triggered a failure because of non-contiguous input layout but did not."

# Test case 2: Non-standard stride via advanced slicing.
# Slicing can produce non-contiguous tensors.
def test_non_contiguous_input_slicing():
    cuda_module = build_kernel()

    # Create a tensor and take a slice that is not contiguous.
    x = torch.randn(32, 64, 128, device="cuda", dtype=torch.float32)
    # Take every other element along dimension 1. The resulting tensor is non-contiguous.
    x_slice = x[:, ::2, :]
    reduce_dim = 1

    # Compute expected result using torch.sum
    expected = torch.sum(x_slice, dim=reduce_dim, keepdim=True)

    # Call the CUDA kernel
    kernel_out = cuda_module.forward(x_slice, reduce_dim)
    torch.cuda.synchronize()

    # Since the kernel does not fetch values according to the true strides, we expect error.
    assert not torch.allclose(kernel_out, expected, atol=1e-3), \
        "Test should have triggered a failure because of sliced (non-contiguous) input layout but did not."
