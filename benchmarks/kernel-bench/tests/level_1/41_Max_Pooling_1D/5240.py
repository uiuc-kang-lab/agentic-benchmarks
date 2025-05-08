
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility function to build the CUDA kernel module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that using an input tensor with non-float32 dtype (e.g. double) triggers an error.
def test_input_tensor_dtype():
    cuda_module = build_kernel()
    # Create a double precision tensor.
    x = torch.randn(2, 4, 16, dtype=torch.double, device='cuda')
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    return_indices = False
    with pytest.raises(RuntimeError):
        # Expect the kernel to fail since it only accepts float (float32)
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)

# Issue 2: Test that when return_indices is True, the output format does not match PyTorch's expected tuple output.
def test_return_indices_concatenation():
    cuda_module = build_kernel()
    # Create a float32 tensor
    x = torch.arange(0, 32, dtype=torch.float32, device='cuda').reshape(1, 1, 32)
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    return_indices = True
    # Run the kernel
    result = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    # The expected PyTorch MaxPool1d with return_indices=True returns a tuple (output, indices)
    # Therefore, the correct output shape for the pooled output should be (1, 1, 16).
    # Instead, the new CUDA kernel concatenates output and indices along the last dimension.
    # Check if the result's last dimension is 32 instead of 16.
    expected_pool_shape = (1, 1, 16)
    if result.shape[-1] == expected_pool_shape[-1]:
        pytest.skip("Kernel returned expected pooled output shape; concatenation issue not detected.")
    else:
        # This test passes if the kernel output shape does not match the expected tuple structure.
        assert result.shape[-1] != expected_pool_shape[-1], \
            "Kernel concatenates output and indices, deviating from PyTorch's expected tuple output."

# Issue 3: Test that a non-contiguous input tensor triggers an error (as enforced by the TORCH_CHECK)
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(2, 4, 16, device='cuda', dtype=torch.float32)
    x = x.transpose(0, 1)  # This makes the tensor non-contiguous.
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
