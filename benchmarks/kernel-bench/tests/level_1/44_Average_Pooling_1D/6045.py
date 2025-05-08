
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension from kernel.cu
    cuda_module = load(
        name="avg_pool1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test shared memory over-allocation.
# Use parameters that force a very large shared memory allocation.
def test_shared_memory_allocation():
    cuda_mod = build_kernel()
    batch_size = 2
    in_channels = 2
    # Pick an input length and kernel parameters such that the computed shared memory tile is huge.
    # For BLOCK_SIZE=256, using large stride and kernel_size makes shared memory requirement enormous.
    input_length = 4096
    kernel_size = 1024  # very large kernel window
    stride = 512        # large stride
    # Create a simple input tensor (float32, contiguous, on GPU)
    x = torch.randn(batch_size, in_channels, input_length, device='cuda', dtype=torch.float32)
    # The kernel should be launched with huge shared memory; if the device cannot allocate, an error would be raised.
    with pytest.raises(Exception):
        out = cuda_mod.forward(x, kernel_size, stride, padding=0)
        # Force synchronization to catch asynchronous errors
        torch.cuda.synchronize()

# Issue 2: Test data-type limitation.
def test_input_tensor_type():
    cuda_mod = build_kernel()
    batch_size = 4
    in_channels = 4
    input_length = 128
    kernel_size = 4
    stride = 2
    padding = 1
    # Create an input in a data type other than float32 (e.g., float64)
    x = torch.randn(batch_size, in_channels, input_length, device='cuda', dtype=torch.float64)
    with pytest.raises(RuntimeError):
        out = cuda_mod.forward(x, kernel_size, stride, padding)
        torch.cuda.synchronize()

# Issue 3: Test non-contiguous input.
def test_non_contiguous_input():
    cuda_mod = build_kernel()
    batch_size = 4
    in_channels = 4
    input_length = 128
    kernel_size = 4
    stride = 2
    padding = 1

    # Create a contiguous input then deliberately make it non-contiguous via transpose.
    x = torch.randn(batch_size, in_channels, input_length, device='cuda', dtype=torch.float32)
    # Transpose channels and batch dims to make it non-contiguous, then transpose back to correct shape.
    x_noncontig = x.transpose(0, 1).transpose(1, 0)  # This double-transpose produces a non-contiguous view
    assert not x_noncontig.is_contiguous(), "Test setup failed: tensor is contiguous."

    # Compute the expected output using PyTorch's built-in AvgPool1d.
    avg_pool = torch.nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
    expected = avg_pool(x_noncontig)

    # Run the custom kernel.
    out = cuda_mod.forward(x_noncontig, kernel_size, stride, padding)
    torch.cuda.synchronize()
    # The kernel assumes contiguous memory; hence, its output may differ from reference.
    # We demand that the outputs are close, which will likely fail.
    assert not torch.allclose(out, expected, atol=1e-5), "Non-contiguous input should lead to incorrect results in the kernel."

# Issue 4: Test input tensor dimensionality check.
def test_input_tensor_dimensions():
    cuda_mod = build_kernel()
    # Create a 2D tensor instead of the expected 3D tensor.
    x = torch.randn(16, 128, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = cuda_mod.forward(x, kernel_size=4, stride=2, padding=1)
        torch.cuda.synchronize()
