
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to compile the CUDA kernel from kernel.cu.
def build_kernel():
    # Assume kernel.cu is in the same directory as this test file
    src_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="l1_norm_module",
        sources=[src_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1: Misaligned memory due to vectorized float4 loads.
# This test forces a misaligned starting pointer by creating a tensor with an extra column
# and then slicing along the second dimension such that the underlying data pointer is offset.
@pytest.mark.cuda
def test_unaligned_memory():
    cuda_module = build_kernel()
    # Create a tensor with one extra element so that when we slice off one column we get a misaligned pointer.
    # Typically, PyTorch tensors are 16-byte aligned. By creating an extra column and then slicing, we can force a misalignment.
    batch_size = 1
    dim = 16385  # 16385 elements => 16385*4 bytes is not a multiple of 16 if we shift it.
    x_full = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    # Slice to get a tensor of shape (1, 16384) but with non-zero storage offset.
    x = x_full[:, 1:].contiguous()  # This new tensor likely has a starting pointer that is not 16-byte aligned.
    # Compute expected output on CPU/GPU
    norm = torch.sum(torch.abs(x), dim=1, keepdim=True).clamp_min(1e-12)
    expected = x / norm
    # Run the kernel
    out = cuda_module.forward(x)
    torch.cuda.synchronize()
    # If the misalignment issue occurs, we expect the output to deviate from the expected result.
    # We test that the maximum absolute difference is larger than a very small tolerance.
    max_diff = (out - expected).abs().max().item()
    assert max_diff > 1e-3, f"Expected misaligned memory to cause an error, but max difference was {max_diff}"

# Issue 2: Incorrect warp-level reduction when blockDim.x < warpSize.
# When dimension D is small (<32), the kernel launches with less than 32 threads.
# The fixed mask in __shfl_down_sync may include inactive lanes leading to an incorrect norm.
@pytest.mark.cuda
def test_small_dimension():
    cuda_module = build_kernel()
    batch_size = 1
    dim = 16  # Less than warp size (32)
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float32)
    norm = torch.sum(torch.abs(x), dim=1, keepdim=True).clamp_min(1e-12)
    expected = x / norm
    out = cuda_module.forward(x)
    torch.cuda.synchronize()
    max_diff = (out - expected).abs().max().item()
    assert max_diff > 1e-3, f"Kernel warp reduction bug not triggered; max diff: {max_diff}"

# Issue 3: Kernel only supports 2D tensors.
# If we pass a higher-dimensional tensor, the kernel should trigger a TORCH_CHECK failure.
@pytest.mark.cuda
def test_higher_dimensions():
    cuda_module = build_kernel()
    x = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)  # 3D tensor
    with pytest.raises(RuntimeError, match="Expected a 2D tensor."):
        _ = cuda_module.forward(x)

# Issue 4: Kernel assumes float32 input but does not check the type.
# Passing a non-float32 tensor (e.g. float64) may lead to misinterpretation of the data.
@pytest.mark.cuda
def test_non_float():
    cuda_module = build_kernel()
    batch_size = 1
    dim = 16384
    # Create a tensor with type float64
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x)
