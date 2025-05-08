
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build and load the CUDA extension from kernel.cu
def build_kernel():
    return load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )

# Test case 1: Pass a tensor of a type other than float32.
# Expect that the kernel returns an incorrect result (or raises an error) because it
# incorrectly interprets the tensor data.
def test_dtype_issue():
    cuda_module = build_kernel()
    # Create a tensor with double (float64) type on CUDA
    x = torch.randn(128, 4000, device="cuda", dtype=torch.float64)
    # The kernel does not check type and will use x.data_ptr<float>()
    out_kernel = cuda_module.forward(x, 1)
    # Reference output computed with torch.cumsum (after casting x to float64 so that both are the same type)
    out_ref = torch.cumsum(x, dim=1)
    # The kernel output should be different from the correct result.
    assert not torch.allclose(out_kernel, out_ref), (
        "Kernel incorrectly handled non-float32 input: output matches torch.cumsum "
        "despite the type mismatch."
    )

# Test case 2: Create an input tensor with an inner_size greater than allowed threads per block.
# Here we deliberately create a tensor such that the product of dimensions after the cumulative sum dimension
# exceeds 1024. The kernel launch configuration uses inner_size as the number of threads per block,
# and this should trigger an error.
def test_thread_block_limit():
    cuda_module = build_kernel()
    # Create a tensor of shape (2, 10, 2048). For cumulative sum along dimension 1:
    # outer_size = 2, inner_size = 2048. This inner_size exceeds typical CUDA block size limits.
    x = torch.randn(2, 10, 2048, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # The kernel launch should fail because 2048 threads per block exceeds the limit.
        _ = cuda_module.forward(x, 1)

# Test case 3: Create an input tensor that is contiguous but misaligned with respect to 128-bit boundaries.
# We do this by narrowing the tensor along the cumulative dimension.
# The kernel assumes 128-bit alignment, so a misaligned tensor may produce incorrect cumulative sum results.
def test_alignment_issue():
    cuda_module = build_kernel()
    # Create a contiguous tensor of shape (128, 4000)
    x_full = torch.randn(128, 4000, device="cuda", dtype=torch.float32)
    # Narrow the tensor along dimension 1, shifting the underlying pointer.
    # This produces a new tensor that is still contiguous but likely not 16-byte aligned.
    x = x_full.narrow(1, 1, x_full.size(1) - 1)
    # Check that the tensor is contiguous
    assert x.is_contiguous(), "The tensor must be contiguous."
    # Check if the underlying pointer is misaligned for 16 bytes (128 bits).
    # For float32 (4 bytes per element), the pointer should be a multiple of 4 elements.
    ptr = x.data_ptr()
    if ptr % 16 == 0:
        pytest.skip("Unable to create a misaligned tensor; the storage is 16-byte aligned.")
    # Compute the cumulative sum using our kernel
    out_kernel = cuda_module.forward(x, 1)
    # Compute the reference result using PyTorch's cumsum
    out_ref = torch.cumsum(x, dim=1)
    # The results should differ because the kernel assumed aligned memory.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-4), (
        "Kernel incorrectly handled misaligned memory: output matches torch.cumsum "
        "despite misalignment."
    )
