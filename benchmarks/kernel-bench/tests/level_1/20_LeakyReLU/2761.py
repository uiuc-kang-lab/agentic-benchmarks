
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test for tensor size not being a multiple of 4.
def test_non_multiple_of_vector_size():
    kernel_module = build_kernel()
    # Create a tensor where the number of elements is not a multiple of 4.
    # For example, a 1-D tensor of size 10.
    x = torch.randn(10, device="cuda", dtype=torch.float32).contiguous()
    # We expect an out-of-bounds access (which may result in a CUDA error)
    with pytest.raises(RuntimeError):
        # Launch the kernel. According to the kernel implementation, this should
        # cause an illegal memory access when reading a float4 near the end.
        y = kernel_module.forward(x, 0.01)
        torch.cuda.synchronize()  # force synchronization to trigger error

# Issue 2: Test for input tensor having wrong data type (float64).
def test_wrong_dtype():
    kernel_module = build_kernel()
    # Create a tensor of type double, but the kernel expects float32.
    x = torch.randn(1024, device="cuda", dtype=torch.float64).contiguous()
    with pytest.raises(RuntimeError):
        # The reinterpret_cast to float4 will misinterpret data if dtype isn't float,
        # or the kernel may simply crash.
        y = kernel_module.forward(x, 0.01)
        torch.cuda.synchronize()

# Issue 3: Test for misaligned tensor.
def test_misaligned_tensor():
    kernel_module = build_kernel()
    # Create a tensor with extra padding and then slice it to create a misaligned pointer.
    # Allocating an extra element and slicing from index 1 often results in misalignment.
    base = torch.randn(1025, device="cuda", dtype=torch.float32)
    x = base[1:].contiguous()  # Force a sub-tensor that may not start at a naturally aligned boundary.
    # Although contiguous, the start address may no longer be 16-byte aligned for float4 loads.
    with pytest.raises(RuntimeError):
        y = kernel_module.forward(x, 0.01)
        torch.cuda.synchronize()
