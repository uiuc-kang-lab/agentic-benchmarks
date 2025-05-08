
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension
def build_kernel():
    module_path = os.path.dirname(os.path.abspath(__file__))
    cuda_module = load(
        name="leaky_relu_cuda",
        sources=[os.path.join(module_path, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1:
# Attempt to run the kernel on a tensor with dtype other than float32.
# Expected result: The kernel should either error out or produce incorrect results,
# so we capture the CUDA error.
def test_non_float32_dtype():
    cuda_module = build_kernel()
    # Create a double precision tensor - the kernel expects float32
    x = torch.randn(128, 128, dtype=torch.double, device="cuda")
    negative_slope = 0.01
    with pytest.raises(RuntimeError):
        # Trying to call the kernel with the wrong dtype should raise an error.
        out = cuda_module.forward(x, negative_slope)
        torch.cuda.synchronize()

# Test case 2:
# Create a misaligned tensor by slicing out a sub-tensor with an offset.
# This misalignment may trigger problems with the float4 vectorized access.
def test_misaligned_tensor():
    cuda_module = build_kernel()
    # Allocate extra element to easily obtain a misaligned view.
    x_full = torch.randn(1000 + 1, dtype=torch.float32, device="cuda")
    # Slicing starting from index 1 can result in a pointer that is not 16-byte aligned.
    # Note: The sliced tensor is still contiguous in itself.
    x = x_full[1:]
    negative_slope = 0.02
    # Even if the operation might run without a CUDA error on some hardware, misaligned
    # accesses can produce correctness issues. Here we check for a runtime error.
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, negative_slope)
        torch.cuda.synchronize()
