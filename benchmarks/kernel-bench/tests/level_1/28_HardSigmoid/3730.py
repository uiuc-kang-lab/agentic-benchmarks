
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build and load our CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference hard sigmoid function in Python
def ref_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 3) / 6, min=0, max=1)

# Test 1: numel not divisible by VEC_SIZE.
# For float, VEC_SIZE is 4 so number of elements should be a non-multiple of 4.
def test_non_multiple_vector_size():
    cuda_module = build_kernel()
    # Create a tensor whose total elements is not a multiple of 4
    x = torch.linspace(-10, 10, steps=10, device='cuda', dtype=torch.float32)
    # Invoke the CUDA kernel function which is expected to have out-of-bound issues in its vectorized loop.
    y = cuda_module.forward(x)
    torch.cuda.synchronize()
    y_ref = ref_hardsigmoid(x)
    # Expect the result to differ from the reference, indicating a bug.
    assert not torch.allclose(y, y_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct output for a non-multiple vector size input."

# Test 2: Misaligned memory access.
# We simulate misalignment by using a sub-tensor view that does not start at the underlying allocation start.
def test_misaligned_memory():
    cuda_module = build_kernel()
    # Create a larger tensor and then take a sub-tensor view that is likely misaligned.
    base = torch.randn(1025, device='cuda', dtype=torch.float32)
    # Taking a slice that does not start at the first element; the pointer returned by data_ptr() will be offset.
    # Even though the tensor remains contiguous, the starting address may not meet the alignment requirements for float4.
    x = base[1:]
    y = cuda_module.forward(x)
    torch.cuda.synchronize()
    y_ref = ref_hardsigmoid(x)
    # We expect the misaligned version to produce incorrect results due to vectorized load/store assumptions.
    assert not torch.allclose(y, y_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct output for misaligned memory input."

# Test 3: Unsupported data type.
# The kernel only supports float and double. Using a half-precision tensor should trigger an error.
def test_unsupported_dtype():
    cuda_module = build_kernel()
    x = torch.randn(1024, device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # The kernel dispatch macro does not cover half-precision, so we expect a RuntimeError.
        y = cuda_module.forward(x)
        torch.cuda.synchronize()
        
if __name__ == "__main__":
    pytest.main([__file__])
