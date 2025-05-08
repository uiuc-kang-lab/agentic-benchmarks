
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Load the CUDA extension from kernel.cu.
    cuda_module = load(
        name="custom_sum_reduce",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2: Test with a non-contiguous tensor.
def test_non_contiguous_input():
    mod = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous with a transpose.
    x = torch.randn(16, 256, 256, device='cuda')
    # Transpose to break contiguity. For example, swap dim0 and dim1:
    x_noncontig = x.transpose(0, 1)
    # Use reduction along the new dim 1.
    dim = 1  
    # The CUDA kernel does not account for nonstandard strides,
    # so the result will likely differ from torch.sum.
    result_kernel = mod.forward(x_noncontig, dim)
    result_torch = torch.sum(x_noncontig, dim=dim, keepdim=True)
    
    # We expect them to differ.
    with pytest.raises(AssertionError):
        assert torch.allclose(result_kernel, result_torch, atol=1e-5), \
            f"Kernel result matches expected output even for non-contiguous input!"

# Issue 2 (additional check on arbitrary strides):
def test_arbitrary_strides():
    mod = build_kernel()
    # Create a larger tensor and then slice it to create irregular strides.
    x = torch.randn(32, 64, 32, device='cuda')
    # Slicing with a step produces non-contiguous tensor with arbitrary strides.
    x_arbitrary = x[:, ::2, :]
    # Reduce along the middle dimension.
    dim = 1
    result_kernel = mod.forward(x_arbitrary, dim)
    result_torch = torch.sum(x_arbitrary, dim=dim, keepdim=True)
    
    # We expect the CUDA kernel (which ignores the actual strides) to produce differences.
    with pytest.raises(AssertionError):
        assert torch.allclose(result_kernel, result_torch, atol=1e-5), \
            f"Kernel result should differ from torch.sum for arbitrarily strided tensor!"

# Issue 3: Test that the misleading shared memory claim does not affect correctness,
# but note that there is no actual shared memory use.
def test_shared_memory_claim():
    mod = build_kernel()
    x = torch.randn(8, 10, 12, device='cuda')
    dim = 1
    result = mod.forward(x, dim)
    # At least the kernel runs and returns a tensor of expected shape.
    expected_shape = list(x.size())
    expected_shape[dim] = 1
    assert list(result.shape) == expected_shape, \
        f"Expected output shape {expected_shape}, got {result.shape}"

# Issue 4: Test behavior with a non-floating point type.
def test_non_floating_point_dtype():
    mod = build_kernel()
    # Create an integer tensor. The kernel dispatch only covers floating-point types.
    x_int = torch.randint(0, 10, (4, 5, 6), device='cuda', dtype=torch.int32)
    dim = 1
    with pytest.raises(RuntimeError):
        # Expect a runtime error because the kernel does not support int32.
        _ = mod.forward(x_int, dim)
