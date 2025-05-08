
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="min_reduce_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Non‑contiguous tensor issues.
# The kernel immediately calls input.contiguous(), which may hide potential issues in non‑contiguous scenarios.
# In a more general implementation, we might want to support arbitrary strides.
def test_non_contiguous_tensor():
    # Create a tensor and deliberately make it non‑contiguous by transposing dimensions.
    x = torch.randn(16, 256, 256, device='cuda')
    x_non_contig = x.transpose(0, 1)  # now non‑contiguous
    # Use reduction over dim=0 in the original shape (after transpose, reduction is more complicated!)
    # The kernel will call .contiguous() and treat the memory as contiguous,
    # so the result will not match a reduction done on the non‑contiguous view.
    # Thus, we flag that the kernel is not truly general.
    my_module = build_kernel()
    with pytest.raises(AssertionError):
        # Expect that if we compare to PyTorch's min reduction on the non‐contiguous tensor,
        # the results will differ in general.
        result_kernel = my_module.forward(x_non_contig, 0)
        result_torch = torch.min(x_non_contig, dim=0)[0]
        # Force an error by checking for equality even if the kernel did a .contiguous() conversion.
        assert torch.allclose(result_kernel, result_torch), "Kernel did not correctly handle non-contiguous input."

# Test case 2: Reduction on an empty reduction dimension.
# When the chosen reduction dimension is zero-sized, the kernel will try to access an element out-of-bounds.
def test_empty_reduction_dimension():
    # Create a tensor with an empty reduction dimension.
    # For example, let the reduction dimension (dim=1) be 0.
    x = torch.randn(16, 0, 256, device='cuda')
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel to crash or throw an error due to accessing out-of-bound memory.
        _ = my_module.forward(x, 1)
        torch.cuda.synchronize()

# Test case 3: Unsupported data type (float16).
# The kernel is dispatched only through AT_DISPATCH_ALL_TYPES so likely does not support half.
def test_unsupported_half_dtype():
    x = torch.randn(16, 256, 256, device='cuda', dtype=torch.float16)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x, 1)
        torch.cuda.synchronize()

# Test case 4: Unsupported complex dtype.
# Complex numbers are not handled by AT_DISPATCH_ALL_TYPES.
def test_unsupported_complex_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x = torch.randn(16, 256, 256, device='cuda', dtype=torch.complex64)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x, 1)
        torch.cuda.synchronize()
