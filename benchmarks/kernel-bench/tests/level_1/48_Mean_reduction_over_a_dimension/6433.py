
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    cuda_module = load(
        name="mean_reduce_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function: a CPU reference implementation for mean reduction
def reference_mean(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.mean(input_tensor, dim=dim)

# Issue 1 test: non-contiguous input tensor
# The kernel assumes contiguous memory but a non-contiguous tensor (e.g. from a transpose) will yield wrong results.
def test_non_contiguous_input():
    # Create a contiguous tensor and then transpose to get a non-contiguous one.
    x = torch.randn(16, 256, 256, device='cuda')
    # Make tensor non-contiguous by transposing first two dims.
    x_noncontig = x.transpose(0, 1)
    # Reduce along dimension 2 (which is still the last dimension for both x and x_noncontig)
    reduction_dim = 2

    # Build module
    mod = build_kernel()

    # Kernel invocation: The kernel expects a contiguous tensor for correct indexing.
    # So our kernel result (when given non-contiguous input) likely differs from reference.
    result_kernel = mod.forward(x_noncontig, reduction_dim)
    result_ref = reference_mean(x_noncontig, reduction_dim)

    # They should NOT be close if the kernel fails to properly account for non-contiguity.
    assert not torch.allclose(result_kernel, result_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct result on non-contiguous input."

# Issue 2 test: invalid reduction dimension
def test_invalid_dimension():
    mod = build_kernel()
    # Create 1D tensor but pass an out-of-bound dimension (e.g. 1 for a 1D tensor)
    x = torch.randn(10, device='cuda')
    invalid_dim = 1
    with pytest.raises(IndexError):
        # Expect an out-of-bound error at the CPU dispatch stage when trying to access sizes[invalid_dim].
        mod.forward(x, invalid_dim)

# Issue 3 test: lack of kernel error checking (simulate an error)
# One way to simulate an error is to provide an input tensor with unsupported dtype.
def test_invalid_dtype():
    mod = build_kernel()
    # Create an integer tensor on CUDA. The kernel dispatch is done for floating types only.
    x = torch.randint(0, 10, (16, 256, 256), dtype=torch.int32, device='cuda')
    reduction_dim = 1
    with pytest.raises(RuntimeError):
        # Expect a runtime error from the type-dispatch mechanism in AT_DISPATCH_FLOATING_TYPES.
        mod.forward(x, reduction_dim)
        
# Issue 4 test: if a CUDA error occurs during kernel execution and is not caught, the result may be garbage.
# We try a case that might lead to a kernel error (for instance, a tensor with zero size along the reduction dim).
def test_zero_length_reduction():
    mod = build_kernel()
    # Create tensor with zero length in reduction dimension.
    x = torch.randn(16, 0, 256, device='cuda')
    reduction_dim = 1
    with pytest.raises(RuntimeError):
        # The kernel will likely launch with dim_size=0, causing a division by zero or invalid memory access.
        mod.forward(x, reduction_dim)
