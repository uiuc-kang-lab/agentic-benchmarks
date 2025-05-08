
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Always rebuild the module for testing purposes.
    module = load(
        name="combined_coalesced_max_reduce",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Incorrect handling of non-contiguous inputs.
# The kernel calculates indices assuming a contiguous memory layout.
# This test creates a noncontiguous view (by transposing two dimensions) so that the kernel’s arithmetic is wrong.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create a contiguous tensor and then obtain a non-contiguous view by transposing two dimensions.
    batch_size, dim1, dim2 = 16, 256, 256
    x = torch.randn(batch_size, dim1, dim2, device="cuda")
    x_nc = x.transpose(1, 2)  # non contiguous view (shape: [16, 256, 256])
    # Here we choose to reduce on dimension 1.
    dim = 1  
    try:
        output_kernel = cuda_module.forward(x_nc, dim)
    except Exception as e:
        pytest.fail("Kernel raised an exception on non-contiguous input: {}".format(e))
    # Expected output computed with torch.max does work correctly with non-contiguous tensors.
    output_ref = torch.max(x_nc, dim=dim)[0]
    # We expect the kernel result to be different since it assumed a contiguous layout.
    # Hence, the test passes only if the kernel output does NOT match the correct answer.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct result on a non-contiguous input, but it relies on contiguous layout assumptions."

# Issue 2: Use of "#pragma unroll" with a loop bound that is not compile-time constant.
# Although this might not crash, it can lead to suboptimal performance or even erroneous behavior
# for unusual reduction dimensions. We test a case with a reduction dimension size that is large 
# (and not a known compile-time constant) to see if the computed result deviates from torch.max.
def test_large_reduction_dim():
    cuda_module = build_kernel()
    # Create an input tensor where the reduction dimension is large.
    # For example, suppose we have shape [outer, reduction, inner] = [8, 10000, 4]
    outer, reduction, inner = 8, 10000, 4
    x = torch.randn(outer, reduction, inner, device="cuda")
    dim = 1  # reduce along the long dimension
    try:
        output_kernel = cuda_module.forward(x, dim)
    except Exception as e:
        pytest.fail("Kernel raised an exception for a large reduction dimension: {}".format(e))
    output_ref = torch.max(x, dim=dim)[0]
    # If the pragma unroll causes unintended behavior, the kernel output may be off.
    # We expect that for a correct implementation these should match.
    assert torch.allclose(output_kernel, output_ref, atol=1e-5), \
        "Kernel output does not match torch.max for a large reduction dimension, possibly due to misuse of unrolling pragmas."

# Issue 3: Limited dtype support (only floating point and half).
# This test creates an integer tensor and expects the kernel launch to fail (or at least not process the data correctly).
def test_invalid_dtype():
    cuda_module = build_kernel()
    # Create an integer tensor (which is not supported by the AT_DISPATCH_FLOATING_TYPES_AND_HALF macro)
    x = torch.randint(0, 100, (16, 256, 256), device="cuda", dtype=torch.int32)
    dim = 1
    with pytest.raises(RuntimeError):
        # The kernel should raise an error (or the dispatch mechanism should not find a valid kernel)
        _ = cuda_module.forward(x, dim)
