
import torch
import pytest
from torch.utils.cpp_extension import load

# This function builds the extension from kernel.cu.
def build_kernel():
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Tail-element handling.
# Create an input tensor whose number of elements is not divisible by 4.
# The wrong tail handling should produce incorrect results.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
def test_tail_handling():
    kernel = build_kernel()
    # Create a tensor with a total number of elements not divisible by 4.
    # e.g., a (7, 7) matrix => 49 elements, 49 mod 4 = 1 element extra.
    A = torch.randn(7, 7, device="cuda", dtype=torch.float32)
    s = 2.5
    # Call the kernel
    C = kernel.forward(A, s)
    torch.cuda.synchronize()
    # Reference output computed in PyTorch
    C_ref = A * s
    # We expect the outputs to be different due to tail handling error.
    # In a correctly written kernel, these would match.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Tail handling bug not triggered: output unexpectedly matches reference."

# Issue 2: Alignment assumptions.
# Provide a non-contiguous tensor. For example, by slicing a larger tensor.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
def test_alignment_requirement():
    kernel = build_kernel()
    # Create a larger tensor and then take a slice that is likely not 16-byte aligned.
    A_full = torch.randn(17, 17, device="cuda", dtype=torch.float32)
    A = A_full[:, 1:]  # slicing along a dimension typically makes tensor non-contiguous
    s = 3.14
    # Make sure A is non-contiguous.
    assert not A.is_contiguous(), "Test tensor is unexpectedly contiguous"
    # Run the kernel; if the kernel assumes proper alignment, the behavior can be undefined.
    C = kernel.forward(A, s)
    torch.cuda.synchronize()
    C_ref = A * s
    # Expect a discrepancy due to misaligned vectorized loads/stores.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Alignment bug not triggered: output unexpectedly matches reference."

# Issue 3: Missing grid-stride loop coverage.
# Create a tensor that is so large that the originally computed launch configuration 
# (which assigns one vector per thread) might not cover all elements correctly.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
def test_grid_stride_deficiency():
    kernel = build_kernel()
    # Set up a tensor with a moderate number of elements that is not a multiple of 4
    # and where the computed grid does not allow every thread to process multiple items.
    # For example, use a (1025, 1025) matrix.
    A = torch.randn(1025, 1025, device="cuda", dtype=torch.float32)
    s = 0.5
    C = kernel.forward(A, s)
    torch.cuda.synchronize()
    C_ref = A * s
    # In a correct implementation (with grid-stride loops) the result would be exact.
    # Here, we test that some elements are processed incorrectly.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Grid-stride loop issue not triggered: output unexpectedly matches reference."

# Issue 4: Handling of non-contiguous tensors.
# Even if alignment is not the only problem with non-contiguous memory,
# the kernel assumes a simple memory layout.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
def test_noncontiguous_input():
    kernel = build_kernel()
    # Create a contiguous tensor and then transpose to make it non-contiguous.
    A = torch.randn(128, 256, device="cuda", dtype=torch.float32).t()  # now non-contiguous
    s = 1.1
    assert not A.is_contiguous(), "Test tensor is unexpectedly contiguous"
    C = kernel.forward(A, s)
    torch.cuda.synchronize()
    C_ref = A * s
    # We expect a mismatch due to the kernel not handling non-contiguous input.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Non-contiguous input bug not triggered: output unexpectedly matches reference."
