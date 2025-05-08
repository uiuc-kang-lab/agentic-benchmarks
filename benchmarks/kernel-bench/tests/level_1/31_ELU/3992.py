
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

# Test case 1: Trigger out-of-bound memory access by using a tensor size 
# that is not a multiple of 4. The flawed boundary check causes the kernel 
# to access memory out-of-bound.
def test_boundary_check():
    my_module = build_kernel()
    # Create a 1-D tensor of size 257 so that idx=257 produces an out-of-bound access.
    x = torch.randn(257, device="cuda", dtype=torch.float32)
    # In this kernel, alpha is a scalar.
    # When the kernel accesses index 257 it should trigger a CUDA error.
    with pytest.raises(RuntimeError):
        out = my_module.forward(x, 1.0)
        torch.cuda.synchronize()  # Force kernel synchronization to catch error

# Test case 2: Test the wrong alignment logic.
# Although the alignment code does nothing productive, we can trigger the error
# by having an input tensor whose number of elements is not a multiple of 4.
def test_alignment_logic():
    my_module = build_kernel()
    # Create a tensor with length 5 (not a multiple of 4), where the alignment logic will
    # compute an aligned index that passes the boundary test for threads with idx>=n.
    x = torch.randn(5, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = my_module.forward(x, 1.0)
        torch.cuda.synchronize()

# Test case 3: The kernel does not support data types other than float32.
# Passing a double tensor should raise an error (or crash).
def test_input_dtype():
    my_module = build_kernel()
    # Create a double tensor. CHECK_INPUT only verifies CUDA and contiguity,
    # but the kernel code casts data_ptr to float*. This mismatch can lead to errors.
    x = torch.randn(16, 16, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        out = my_module.forward(x, 1.0)
        torch.cuda.synchronize()

# Test case 4: Passing a non-contiguous tensor should trigger the CHECK_INPUT error.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    non_contiguous_x = x.t()  # Transpose to make non-contiguous
    with pytest.raises(RuntimeError):
        out = my_module.forward(non_contiguous_x, 1.0)
        torch.cuda.synchronize()
