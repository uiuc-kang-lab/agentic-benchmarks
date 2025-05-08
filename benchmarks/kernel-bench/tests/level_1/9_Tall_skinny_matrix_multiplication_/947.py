
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# 1. Test for float32-only assumption.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create double tensors which the kernel does not support.
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, dtype=torch.float64, device="cuda")
    B = torch.randn(K, N, dtype=torch.float64, device="cuda")
    with pytest.raises(Exception):
        # Expecting an exception because the kernel uses float pointers.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# 2. Test for contiguous input assumption.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create contiguous input and then produce a non-contiguous view.
    M, K, N = 64, 32, 48
    A_full = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B_full = torch.randn(K, N, dtype=torch.float32, device="cuda")
    # Make non-contiguous by transposing (Note: for 2D tensors, .t() returns non-contiguous)
    A = A_full.t()
    B = B_full.t()
    # Even though the shapes swap, we force the kernel by providing non-standard shapes via our API.
    # The kernel logic may not be compatible with non-contiguous memory.
    with pytest.raises(Exception):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# 3. Test for nonstandard strides (e.g. negative strides).
def test_negative_stride_tensor():
    my_module = build_kernel()
    # Create a tensor and then flip it to get negative strides.
    M, K, N = 32, 16, 24
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    A_neg = torch.flip(A, dims=[0])
    B_neg = torch.flip(B, dims=[0])
    # Although torch.matmul supports flipped tensors, our kernel’s manual indexing is not designed 
    # to support negative strides.
    with pytest.raises(Exception):
        C = my_module.forward(A_neg, B_neg)
        torch.cuda.synchronize()

# 4. Test for fixed BLOCK_SIZE issues when dimensions are not multiples of BLOCK_SIZE.
def test_non_multiple_dimensions():
    my_module = build_kernel()
    # Choose dimensions that are not multiples of BLOCK_SIZE (16).
    M, K, N = 30, 22, 37  # None of these are multiples of 16.
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    # This should trigger potential indexing issues in the tiling scheme.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # If the kernel mishandles boundaries, the result will not match.
    assert torch.allclose(C, C_ref, atol=1e-4), f"Kernel output differs from torch.matmul! Max diff: {(C - C_ref).abs().max()}"

# 5. Test for missing CUDA kernel launch error checking.
def test_invalid_dimension_error():
    my_module = build_kernel()
    # Create 3D tensors (invalid dimensions for the implemented 2D matmul kernel) to trigger the dimension check.
    A = torch.randn(4, 4, 4, dtype=torch.float32, device="cuda")
    B = torch.randn(4, 4, 4, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
