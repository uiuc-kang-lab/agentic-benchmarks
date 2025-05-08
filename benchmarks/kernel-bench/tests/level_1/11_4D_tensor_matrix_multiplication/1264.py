
import torch
import pytest
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

# A helper function to compute the reference result using einsum.
def reference_einsum(A, B):
    # A shape: [b, i, j, l]; B shape: [l, k]
    return torch.einsum("bijl,lk->bijk", A, B)

# Test 1: Non-contiguous input tensor A
def test_non_contiguous_A():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    b, i, j, l, k = 16, 32, 32, 16, 24
    # Create contiguous tensors
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    # Make A non-contiguous by transposing two inner dimensions and then transposing back
    A_noncontig = A.transpose(2, 3)
    A_noncontig = A_noncontig.transpose(2, 3)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    
    my_kernel = build_kernel()
    # The kernel does not account for non-contiguous A, so the result will be wrong.
    C_kernel = my_kernel.forward(A_noncontig, B)
    C_ref = reference_einsum(A_noncontig, B)
    # The test is expected to fail (i.e. the kernel result will not match the reference)
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel should fail for non-contiguous A."

# Test 2: Wrong block size assumption (simulate by passing tensor sizes that force K not to be a multiple of TILE_K)
def test_K_not_multiple_of_TILE_K():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    b, i, j, l, k = 8, 16, 16, 32, 50  # k=50 which is not a multiple of TILE_K (32)
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    my_kernel = build_kernel()
    C_kernel = my_kernel.forward(A, B)
    C_ref = reference_einsum(A, B)
    # Even though the kernel has a check on (k < K) when loading B, the tiling assumption (each block has TILE_K threads)
    # might lead to issues on the edge.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel should produce incorrect results when K is not a multiple of TILE_K."

# Test 3: Input tensors with wrong data type (float64 instead of float32)
def test_wrong_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    b, i, j, l, k = 4, 8, 8, 10, 12
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float64)
    B = torch.randn(l, k, device="cuda", dtype=torch.float64)
    my_kernel = build_kernel()
    # The kernel is compiled for float only.
    with pytest.raises(RuntimeError):
        _ = my_kernel.forward(A, B)

# Test 4: Non-standard layout for the flattened (batch, i, j) grouping
def test_nonstandard_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    b, i, j, l, k = 6, 10, 10, 15, 20
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    # Create a tensor with unexpected strides by permuting dimensions and then making it contiguous in that new order.
    A_perm = A.permute(0, 2, 1, 3).contiguous()
    # Note: The kernel assumes A is laid out as [b, i, j, l] flattened along the first three dims,
    # so A_perm has a different memory layout.
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    my_kernel = build_kernel()
    C_kernel = my_kernel.forward(A_perm, B)
    # Compute reference using einsum on A_perm (which is logically equivalent to the correct permutation)
    C_ref = torch.einsum("bjik,lk->bijk", A_perm, B)
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel should fail for nonstandard A memory layout."
