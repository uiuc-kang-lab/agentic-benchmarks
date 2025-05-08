
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension (this compiles kernel.cu)
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that non-float inputs are rejected.
def test_dtype_error():
    my_module = build_kernel()
    # Create double tensors (non-float32) on CUDA.
    A = torch.randn(32, 32, dtype=torch.double, device="cuda")
    B = torch.randn(32, 32, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError, match="must be a float tensor"):
        my_module.forward(A, B)

# Issue 1 (non-contiguous): Test that non-contiguous inputs are rejected.
def test_noncontiguous_error():
    my_module = build_kernel()
    A = torch.randn(64, 64, dtype=torch.float32, device="cuda").t()  # transpose makes it non–contiguous
    B = torch.randn(64, 64, dtype=torch.float32, device="cuda").t()  # non–contiguous as well
    with pytest.raises(RuntimeError, match="must be contiguous"):
        my_module.forward(A, B)

# Issue 2: Test unrolling assumption.
# If TILE_SIZE is assumed to be a multiple of 8, then using a problem whose inner dimension is not a multiple of 8 
# in the custom kernel path may lead to incorrect results.
# We trigger the custom kernel (by making the total operations below CUBLAS_THRESHOLD) and use dimensions that are not multiples of 8.
def test_unrolling_tile_issue():
    my_module = build_kernel()
    # Use dimensions that force custom kernel launch:
    # Use a K value that is not a multiple of 8.
    M, K, N = 17, 14, 19  # total operations = 17*14*19 = 4522 < 100000 threshold
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # With an unrolling bug the results can be quite different.
    assert torch.allclose(C, C_ref, atol=1e-4), f"Custom kernel (unrolling issue) output differs from torch.matmul!"

# Issue 3: Test potential cuBLAS mismatch (transposed output).
# We force the cuBLAS path by using a large operation count.
def test_cublas_transpose_issue():
    my_module = build_kernel()
    # Choose dimensions that guarantee (M*K*N) > CUBLAS_THRESHOLD.
    # Also choose non-square shapes so that any unintended transposition is evident.
    M, K, N = 128, 256, 64  # total operations = 128*256*64 = 2,097,152 > 100000
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # In case the output is accidentally transposed, the maximum difference will be large.
    assert torch.allclose(C, C_ref, atol=1e-4), f"cuBLAS path may have computed a transposed output! Max diff: {(C-C_ref).abs().max().item()}"

# Issue 4: Test kernel launch error checking.
# We force an error by intentionally passing in an incorrectly shaped tensor.
def test_kernel_launch_error():
    my_module = build_kernel()
    # Create A and B where inner dimensions do not match.
    A = torch.randn(32, 10, dtype=torch.float32, device="cuda")
    B = torch.randn(32, 32, dtype=torch.float32, device="cuda")  # Mismatch: A.shape[1] != B.shape[0]
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)
