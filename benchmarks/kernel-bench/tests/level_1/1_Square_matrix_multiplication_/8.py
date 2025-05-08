
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu; ensure verbose output for debugging.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Only supports float32 inputs.
def test_non_float32_type():
    my_module = build_kernel()
    N = 128
    # Create tensors with double type
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError, match="must be a float32 tensor"):
        _ = my_module.forward(A, B)

# Issue 2: Only supports square matrices.
def test_non_square_matrix():
    my_module = build_kernel()
    # Create non-square matrices, e.g., shape (N, M) where N != M.
    A = torch.randn(128, 64, dtype=torch.float32, device='cuda')
    B = torch.randn(128, 64, dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError, match="must be a square matrix"):
        _ = my_module.forward(A, B)

# Issue 3: Requires contiguous input.
def test_non_contiguous_input():
    my_module = build_kernel()
    N = 128
    # Create a contiguous tensor and then make a non-contiguous view by transposing.
    A = torch.randn(N, N, dtype=torch.float32, device='cuda').t()
    B = torch.randn(N, N, dtype=torch.float32, device='cuda')
    # The CHECK_CONTIGUOUS for A should fail.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _ = my_module.forward(A, B)

# Issue 4: Matrix size not a multiple of TILE_SIZE (32) can be suboptimal.
# Although the kernel handles the boundary condition, this test ensures that
# the correctness is maintained even when N % TILE_SIZE != 0.
def test_non_multiple_of_tile_size():
    my_module = build_kernel()
    # Choose a matrix size not divisible by 32.
    N = 70
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, N, dtype=torch.float32, device='cuda')
    C = my_module.forward(A, B)
    # Synchronize to make sure the kernel has finished
    torch.cuda.synchronize()
    # Compare with PyTorch's built-in matmul result.
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C, C_ref, atol=1e-5), (
        f"Kernel output differs from reference output! Max difference: {abs(C-C_ref).max()}"
    )
