
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

@pytest.fixture(scope="module")
def cuda_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return build_kernel()

# Issue 1: Kernel only supports square matrices.
def test_non_square_matrix(cuda_module):
    # Create non-square matrices
    A = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    B = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError) as excinfo:
        cuda_module.forward(A, B)
    # Check error message contains "square"
    assert "square" in str(excinfo.value)

# Issue 2: Forced unrolling leading to potential problems when number of tiles is < 4.
def test_non_multiple_of_tile_size(cuda_module):
    # Choose a matrix size that is not a multiple of TILE_SIZE (16)
    # This forces the kernel to take the edge path where the number of tiles is not a multiple of 4.
    N = 33  # 33 is not a multiple of 16 so the tile loop iterates only 3 times
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Tolerance might be different on edge cases.
    assert torch.allclose(C, C_ref, atol=1e-5), \
        f"Kernel output differs from torch.matmul! Max diff: {(C - C_ref).abs().max().item()}"

# Issue 3: Fixed TILE_SIZE limits flexibility. For a much larger matrix the fixed size might hurt performance
# (here we check correctness rather than performance, yet it exposes inflexibility in handling diverse sizes)
def test_large_matrix_fixed_tile_size(cuda_module):
    N = 1024  # a larger matrix to show that kernel is not adapting TILE_SIZE.
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C, C_ref, atol=1e-5), \
        f"Kernel output differs from torch.matmul! Max diff: {(C - C_ref).abs().max().item()}"

# Issue 4: Kernel accepts only float32; other dtypes should result in an error.
def test_incorrect_dtype(cuda_module):
    N = 64
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)  # double precision
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError) as excinfo:
        cuda_module.forward(A, B)
    # Check whether the error message mentions float32
    assert "float32" in str(excinfo.value)
