
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

# Issue 1: The kernel assumes square matrices. This test passes non-square matrices to trigger the error.
def test_non_square_matrix():
    my_module = build_kernel()
    # Create non-square tensors, e.g., 32x64 matrices. The kernel expects square matrices so its TORCH_CHECK should fail.
    A = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    B = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must be a square matrix"):
        _ = my_module.forward(A, B)

# Issue 2: The kernel only supports float32. This test uses a different dtype to trigger the error.
def test_non_float32_input():
    my_module = build_kernel()
    N = 128
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)  # double precision
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="must be a float32 tensor"):
        _ = my_module.forward(A, B)

# Issue 3: The kernel requires inputs to be contiguous. This test passes non-contiguous inputs to trigger the error.
def test_non_contiguous_input():
    my_module = build_kernel()
    N = 128
    # Create a contiguous tensor and then take a transpose to make it non-contiguous.
    A_cont = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B_cont = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = A_cont.t()  # non-contiguous version
    B = B_cont.t()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _ = my_module.forward(A, B)

if __name__ == "__main__":
    pytest.main([__file__])
