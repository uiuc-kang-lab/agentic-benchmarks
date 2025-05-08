import torch
from torch.utils.cpp_extension import load
import sys

def test_custom_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernels/level_1/1_Square_matrix_multiplication_/6.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

    # Set N such that N % TILE_SIZE != 0, e.g., 70 if TILE_SIZE is 32.
    N = 70
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)

    # Try to run your kernel
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)


    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from reference output! Difference: {(C-C_ref).abs().max()}"

if __name__ == "__main__":
    test_custom_kernel()