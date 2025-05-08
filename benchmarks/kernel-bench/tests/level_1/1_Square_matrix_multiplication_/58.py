
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Test 1: Trigger error due to input type mismatch (double instead of float32)
def test_input_tensor_type():
    N = 256
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")  # double precision
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be a float32 tensor"):
        kernel_module.forward(A, B)

# Test 2: Trigger error due to non-square matrices (lack of proper dimension check)
def test_non_square_matrices():
    # Create rectangular matrices. The kernel assumes square matrices, so these inputs are invalid.
    A = torch.randn(256, 300, dtype=torch.float32, device="cuda")
    B = torch.randn(256, 300, dtype=torch.float32, device="cuda")
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError, match="Matrices must be square and equal size"):
        kernel_module.forward(A, B)

# Test 3: Check the performance-related behavior (misleading float4 comment)
# While we cannot directly test the absence of float4 coalesced loads, we can compare outputs against torch.matmul.
def test_kernel_correctness_on_small_square():
    N = 64  # small square size that is not a multiple of TILE_DIM (32) to stress boundary handling
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    C_kernel = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_torch = torch.matmul(A, B)
    # Use a loose tolerance in case suboptimal load ordering changes rounding slightly.
    assert torch.allclose(C_kernel, C_torch, atol=1e-4), \
        f"Kernel output differs from torch.matmul reference! Max diff: {(C_kernel - C_torch).abs().max()}"

if __name__ == "__main__":
    pytest.main([__file__])
