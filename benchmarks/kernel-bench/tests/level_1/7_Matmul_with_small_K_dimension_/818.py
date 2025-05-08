
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

def test_dtype_issue():
    # Issue 1: Passing tensors with dtype torch.double should raise a runtime error
    mod = build_kernel()
    M, K, N = 128, 32, 128
    A = torch.randn(M, K, dtype=torch.double, device="cuda")
    B = torch.randn(K, N, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError):
        # The kernel expects float32 so this call is likely to result in memory corruption or an error.
        C = mod.forward(A, B)
        torch.cuda.synchronize()

def test_dimension_mismatch_issue():
    # Issue 2: The kernel does not check for inner dimension compatibility.
    # PyTorch’s native matmul will raise an error for mismatched dimensions,
    # while our custom kernel will compute something (likely wrong) without bounds checking.
    mod = build_kernel()
    # Create mismatched tensors:
    # A is (M, K) and B is (K+1, N)
    M, K, N = 128, 32, 128
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K + 1, N, dtype=torch.float32, device="cuda")
    
    # torch.matmul should raise an error due to incompatible dimensions.
    with pytest.raises(RuntimeError):
        _ = torch.matmul(A, B)
    
    # However, our custom kernel will still produce an output with shape (M, N)
    # even though the dimensions are mismatched.
    C_custom = mod.forward(A, B)
    torch.cuda.synchronize()
    # The output shape is determined by A.size(0) and B.size(1)
    assert C_custom.shape == (M, N), (
        f"Expected output shape ({M}, {N}), got {C_custom.shape}"
    )
