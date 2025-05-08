
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case for Issue 1: Excessive threads per block causing kernel launch failure.
def test_excessive_threads():
    # These dimensions force the custom kernel path and yield block_size=128,
    # resulting in 128x128 threads per block, which should exceed hardware limits.
    M, K, N = 128, 128, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # This call is expected to produce a CUDA error because of too many threads per block.
        C = mod.forward(A, B)
        torch.cuda.synchronize()

# Test case for Issue 4: Incorrect input tensor type.
def test_wrong_dtype():
    # Passing double tensors (float64) should trigger the CHECK_INPUT failure.
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float64).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float64).contiguous()
    mod = build_kernel()
    with pytest.raises(Exception):  # Could be TORCH_CHECK error
        C = mod.forward(A, B)
        torch.cuda.synchronize()

# Test case for Issue 3: Incorrect cuBLAS parameters leading to a wrong result.
def test_cublas_incorrect_result():
    # Choose dimensions that force the cuBLAS fallback path.
    # For instance any dimension > 128 should call the cuBLAS path.
    M, K, N = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    mod = build_kernel()
    C_kernel = mod.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference using torch.matmul (which is correct)
    C_ref = torch.matmul(A, B)
    # Because of the parameter ordering error in cuBLAS, the results should differ.
    # We expect the maximum absolute difference to be significant.
    err = (C_kernel - C_ref).abs().max().item()
    assert err > 1e-3, f"cuBLAS path appears correct but was expected to be wrong. Max error found: {err}"

# Test case for Issue 5: cuBLAS handle resource leak.
def test_cublas_handle_leak():
    # While we cannot directly test for a resource leak in a short test,
    # repeatedly calling the kernel will eventually stress the handle management.
    # Here, we call the forward function in a loop to simulate repeated use.
    M, K, N = 256, 256, 256
    mod = build_kernel()
    for _ in range(10):
        A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
        B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
        C = mod.forward(A, B)
        torch.cuda.synchronize()
        C_ref = torch.matmul(A, B)
        # In a correct implementation, the result would match (if not for issue 3),
        # but here we are only interested in ensuring that repeated calls do not crash.
        # (We do not check output correctness due to the cuBLAS parameter issue.)
        assert C is not None

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
