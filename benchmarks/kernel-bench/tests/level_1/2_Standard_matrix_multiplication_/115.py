
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the extension from kernel.cu.
    cuda_module = load(
        name="hybrid_mm",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

def test_input_tensor_type():
    """
    Test 1: Pass non-float32 (double) tensors to trigger the type assumption issue.
    Expect an exception from our CHECK_INPUT macro.
    """
    cuda_module = build_kernel()
    M, K, N = 32, 32, 32
    A = torch.randn(M, K, dtype=torch.double, device="cuda")
    B = torch.randn(K, N, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(A, B)

def test_large_matrix_cublas_incorrect_result():
    """
    Test 2: Use large matrices to force the cuBLAS branch.
    If the call parameters for row-major conversion are incorrect, then the result will differ from torch.matmul.
    """
    cuda_module = build_kernel()
    # Choose dimensions larger than 128 to force the cuBLAS path.
    M, K, N = 256, 256, 256
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    C_kernel = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # If the cuBLAS hack is implemented correctly this should pass.
    # If not, there will be a significant difference.
    assert torch.allclose(C_kernel, C_ref, atol=1e-3), (
        f"cuBLAS-based kernel result does not match torch.matmul! Max diff: {(C_kernel - C_ref).abs().max()}"
    )

def test_kernel_error_checking():
    """
    Test 3: Intentionally create an invalid scenario that should cause the kernel launch or cuBLAS call to fail.
    For instance, passing mismatched dimensions.
    Expect the CHECK_INPUT macros or cuBLAS error reporting to raise an exception.
    """
    cuda_module = build_kernel()
    # Create mismatched matrix dimensions (A: MxK, B: K2xN) so that multiplication is impossible.
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K + 1, N, dtype=torch.float32, device="cuda")  # Wrong inner dimension.
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(A, B)

def test_cublas_handle_resource_leak():
    """
    Test 4: Repeated calls to the kernel to simulate long runtime.
    Although we cannot exactly check for resource leaks in a pytest,
    we can at least call the function repeatedly and verify it produces correct results.
    
    (Note: In a real scenario, lacking proper destruction of the cuBLAS handle could cause problems 
    upon module reloads or shutdown.)
    """
    cuda_module = build_kernel()
    M, K, N = 256, 256, 256  # Use large matrices to force cuBLAS branch.
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    for _ in range(10):
        C_kernel = cuda_module.forward(A, B)
        torch.cuda.synchronize()
        C_ref = torch.matmul(A, B)
        assert torch.allclose(C_kernel, C_ref, atol=1e-3), (
            f"Repeated call error: max diff {(C_kernel - C_ref).abs().max()}"
        )
