
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_tensor_type_must_be_float32():
    # Issue 1: Passing double precision tensors should trigger an error because the kernel expects float32.
    M, K, N = 32, 32, 32
    A = torch.randn(M, K, dtype=torch.double, device="cuda")
    B = torch.randn(K, N, dtype=torch.double, device="cuda")
    mod = build_kernel()
    with pytest.raises(RuntimeError) as exc_info:
        mod.forward(A, B)
    assert "must be a CUDA tensor" not in str(exc_info.value), "Unexpected error message."

def test_non_contiguous_inputs():
    # Issue 2: Non-contiguous inputs should trigger the CHECK_CONTIGUOUS condition.
    M, K, N = 32, 32, 32
    # Create non-contiguous tensors by transposing (which does not make a contiguous tensor).
    A = torch.randn(K, M, device="cuda", dtype=torch.float32).t()  # Now shape (M, K) but non-contiguous
    B = torch.randn(N, K, device="cuda", dtype=torch.float32).t()  # Now shape (K, N) but non-contiguous
    mod = build_kernel()
    with pytest.raises(RuntimeError) as exc_info:
        mod.forward(A, B)
    assert "must be contiguous" in str(exc_info.value)

def test_batched_input_not_supported():
    # Issue 3: Batched inputs (or generally, inputs with more than 2 dimensions)
    # This should trigger a wrong behavior because the kernel assumes 2D tensors.
    batch, M, K, N = 4, 8, 8, 8
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, N, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    # The kernel picks the size(0) and size(1) of A and B respectively assuming 2D tensors.
    # Therefore, the output shape from the custom kernel will not match the batched matmul
    # performed by torch.matmul. We test that the shape is incorrect.
    C_kernel = mod.forward(A, B)
    C_ref = torch.matmul(A, B)
    assert C_kernel.shape != C_ref.shape, "Batched matrix multiplication did not trigger a shape mismatch."

if __name__ == "__main__":
    pytest.main([__file__])
