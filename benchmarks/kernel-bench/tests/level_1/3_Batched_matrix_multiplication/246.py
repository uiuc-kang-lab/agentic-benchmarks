
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

# Helper to load our CUDA extension
def build_kernel():
    cuda_module = load(
        name="bmm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger issue #1 - using non-float32 (e.g., double) input tensors.
def test_input_tensor_type_error():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    kernel = build_kernel()
    batch_size = 16
    M = 32
    K = 48
    N = 64
    # Create double tensors (float64) instead of float32.
    A = torch.randn(batch_size, M, K, dtype=torch.float64, device="cuda")
    B = torch.randn(batch_size, K, N, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError):
        # This should trigger an error because the kernel only supports float.
        C = kernel.forward(A, B)
        torch.cuda.synchronize()

# Test case 2: Trigger issue #2 - non-contiguous inputs.
def test_non_contiguous_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    kernel = build_kernel()
    batch_size = 8
    M = 40
    K = 50
    N = 60
    A = torch.randn(batch_size, M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(batch_size, K, N, dtype=torch.float32, device="cuda")
    # Make A and B non-contiguous by transposing inner dimensions and transposing them back.
    A_t = A.transpose(1, 2)  # now shape (batch_size, K, M)
    B_t = B.transpose(1, 2)  # now shape (batch_size, N, K)
    A_noncontig = A_t.transpose(1, 2)  # shape restored but non-contiguous memory
    B_noncontig = B_t.transpose(1, 2)
    # Without enforcing contiguity, the kernel may compute wrong results.
    C = kernel.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    # Compute reference using torch.bmm on contiguous tensors.
    C_ref = torch.bmm(A_noncontig.contiguous(), B_noncontig.contiguous())
    # This test assumes the kernel misbehaves on non-contiguous data.
    # We check that the maximum difference is non-zero.
    diff = (C - C_ref).abs().max().item()
    assert diff > 1e-3, f"Expected a significant difference with non-contiguous input, got {diff}"

# Test case 3: Trigger issue #3 - dimensions not divisible by TILE_SIZE.
def test_non_divisible_dimensions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    kernel = build_kernel()
    batch_size = 4
    # Choose dimensions that are not multiples of TILE_SIZE (32)
    M = 45  # not divisible by 32
    K = 50  # not divisible by 32
    N = 37  # not divisible by 32
    A = torch.randn(batch_size, M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(batch_size, K, N, dtype=torch.float32, device="cuda")
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result.
    C_ref = torch.bmm(A, B)
    # Due to edge-case handling, there may be inaccuracies.
    # Here, we expect the maximum error to exceed a tight threshold if there's an issue.
    diff = (C - C_ref).abs().max().item()
    assert diff > 1e-3, f"Expected a significant difference for non-divisible dimensions, got {diff}"
