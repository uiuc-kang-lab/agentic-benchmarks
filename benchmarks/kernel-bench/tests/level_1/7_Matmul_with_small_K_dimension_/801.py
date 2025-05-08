
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

# Test 1: Using an unsupported data type (float64) should trigger issues.
def test_input_tensor_type():
    # Create double precision (float64) tensors.
    M, K, N = 128, 64, 128
    A = torch.randn(M, K, dtype=torch.float64, device='cuda')
    B = torch.randn(K, N, dtype=torch.float64, device='cuda')
    mod = build_kernel()
    # Expecting that using a float64 tensor will either raise an error or produce wrong results.
    with pytest.raises(RuntimeError):
        _ = mod.forward(A, B)

# Test 2: Batched matrix multiplication is not supported.
def test_batched_input():
    # Create batched matrices (3D tensors)
    batch = 4
    M, K, N = 128, 64, 128
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, N, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    # The kernel expects 2D input, so we simulate a mis-use.
    # We pass the first batch element and compare to the reference.
    C_kernel = mod.forward(A[0], B[0])
    C_ref = torch.matmul(A[0], B[0])
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel did not compute correct result for batched input extraction. Batched case not supported!"

# Test 3: Non-contiguous input tensors.
def test_non_contiguous_input():
    # Create contiguous input and then make a non-contiguous view by transposing.
    M, K, N = 128, 64, 128
    A_base = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_base = torch.randn(K, N, device="cuda", dtype=torch.float32)
    A_noncontig = A_base.t()  # Transposed makes it non-contiguous.
    B_noncontig = B_base.t()
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        _ = mod.forward(A_noncontig, B_noncontig)
        
# Note: Test 3 relies on the TORCH_CHECK in the extension that enforce contiguity.

if __name__ == "__main__":
    pytest.main([__file__])
