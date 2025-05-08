
import torch
import pytest
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

# Issue 1: Mis-aligned loads due to the assumption of 128-bit alignment.
# We create tensors with a nonzero storage_offset using as_strided.
def test_misaligned_tensors():
    M, K, N = 128, 256, 64
    # Create normally aligned tensors.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Create misaligned versions by shifting the storage offset by 1 element.
    A_misaligned = A.as_strided(A.size(), A.stride(), storage_offset=1)
    B_misaligned = B.as_strided(B.size(), B.stride(), storage_offset=1)
    kernel = build_kernel()
    # Run kernel multiplication on misaligned tensors.
    C_kernel = kernel.forward(A_misaligned, B_misaligned)
    # Compute reference output using torch.matmul (which handles arbitrary alignment correctly).
    C_ref = torch.matmul(A_misaligned, B_misaligned)
    # We expect a noticeable difference because __ldg() might read misaligned data incorrectly.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel unexpectedly produced correct result with misaligned tensors. "
        "This may hide alignment issues."
    )

# Issue 2: The kernel enforces contiguous memory.
def test_noncontiguous_tensors():
    M, K, N = 128, 256, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Create non-contiguous tensors by transposing.
    A_noncontig = A.t()
    B_noncontig = B.t()
    kernel = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _ = kernel.forward(A_noncontig, B_noncontig)

# Issue 3: The kernel does not verify that dimensions match.
def test_dimension_mismatch():
    # Create A of shape (M, K) and B of shape (K+1, N) to trigger a mismatch.
    M, K, N = 128, 64, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_wrong = torch.randn(K + 1, N, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # Since the inner dimensions do not match, the kernel may perform out-of-bound accesses,
    # so we expect either a RuntimeError or an incorrect result.
    with pytest.raises(RuntimeError):
        _ = kernel.forward(A, B_wrong)

# Issue 4: The kernel launcher does not check the return value of cudaDeviceSynchronize().
# A common way to force a kernel launch error is to pass non-CUDA tensors.
def test_cuda_tensor_requirement():
    M, K, N = 128, 256, 64
    A = torch.randn(M, K, device="cpu", dtype=torch.float32)
    B = torch.randn(K, N, device="cpu", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        _ = kernel.forward(A, B)
        
if __name__ == "__main__":
    pytest.main([__file__])
