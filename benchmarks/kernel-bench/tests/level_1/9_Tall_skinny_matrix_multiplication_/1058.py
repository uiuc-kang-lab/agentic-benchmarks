
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper to compile the CUDA extension.
def build_kernel(extra_cuda_cflags=None):
    flags = ["-O3", "--use_fast_math"]
    if extra_cuda_cflags is not None:
        flags.extend(extra_cuda_cflags)
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=flags,
        verbose=True,
    )
    return cuda_module

# Test case for Issue 1: Unrolling assumes BLOCK_SIZE is divisible by 4.
# We force a non-divisible BLOCK_SIZE (e.g., 15) via a preprocessor flag.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_unroll_loop_nondivisible_bs():
    # Build the kernel with BLOCK_SIZE set to 15 (not divisible by 4)
    my_module = build_kernel(extra_cuda_cflags=["-DBLOCK_SIZE=15"])
    
    # Create matrices with dimensions that make sense:
    # Let M and N be multiples of BLOCK_SIZE for grid, and K chosen arbitrarily.
    M = 128
    N = 128
    K = 100  # arbitrary K
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    
    # Run the kernel. Because of the unrolling issue, we expect an incorrect result.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    
    # The results should differ beyond tolerance
    if torch.allclose(C, C_ref, atol=1e-5):
        pytest.fail("Kernel returned correct result with a non-divisible BLOCK_SIZE, "
                    "but it is expected to fail due to unrolling assumptions.")

# Test case for Issue 2: Input tensor dtype is not float32.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_input_tensor_wrong_dtype():
    my_module = build_kernel()
    
    M = 64
    N = 64
    K = 32
    # Create double precision inputs
    A = torch.randn(M, K, device="cuda", dtype=torch.float64).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float64).contiguous()
    
    with pytest.raises(Exception):
        # The kernel expects float tensors and should throw an error or produce a wrong result.
        # We treat any exception as a triggered issue.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test case for Issue 3: Input tensors are not contiguous.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_input_tensor_noncontiguous():
    my_module = build_kernel()
    
    M = 64
    N = 32
    K = 48
    # Create a tensor and then take a transpose (which makes it non-contiguous)
    A_base = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B_base = torch.randn(N, K, device="cuda", dtype=torch.float32)
    
    # According to the logic in matmul_cuda, one of the matrices is expected to be transposed.
    # Here we deliberately pass noncontiguous tensors.
    A = A_base.t()  # shape becomes (M, K) but non-contiguous
    B = B_base.t()  # shape becomes (K, N) but non-contiguous
    
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.contiguous(), B.contiguous())
    
    if torch.allclose(C, C_ref, atol=1e-5):
        pytest.fail("Kernel produced correct result with non-contiguous inputs, "
                    "but it is expected to malfunction due to unhandled tensor layout.")

# Test case for Issue 4: Lack of kernel error checking.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_kernel_error_checking():
    my_module = build_kernel()
    
    # Intentionally pass a tensor on CPU to trigger the "Inputs must be CUDA tensors" error.
    M = 32
    N = 32
    K = 16
    A = torch.randn(M, K, device="cpu", dtype=torch.float32)
    B = torch.randn(K, N, device="cpu", dtype=torch.float32)
    
    with pytest.raises(Exception):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
