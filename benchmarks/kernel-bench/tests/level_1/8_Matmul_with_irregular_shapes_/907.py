
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_inner_dimensions():
    """
    This test triggers the issue where the kernel does not check that
    the inner dimensions of A and B are compatible. We deliberately
    create a mismatch such that A.size(1) != B.size(0).
    Expected behavior: The out-of-bound memory access should cause a runtime error.
    """
    # Create A with shape (M, K) and B with mismatching inner dimension: (K_wrong, N)
    M, K, K_wrong, N = 100, 64, 32, 80  # K != K_wrong
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K_wrong, N, device='cuda', dtype=torch.float32)
    
    my_module = build_kernel()
    # Since the kernel uses A.size(1) (i.e. 64) for iteration over B's rows,
    # out-of-bound access will occur if B has only 32 rows.
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_dtype():
    """
    This test triggers the issue where the kernel is hardcoded for float32,
    but a tensor of a different dtype (float64) is passed.
    
    Expected behavior: The kernel will reinterpret the double data as float,
    resulting in an output that does not match the correct result.
    """
    N = 128
    # Create two double precision matrices on CUDA.
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    
    my_module = build_kernel()
    # Run the kernel
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute the reference result using torch.matmul (with proper dtype)
    C_ref = torch.matmul(A, B)
    
    # Because the kernel mistakenly processes double arrays as float arrays,
    # the result will be significantly different.
    # We expect that the two results are not nearly equal.
    if torch.allclose(C_kernel, C_ref, atol=1e-4):
        pytest.fail("Kernel output matches reference output despite incorrect dtype usage.")
