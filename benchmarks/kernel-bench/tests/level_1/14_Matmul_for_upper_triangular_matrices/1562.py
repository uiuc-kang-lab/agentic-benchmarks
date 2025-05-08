
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

# Test 1: Pass double tensors to trigger issue 1 (unsupported dtype)
def test_invalid_tensor_type():
    N = 128
    A = torch.triu(torch.randn(N, N, dtype=torch.double, device='cuda'))
    B = torch.triu(torch.randn(N, N, dtype=torch.double, device='cuda'))
    kernel_module = build_kernel()
    # The kernel is compiled for float. We expect either a crash or a wrong result.
    # Here we run and check that the output is not close to expected, indicating failure.
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.triu(torch.matmul(A, B))
    # The result may be incorrect if double tensors are passed.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly produced correct results for double tensors."

# Test 2: Pass non-square matrices to trigger issue 2 (dimension assumptions)
def test_non_square_matrix():
    # Create non-square matrices; even if we use triu the kernel expects square.
    N1, N2 = 128, 256
    A_full = torch.randn(N1, N2, device="cuda", dtype=torch.float32)
    B_full = torch.randn(N1, N2, device="cuda", dtype=torch.float32)
    # Make them upper triangular in the sense of the smallest square (N1 x N1)
    A = torch.triu(A_full[:, :N1])
    B = torch.triu(B_full[:, :N1])
    kernel_module = build_kernel()
    # In a more general situation, the function should either support rectangular inputs
    # or fail gracefully. Our kernel may silently assume the first dimension as N.
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.triu(torch.matmul(A, B))
    # Check that results do not match, indicating that the kernel logic is not handling non-square cases beyond its assumptions.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly computed correct result for non-square matrices."

# Test 3: Pass batched matrices (3D tensor) to trigger issue 2 (lack of batch support)
def test_batched_matrices():
    BATCH, N = 4, 64
    # Create a batched upper triangular matrices. The kernel expects a 2D tensor.
    A = torch.triu(torch.randn(BATCH, N, N, device="cuda", dtype=torch.float32))
    B = torch.triu(torch.randn(BATCH, N, N, device="cuda", dtype=torch.float32))
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Attempting to call the kernel forward with 3D tensors should fail.
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()
