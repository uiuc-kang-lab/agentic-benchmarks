
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

# Issue 1: The kernel only supports float32 tensors. Passing a double tensor should trigger an error.
def test_input_tensor_type():
    N = 64
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel internally casts to float*.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: The kernel is labeled as exploiting symmetric properties but does not use them.
# Although this is not a correctness bug (it computes the same result as torch.matmul),
# we test with a non-symmetric matrix to expose the misleading name.
def test_symmetric_claim_misleading():
    N = 64
    # Generate non-symmetric matrices (despite name, the kernel treats them as normal matrices)
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # The result should be correct even if it doesn't leverage symmetry,
    # highlighting that the kernel is performing a full matrix multiplication.
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), f"Kernel output differs from reference output!"

# Issue 3: Lack of error checking after kernel launch.
# A potential way to trigger a kernel launch error is to provide input matrices of incompatible sizes,
# which our forward function checks for (e.g., non-square matrices).
def test_incompatible_matrix_sizes():
    N = 64
    # Create a non-square matrix for A and a square matrix for B.
    A = torch.randn(N, N+1, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the forward function should cause an error.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
        
if __name__ == "__main__":
    pytest.main([__file__])
