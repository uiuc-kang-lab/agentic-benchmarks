
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from the provided kernel.cu file.
    module = load(
        name="triangular_mm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Test 1: Trigger issue with tensor datatype.
# The kernel expects float32, so passing double tensors should cause incorrect behavior.
def test_dtype(kernel_module):
    N = 64
    # Create lower triangular matrices of type double, which is not supported by kernel.
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    A = torch.tril(A)
    B = torch.tril(B)
    with pytest.raises(RuntimeError):
        # We expect the kernel launch or subsequent computation to fail or produce wrong result.
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()

# Test 2: Trigger issue with non-contiguous inputs.
def test_noncontiguous(kernel_module):
    N = 64
    # Create contiguous lower triangular matrix and then make it non-contiguous by transposing.
    A_contig = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B_contig = torch.randn(N, N, device='cuda', dtype=torch.float32)
    A = torch.tril(A_contig).t()  # Transpose makes it non-contiguous even though triangular structure is preserved.
    B = torch.tril(B_contig).t()
    assert not A.is_contiguous() or not B.is_contiguous(), "Test failed: tensors are contiguous."
    with pytest.raises(RuntimeError):
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()

# Test 3: Trigger issue with matrix dimensions not divisible by TILE_SIZE.
def test_nondivisible_dimension(kernel_module):
    # Use a matrix size that is not a multiple of the TILE_SIZE (which is 32).
    N = 70  # 70 is not divisible by 32.
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    # Force lower triangular structure.
    A = torch.tril(A)
    B = torch.tril(B)
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()

    # Compute reference result using PyTorch. Note: the original PyTorch method applies tril after multiplication.
    C_ref = torch.matmul(A, B)
    # Because the CUDA kernel masks contribution using the condition (k between col and row) 
    # it should mimic lower triangular multiplication.
    # However, if there is an issue with boundaries, the result may differ.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel unexpectedly produced a result matching full matmul for nondivisible dimensions, " \
        "which may indicate that the handling of tile boundaries or diagonal conditions is not general."

if __name__ == "__main__":
    pytest.main([__file__])
