
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function which compiles and loads the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper reference implementation using PyTorch
def ref_lower_triangular_mm(A, B):
    # Standard matmul of lower triangular matrices then force lower triangular property.
    return torch.tril(torch.matmul(A, B))

# Issue 1: Incorrect tiled loop iteration bounds.
# Use a matrix size such that the summation dimension's tiling does not align with blockIdx assumption.
def test_incorrect_tiled_loop():
    # Use a size not a multiple of TILE_SIZE (e.g. 70) so that tiling for summation is forced to iterate over partial tiles.
    N = 70
    # Create lower triangular matrices float32.
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)

    kernel = build_kernel()
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    C_ref = ref_lower_triangular_mm(A, B)

    # Expect difference due to incorrect tiling loop
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Test did not trigger error in tiled loop iteration bounds (unexpected match)."


# Issue 2: Incorrect shared memory loads due to wrong triangular check.
# Construct matrices where the valid contributions occur from indices that would be skipped by the kernel's condition.
def test_incorrect_shared_memory_load():
    # We'll design a matrix of moderate size where off-diagonal contributions are important.
    N = 64  # multiple of TILE_SIZE, but the way triangular checks are applied is still wrong.
    # Create lower triangular matrices with ones so that the multiplication result can be analytically predicted.
    A = torch.tril(torch.ones(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.ones(N, N, device="cuda", dtype=torch.float32))
    
    kernel = build_kernel()
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    C_ref = ref_lower_triangular_mm(A, B)
    
    # The kernel will zero out some elements that should be nonzero.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Test did not trigger error in shared memory load (triangular check may be working incorrectly)."


# Issue 3: Kernel does not check or handle input tensor types other than float32.
# Provide a double-precision (float64) matrix, expecting the kernel to produce invalid results.
def test_incorrect_input_type():
    N = 32
    # Create lower triangular matrices of type double. The kernel always reads as float.
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    
    kernel = build_kernel()
    # The kernel will use data_ptr<float>() on a double tensor.
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    C_ref = ref_lower_triangular_mm(A.float(), B.float())
    
    # They are unlikely to be close because the memory interpretation will be wrong.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Test did not trigger error for input tensor type mismatch (kernel accepted double input as float)."
        
if __name__ == '__main__':
    pytest.main([__file__])
