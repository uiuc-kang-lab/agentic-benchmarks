
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Incorrect thread mapping (non-standard grid/block mapping)
def test_incorrect_thread_mapping():
    # Using a matrix size that is not a multiple of the block tile dimensions
    N = 70  # 70 is not a multiple of 16, so the nonstandard mapping will likely trigger index errors.
    A = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    ref = torch.triu(torch.matmul(A, B))
    
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Here we expect the kernel result to be wrong because of incorrect thread mapping.
    assert not torch.allclose(C, ref, atol=1e-5), (
        "Test failed to trigger incorrect thread mapping: kernel output matched the reference unexpectedly."
    )

# Issue 2: Lack of type checking on input tensors (only float32 is supported)
def test_input_tensor_type():
    N = 256
    # Create double tensors instead of float (float64 instead of float32)
    A = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float64))
    B = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float64))
    
    my_module = build_kernel()
    # Expecting a wrong result (or possibly a silent error) when using an unsupported type.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute reference multiplication (after converting A and B into float for comparison)
    ref = torch.triu(torch.matmul(A.float(), B.float()))
    # The kernel internally treats the bits as float32 so the output will be off.
    assert not torch.allclose(C, ref, atol=1e-5), (
        "Kernel did not trigger an issue when provided with double precision inputs."
    )

# Issue 3: Assumption of contiguous and square matrices.
def test_non_contiguous_input():
    N = 256
    # Create a triangular matrix and then make it noncontiguous by transposing it.
    A_full = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B_full = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    
    # Make non-contiguous views. For instance, transposing (which for a triangular matrix breaks the triangular structure)
    A_nc = A_full.t()  # now noncontiguous and not upper triangular anymore
    B_nc = B_full.t()
    
    my_module = build_kernel()
    # The kernel will use the raw data pointer assuming contiguous, so the result will be incorrect.
    C = my_module.forward(A_nc, B_nc)
    torch.cuda.synchronize()
    
    # Reference using torch.matmul on the noncontiguous inputs (and then taking upper triangulation)
    ref = torch.triu(torch.matmul(A_nc, B_nc))
    assert not torch.allclose(C, ref, atol=1e-5), (
        "Kernel failed to trigger an issue on noncontiguous inputs."
    )

def test_non_square_input():
    # The kernel is hard-coded for square matrices. Here we pass non-square matrices.
    # For simplicity, we mimic a non-square scenario by using different sizes.
    # Note: Since the kernel uses A.size(0) for N, passing non-square tensors leads to out-of-bound accesses.
    # We expect this to cause a runtime error.
    N = 256
    M = 128  # Non-square: A is (N,N) but B will be crafted to be (N,N) even though logically one would require (N,M).
    # To simulate the issue, we create A as a triangular matrix of size (N, N) and B as a triangular
    # matrix of size (N, M) (by padding B with zeros later) but the kernel does not support this.
    A = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    # Create a (N, M) tensor and then pad it to (N, N), but the non-square aspect is hidden.
    # Instead, we force a non-square scenario by directly manipulating shapes:
    B = torch.triu(torch.randn(N, M, device='cuda', dtype=torch.float32))
    
    my_module = build_kernel()
    # Since the kernel reads up to index N in each row/col, passing a non-square B will likely trigger a CUDA error.
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
