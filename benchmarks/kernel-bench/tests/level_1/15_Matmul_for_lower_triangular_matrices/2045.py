
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # This will compile the kernel.cu file.
    cuda_module = load(
        name="test_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Early exit condition not handling vectorized columns correctly.
# We design a case where some threads in the same vector load should
# compute non-zero lower-triangular results, but the early exit could zero them.
def test_early_exit_condition():
    N = 35  # Not a multiple of TILE_SIZE or VECTOR_SIZE; chosen to force boundary issues.
    # Create lower triangular matrices.
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    
    # Expectation computed on CPU (or Torch, using lower-triangular post processing)
    C_ref = torch.tril(torch.matmul(A, B))
    
    # Build and run kernel.
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    # The output is likely to be wrong if early exit condition mishandles vector lanes.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Error: Kernel early exit condition did not trigger any error!"

# Issue 2: Dimension assumptions (matrix sizes not divisible by 4 or 32)
def test_non_divisible_dimensions():
    N = 70  # arbitrary number not divisible by 4 or 32
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    
    C_ref = torch.tril(torch.matmul(A, B))
    
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    # The kernel may use out-of-bound memory accesses or compute wrong results.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Error: Kernel did not exhibit errors on non-divisible dimensions!"

# Issue 3: Non-contiguous tensor input
def test_non_contiguous_input():
    N = 64  # multiple of 32 to isolate non-contiguity as the issue
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    
    # Make the input non-contiguous by transposing (and then transposing back to preserve shape)
    A_noncontig = A.t().clone().t()  # force non-contiguity
    B_noncontig = B.t().clone().t()
    
    my_module = build_kernel()
    # Likely the kernel will compute wrong result because of unexpected strides.
    C = my_module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    
    C_ref = torch.tril(torch.matmul(A_noncontig, B_noncontig))
    assert not torch.allclose(C, C_ref, atol=1e-4), "Error: Kernel did not fail for non-contiguous inputs!"

# Issue 4: Data type assumption: passing double precision should not work.
def test_incorrect_datatype():
    N = 64
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(A, B)
    # An exception should be thrown since the kernel only supports float32.

# Issue 5: Alignment/Vectorized load misalignment.
# We trigger this by shifting the starting column via as_strided.
def test_misaligned_access():
    N = 64
    full_A = torch.tril(torch.randn(N+1, N+1, device="cuda", dtype=torch.float32))
    full_B = torch.tril(torch.randn(N+1, N+1, device="cuda", dtype=torch.float32))
    # Create a misaligned view by taking a sub-tensor starting from offset 1 in both dims.
    A_misaligned = full_A[1:, 1:][:N, :N]
    B_misaligned = full_B[1:, 1:][:N, :N]
    
    C_ref = torch.tril(torch.matmul(A_misaligned, B_misaligned))
    
    my_module = build_kernel()
    C = my_module.forward(A_misaligned, B_misaligned)
    torch.cuda.synchronize()
    
    # The kernel vectorized loads may be misaligned and produce wrong results.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Error: Kernel did not exhibit misalignment-related errors!"

