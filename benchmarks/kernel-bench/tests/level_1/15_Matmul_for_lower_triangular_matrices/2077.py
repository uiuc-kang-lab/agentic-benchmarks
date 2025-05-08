
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper to build/load the CUDA extension (assuming kernel.cu is present in the working directory)
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue by passing non-float tensors (e.g., double) which should fail due to wrong dtype.
def test_non_float_dtype():
    my_module = build_kernel()
    N = 128
    # Create double tensors on CUDA
    A = torch.randn(N, N, device="cuda", dtype=torch.double)
    B = torch.randn(N, N, device="cuda", dtype=torch.double)
    A = torch.tril(A)
    B = torch.tril(B)
    with pytest.raises(RuntimeError):
        # This should raise an error or produce wrong result, so we check for exception.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 2: Trigger issue with non-contiguous tensors.
def test_non_contiguous_tensors():
    my_module = build_kernel()
    N = 128
    # Create contiguous triangular matrices.
    A_full = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B_full = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A_full = torch.tril(A_full)
    B_full = torch.tril(B_full)
    # Make non-contiguous by transposing (transpose returns non-contiguous tensor) and then triling again.
    A = torch.tril(A_full.t())
    B = torch.tril(B_full.t())
    # If the kernel is launched on non-contiguous tensor pointers, the result may be wrong.
    # Compare the kernel result against torch.matmul followed by tril on a contiguous version.
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference, but ensure inputs are made contiguous.
    A_contig = A.contiguous()
    B_contig = B.contiguous()
    C_ref = torch.tril(torch.matmul(A_contig, B_contig))
    # We expect the result to be different due to wrong memory layout usage.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel should fail with non-contiguous inputs but it matched contiguous reference."

# Test 3: Test potential issues due to loop unrolling by choosing matrix sizes such that
# the inner loop iteration count (row - col + 1) is not a multiple of the unroll factor (4).
def test_loop_unrolling_edge_case():
    my_module = build_kernel()
    # Choose a small N to force many iterations where (row - col + 1) mod 4 != 0.
    N = 35
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    # Ensure that A and B are lower triangular.
    A = torch.tril(A)
    B = torch.tril(B)
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Reference computation using PyTorch matmul followed by tril filtering.
    C_ref = torch.tril(torch.matmul(A, B))
    # While the kernel might compute a result, we are testing that potential unroll issues cause a mismatch.
    # Depending on how the compiler treats #pragma unroll with non-constant bounds, this could be an issue.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel output unexpectedly matches reference despite potential loop unrolling issues."
    
if __name__ == "__main__":
    pytest.main([__file__])
