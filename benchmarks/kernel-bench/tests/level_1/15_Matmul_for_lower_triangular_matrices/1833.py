
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Input type flexibility
# Pass double tensors to trigger the issue (the kernel expects float32)
def test_input_tensor_type():
    N = 64
    A = torch.tril(torch.randn(N, N, dtype=torch.float64, device='cuda'))
    B = torch.tril(torch.randn(N, N, dtype=torch.float64, device='cuda'))
    my_module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
    # We expect an error due to type mismatch during kernel execution.
    assert "CUDA kernel failed" in str(excinfo.value)

# Test 2: Contiguous memory assumption
# Create non-contiguous lower-triangular matrices to trigger wrong results.
def test_non_contiguous_input():
    N = 64
    # Create a contiguous lower-triangular matrix first.
    A = torch.tril(torch.randn(N, N, dtype=torch.float32, device='cuda'))
    B = torch.tril(torch.randn(N, N, dtype=torch.float32, device='cuda'))
    # Make non-contiguous by transposing then transposing back (or selecting a slice)
    A_noncontig = A.t()
    B_noncontig = B.t()
    # They are non-contiguous now:
    assert not A_noncontig.is_contiguous()
    my_module = build_kernel()
    C_kernel = my_module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    # Compute reference from the contiguous tensors
    C_ref = torch.tril(torch.matmul(A_noncontig.contiguous(), B_noncontig.contiguous()))
    # Expect that the kernel result is not equal to the reference due to wrong data layout.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel unexpectedly handled non-contiguous inputs correctly."

# Test 3: Inflexible dimensionality (non-square matrices)
def test_non_square_input():
    N = 64
    M = 32
    # Create a rectangular matrix; note that a lower triangular operation on rectangular tensors is undefined
    A = torch.randn(N, M, dtype=torch.float32, device="cuda")
    B = torch.randn(N, M, dtype=torch.float32, device="cuda")
    my_module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
    assert "must be square" in str(excinfo.value)

# Test 4: Inflexible dimensionality (batched matrices)
def test_batched_input():
    batch = 4
    N = 64
    # Create a batched lower triangular tensor (3D tensor) – kernel expects 2D.
    A = torch.tril(torch.randn(batch, N, N, dtype=torch.float32, device="cuda"))
    B = torch.tril(torch.randn(batch, N, N, dtype=torch.float32, device="cuda"))
    my_module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
    # The error message should complain about A.dim() == 2.
    assert "A must be a 2D tensor" in str(excinfo.value)

# Test 5: Thread divergence and load imbalance
# While we cannot directly “trigger” divergence as an error, we can run the kernel on a problem size
# that exaggerates the irregularity (the triangular region) and check that results are computed correctly.
def test_thread_divergence():
    N = 257  # use a size that is not a multiple of the block dimension so that many threads have fewer iterations
    A = torch.tril(torch.randn(N, N, dtype=torch.float32, device="cuda"))
    B = torch.tril(torch.randn(N, N, dtype=torch.float32, device="cuda"))
    my_module = build_kernel()
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference correctly using torch.matmul and then taking the tril.
    C_ref = torch.tril(torch.matmul(A, B))
    # Even though divergence may hurt performance, the computed result should be correct.
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), f"Kernel output differs from reference output! Max diff: {(C_kernel-C_ref).abs().max()}"
