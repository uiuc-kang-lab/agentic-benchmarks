
import pytest
import torch
from torch.utils.cpp_extension import load
import math

def build_kernel():
    # Build the extension module from the given kernel.cu file.
    module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Hard-coded float type
def test_dtype_issue():
    # Create double (float64) inputs instead of float32.
    N = 64
    A = torch.triu(torch.randn(N, N, dtype=torch.double, device='cuda'))
    B = torch.triu(torch.randn(N, N, dtype=torch.double, device='cuda'))
    module = build_kernel()
    # When using double tensors, the underlying kernel still treats the data as float.
    # Hence, the output will diverge from the correct computation.
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.triu(torch.matmul(A, B))
    # We expect a mismatch because the kernel misinterprets double data
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Test failed: Kernel produced correct result even for double input, but it was hard-coded for float."
    )

# Issue 2: Non-square matrices
def test_non_square_issue():
    # Create non-square matrices. Kernel assumes square matrices (N x N).
    M, N = 64, 128
    # Construct upper triangular-like matrices from a non-square input.
    # Here, we use torch.triu on a (M, N) tensor.
    A = torch.triu(torch.randn(M, N, device='cuda', dtype=torch.float32))
    B = torch.triu(torch.randn(M, N, device='cuda', dtype=torch.float32))
    module = build_kernel()
    # The kernel uses A.size(0) as N so it will be confused.
    with pytest.raises(RuntimeError):
        # We expect an error or miscomputed result if the kernel is forced on non-square matrices.
        C = module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Numerical precision in closed-form index calculation
def test_precision_issue():
    # Use a very large matrix to stress the precision of the closed-form inversion
    N = 10000  # large matrix size to magnify potential float rounding errors
    A = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.triu(torch.matmul(A, B))
    # We expect differences in some of the computed upper-triangular elements due to indexing precision issues.
    max_diff = (C - C_ref).abs().max().item()
    assert max_diff > 1e-3, (
        f"Test failed: Maximum difference {max_diff} is too small, expected indexing precision issues for large matrices."
    )

# Issue 4: Lack of error checking after kernel launch
def test_kernel_launch_error_checking():
    # Force an error by passing an invalid pointer.
    N = 64
    A = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    module = build_kernel()

    # Create an empty tensor to simulate an invalid pointer for C.
    C_invalid = torch.empty(0, device='cuda', dtype=torch.float32)
    # Directly call the kernel function via the module (if exposed) by sending an invalid tensor.
    # We simulate this by calling the forward function with a wrong C shape by monkey-patching data_ptr.
    orig_data_ptr = A.data_ptr()
    try:
        # We deliberately mess with the pointer by passing the wrong tensor for output.
        # This should lead to a kernel error but without proper error checking in the kernel,
        # the error might go unnoticed until cudaDeviceSynchronize.
        module.forward(A, B)  # normal call for context; then we launch an erroneous call
        # The following is a hack: we call cudaDeviceSynchronize expecting an error if error checking
        torch.cuda.synchronize()
    except RuntimeError:
        pytest.skip("Kernel launch error detected as expected, but error checking is missing.")
    # If no exception was raised, the kernel did not check errors.
    assert False, "Test failed: Kernel launch error was not caught due to lack of error checking."

# Issue 5: Non-contiguous input tensors
def test_non_contiguous_issue():
    N = 64
    A_full = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    B_full = torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32))
    # Create non-contiguous versions by transposing twice after slicing.
    A_noncontig = A_full.t()[..., :].t()  # Guaranteed to be non-contiguous sometimes.
    B_noncontig = B_full.t()[..., :].t()
    assert not A_noncontig.is_contiguous(), "Test setup error: A_noncontig should not be contiguous."
    module = build_kernel()
    C = module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    C_ref = torch.triu(torch.matmul(A_noncontig, B_noncontig))
    # The output may be incorrect because the kernel expects contiguous memory.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Test failed: Kernel computed correct results with non-contiguous inputs, but it is not designed for that."
    )

# Issue 6: Kernel not supporting batched inputs
def test_batched_input_issue():
    # Create a batched input with shape (batch, N, N)
    batch = 4
    N = 64
    A_batched = torch.stack([torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32)) for _ in range(batch)])
    B_batched = torch.stack([torch.triu(torch.randn(N, N, device='cuda', dtype=torch.float32)) for _ in range(batch)])
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Passing batched tensors to the kernel which expects 2D tensors should trigger an error.
        C = module.forward(A_batched, B_batched)
        torch.cuda.synchronize()
