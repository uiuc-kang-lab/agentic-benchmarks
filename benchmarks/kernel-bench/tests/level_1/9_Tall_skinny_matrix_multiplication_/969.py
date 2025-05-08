
import torch
import pytest
from torch.utils.cpp_extension import load

# This helper compiles the kernel from kernel.cu
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# -----------------------------------------------------------------------------
# Issue 1: Unnecessary atomic operations
# (Even though this does not lead to a wrong result, we can see its negative impact by
#  accumulating a matrix product twice if we call the kernel twice on the same output.)
def test_unnecessary_atomicAdd():
    my_module = build_kernel()
    # Use a small matrix so that one block covers the full matrix
    M, K, N = 32, 16, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # First multiplication (should be correct)
    C1 = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C1, C_ref, atol=1e-5), \
        f"Non-transposed result differs! max diff: {(C1-C_ref).abs().max().item()}"

    # Call the kernel again on the same C (which was zeroed by torch.zeros inside kernel)
    # If the atomicAdd were avoided and a direct write used, multiple kernel calls would not
    # erroneously accumulate. Here we simulate a misuse scenario.
    # First run:
    C2 = my_module.forward(A, B)
    # Second run: add in the same output buffer.
    my_module.forward(A, B)
    # Now the second result should be exactly double the reference if the kernel always adds,
    # which is not desired behavior.
    torch.cuda.synchronize()
    # Re-build a proper result by doing two separate multiplications with non-cumulative output.
    expected = C2 * 2.0
    # Run the kernel into a fresh output buffer again by building a new input copy.
    C3 = my_module.forward(A, B)
    my_module.forward(A, B)
    torch.cuda.synchronize()
    assert not torch.allclose(C3, C2, atol=1e-5), "Kernel appears not to be using atomicAdd as expected."
    # Note: This test is somewhat artificial—it triggers an inefficiency by showing that repeated calls
    # accumulate rather than simply writing results.

# -----------------------------------------------------------------------------
# Issue 2: Incorrect transposed input indexing (transposed leading dimensions are mishandled)
def test_transA_incorrect_indexing():
    my_module = build_kernel()
    # Create a tall & skinny view by making A be stored transposed.
    # Branch: if (A_cols > A_rows and B_rows == A_rows) then transA = true is used.
    # Let A be of shape (16, 16384): with A_rows=16, A_cols=16384.
    # The expected multiplication is A^T * B.
    A = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    # To match: B_rows must equal A_rows (16); let B be (16, 32).
    B = torch.randn(16, 32, device="cuda", dtype=torch.float32)
    # The expected result using PyTorch is the multiplication of A^T (shape: 16384 x 16) and B (16 x 32)
    C_ref = torch.matmul(A.t(), B)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Due to the wrong ld used when reading A in transposed mode, the result will differ.
    assert not torch.allclose(C, C_ref, atol=1e-3), \
        f"Transposed A: Kernel output unexpectedly matched the reference (check leading dimension usage)!"

def test_transB_incorrect_indexing():
    my_module = build_kernel()
    # Create a situation to trigger transB: if (A_rows >= A_cols and B_cols == A_cols) then transB = true.
    # Let A be (16384, 16) [A_rows>=A_cols] and B be (32, 16) where B_cols==16.
    A = torch.randn(16384, 16, device="cuda", dtype=torch.float32)
    B = torch.randn(32, 16, device="cuda", dtype=torch.float32)
    # Expected: C = A * B^T  [A: (16384, 16), B^T: (16, 32)] -> result shape (16384, 32)
    C_ref = torch.matmul(A, B.t())
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Due to the wrong ld used when reading B in transposed mode, the result will differ.
    assert not torch.allclose(C, C_ref, atol=1e-3), \
        f"Transposed B: Kernel output unexpectedly matched the reference (check leading dimension usage)!"

# -----------------------------------------------------------------------------
# Issue 3: Kernel does not support non–float32 types
def test_incorrect_dtype():
    my_module = build_kernel()
    M, K, N = 32, 16, 32
    # Create double (float64) inputs even though kernel expects float32.
    A = torch.randn(M, K, device="cuda", dtype=torch.float64)
    B = torch.randn(K, N, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)

# Run the tests when this file is executed directly.
if __name__ == '__main__':
    pytest.main([__file__])
