
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension module from kernel.cu.
# Note: The sources list should include the path to kernel.cu.
def build_kernel():
    cuda_module = load(
        name="custom_einsum_cuda",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test non-contiguous input tensors.
# The kernel assumes contiguous tensors; using non-contiguous versions should yield incorrect results.
def test_noncontiguous_input():
    # Dimensions as in sample code.
    b, i, j, l, k = 4, 16, 32, 8, 20
    
    # Create contiguous tensors.
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)

    # Create a non-contiguous version of A by transposing two dimensions and then transposing back.
    A_noncontig = A.transpose(1, 2)
    # Now transpose again to recover shape but non-contiguity remains.
    A_noncontig = A_noncontig.transpose(1, 2)
    assert not A_noncontig.is_contiguous(), "A_noncontig should be non contiguous."
    
    my_module = build_kernel()
    # The correct result computed via einsum.
    C_correct = torch.einsum("bijl,lk->bijk", A_noncontig, B)
    
    # The kernel uses data_ptr without handling strides, so the result will likely be wrong.
    C_kernel = my_module.forward(A_noncontig, B)
    torch.cuda.synchronize()
    
    # The difference should be large so we expect the assertion to fail.
    diff = (C_correct - C_kernel).abs().max().item()
    assert diff > 1e-3, f"Kernel unexpectedly produced correct result for non-contiguous input, diff={diff}"

# Issue 2: Test input tensor data type mismatch.
# The kernel hard-codes float so using double inputs might lead to wrong results or a crash.
def test_wrong_dtype():
    b, i, j, l, k = 4, 16, 32, 8, 20
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float64)
    B = torch.randn(l, k, device="cuda", dtype=torch.float64)
    my_module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # As the kernel expects float32, this should trigger an error (or produce wrong memory accesses).
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Test for unsupported memory layout (non-standard strides).
# For example, if B is not contiguous because it has been transposed,
# the kernel will access wrong data.
def test_noncontiguous_B():
    b, i, j, l, k = 4, 16, 32, 8, 20
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    
    # Make B non-contiguous by transposing.
    B_noncontig = B.t()
    # Note: B_noncontig has shape (k, l) now; we need to revert the shape while remaining non-contiguous.
    B_noncontig = B_noncontig.t()  # This recovers original shape but B_noncontig is still non-contiguous.
    assert not B_noncontig.is_contiguous(), "B_noncontig should be non contiguous."

    my_module = build_kernel()
    C_correct = torch.einsum("bijl,lk->bijk", A, B_noncontig)
    C_kernel = my_module.forward(A, B_noncontig)
    torch.cuda.synchronize()
    
    diff = (C_correct - C_kernel).abs().max().item()
    # Expect a significant difference due to wrong memory addressing in B.
    assert diff > 1e-3, f"Kernel unexpectedly produced correct result for non-contiguous B, diff={diff}"

# Issue 4: Test the behavior with CUDA streams.
# Although the kernel creates an extra stream (stream2) that is never used,
# the kernel should behave identically to torch.einsum for supported inputs.
def test_stream_usage():
    b, i, j, l, k = 4, 16, 32, 8, 20
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    
    my_module = build_kernel()
    # Compute using the kernel.
    C_kernel = my_module.forward(A, B)
    # Compute reference result.
    C_ref = torch.einsum("bijl,lk->bijk", A, B)
    torch.cuda.synchronize()
    
    # Because the extra stream is not meaningfully used,
    # for contiguous, correct inputs, the result should match.
    # However, if there is unexpected interference, differences will show up.
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel output does not match torch.einsum output for contiguous inputs."

# Issue 5: Test that non-CUDA inputs cause an error.
def test_cpu_input_error():
    b, i, j, l, k = 4, 16, 32, 8, 20
    A = torch.randn(b, i, j, l, device="cpu", dtype=torch.float32)
    B = torch.randn(l, k, device="cpu", dtype=torch.float32)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel should check for CUDA tensors.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
