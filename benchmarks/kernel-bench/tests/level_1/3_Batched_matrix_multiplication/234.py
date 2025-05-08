
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build and load the CUDA extension module
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Global sizes used in our tests
batch_size = 16
M = 16
K = 32
N = 16

# Issue 1: Test that misaligned tensors trigger an error.
# This test creates a tensor with an offset in storage that is likely to break the
# 16-byte alignment requirement.
def test_unaligned_input():
    cuda_mod = build_kernel()
    
    # Allocate one extra element, then skip the first element to spoil the alignment.
    A_storage = torch.randn(batch_size * M * K + 1, device="cuda", dtype=torch.float32)
    # Create a view starting from offset 1; this likely breaks the 16-byte alignment.
    A_unaligned = A_storage.narrow(0, 1, batch_size * M * K).view(batch_size, M, K)
    
    B = torch.randn(batch_size, K, N, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError, match="Input tensor A must be 16-byte aligned"):
        _ = cuda_mod.forward(A_unaligned, B)

# Issue 2: Test that using a non-float32 (e.g. float64) tensor raises a problem.
def test_wrong_dtype():
    cuda_mod = build_kernel()
    
    A = torch.randn(batch_size, M, K, device="cuda", dtype=torch.float64)
    B = torch.randn(batch_size, K, N, device="cuda", dtype=torch.float64)
    
    # Since the kernel calls data_ptr<float>(), the wrong data type will lead to havoc.
    # We expect either an exception (if the TORCH_CHECK fails on alignment or dims) or
    # at least that the computed result is not close to the reference.
    with pytest.raises(RuntimeError):
        _ = cuda_mod.forward(A, B)

# Issue 3: Test that the kernel launch errors (or subsequent undefined behavior) are caught.
# In this simple test we do not trigger a CUDA launch error directly, but we use a call
# and then force a cuda synchronization. If the kernel had a launch error and we had error checking,
# this synchronization would raise an error. (Notice: The lack of immediate error checking is an issue.)
def test_kernel_launch_error_checking():
    cuda_mod = build_kernel()
    
    # Provide tensors with valid shapes and types.
    A = torch.randn(batch_size, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, K, N, device="cuda", dtype=torch.float32)
    C = cuda_mod.forward(A, B)
    
    # Force a synchronization. In a proper implementation kernel launch errors would be caught here.
    # Here, we simply check that C is produced but this test will help catch silent failures.
    torch.cuda.synchronize()
    # We do a basic shape check. (This test does not force an error but highlights the missing error-checking.)
    assert C.shape == (batch_size, M, N), "Output tensor C has an unexpected shape."

# Issue 4: Test that non‐contiguous input tensors lead to incorrect behavior.
def test_non_contiguous_input():
    cuda_mod = build_kernel()
    
    # Create a contiguous tensor and then make a non-contiguous view.
    A_contig = torch.randn(batch_size, M * 2, K, device="cuda", dtype=torch.float32)
    # Slice to get a non-contiguous tensor with the expected shape.
    A_noncontig = A_contig[:, ::2, :]  # This is non-contiguous.
    
    B_contig = torch.randn(batch_size, K, N * 2, device="cuda", dtype=torch.float32)
    B_noncontig = B_contig[:, :, ::2]  # Non-contiguous view.
    
    # Even though the shapes are correct, the kernel assumes contiguous memory.
    # We expect the results to differ from the reference computation.
    C_kernel = cuda_mod.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    C_ref = torch.bmm(A_noncontig.contiguous(), B_noncontig.contiguous())
    
    # The outputs are expected to differ because the kernel does not handle non-contiguity.
    # We trigger the issue if the results are (unexpectedly) close.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel did not exhibit error with non-contiguous inputs."

