
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper function to build the CUDA extension.
def build_kernel():
    # Ensure we compile the extension in the current directory
    module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test 1: Trigger issue with misaligned memory.
# We simulate misalignment by allocating a tensor larger than needed and then slicing it.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_misaligned_memory():
    # Create a tensor with an extra row so that slicing off the first row results 
    # in a pointer that may not obey the expected 128-bit alignment.
    M, K, N = 256, 512, 256  # Use smaller K to run test faster
    A_full = torch.randn(M + 1, K, device="cuda", dtype=torch.float32)
    B_full = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Slicing A to potentially break alignment.
    A = A_full[1:]
    # Verify that A is not contiguous in memory in the usual sense.
    assert not A.is_contiguous(), "Test setup failure: A is contiguous."
    
    ext = build_kernel()
    C = ext.forward(A, B_full)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B_full)
    # Likely the misaligned access may produce a wrong result.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Kernel unexpectedly worked correctly with misaligned memory."

# Test 2: Trigger issue with non-contiguous tensors.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_non_contiguous_inputs():
    M, K, N = 256, 512, 256
    # Create contiguous tensors.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Make a non-contiguous version by transposing twice.
    # For example, take a transposed view and then transpose back.
    A_noncontig = A.t().t()  # This view is not guaranteed to be contiguous.
    B_noncontig = B.t().t()
    # Confirm non-contiguity.
    assert not A_noncontig.is_contiguous(), "Test setup failure: A_noncontig is contiguous."
    assert not B_noncontig.is_contiguous(), "Test setup failure: B_noncontig is contiguous."
    
    ext = build_kernel()
    C = ext.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A_noncontig, B_noncontig)
    # The kernel does not handle non-contiguous tensors properly so results may differ.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Kernel unexpectedly produced correct results for non-contiguous tensors."

# Test 3: Trigger kernel launch error by giving mismatched inner dimensions.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dimension_mismatch_error():
    M, K, N = 256, 512, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # Introduce a mismatch: B has inner dimension different from A's column count.
    B = torch.randn(K + 1, N, device="cuda", dtype=torch.float32)
    
    ext = build_kernel()
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the kernel should raise an error.
        C = ext.forward(A, B)
        torch.cuda.synchronize()

# Test 4: (Optional) Ensure that errors from asynchronous kernel launches are checked.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_no_error_checking_after_launch():
    # Since the kernel does not check for launch errors,
    # we can simulate an error case by providing unexpected dimensions.
    M, K, N = 15, 31, 14  # Choose dimensions that are not multiples of TILE_WIDTH.
    # These dimensions are legal but stress the boundary condition paths.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    ext = build_kernel()
    C = ext.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # In some error cases the kernel might output garbage.
    # We are not expecting a launch error but a mismatch in results.
    assert not torch.allclose(C, C_ref, atol=1e-4), "Kernel unexpectedly produced correct output despite potential boundary issues."

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
