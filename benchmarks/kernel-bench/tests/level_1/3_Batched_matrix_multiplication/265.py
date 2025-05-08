
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA extension module
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Test that using tensors of a type other than float32 fails.
def test_input_tensor_type():
    my_module = build_kernel()
    batch_size = 2
    M, K, N = 4, 8, 3
    # Create double precision tensors (float64)
    A = torch.randn(batch_size, M, K, dtype=torch.float64, device='cuda')
    B = torch.randn(batch_size, K, N, dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError):
        # This should fail because the kernel expects float32
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Test that non-contiguous input tensors produce incorrect results.
def test_noncontiguous_inputs():
    my_module = build_kernel()
    batch_size = 2
    M, K, N = 4, 8, 3
    # Create contiguous tensors first
    A = torch.randn(batch_size, M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(batch_size, K, N, dtype=torch.float32, device='cuda')
    # Create non-contiguous tensors by transposing a batch dimension with a spatial dimension
    A_noncontig = A.transpose(1, 2)  # Now shape: (batch_size, K, M), non-contiguous
    B_noncontig = B  # Keep one tensor contiguous for simplicity

    # Permute A_noncontig back to expected shape without calling contiguous()
    # This simulates a scenario where the tensor is non-contiguous.
    A_noncontig = A_noncontig.transpose(1, 2)  # Back to (batch_size, M, K) but non-contiguous

    # Compute reference result using torch.bmm (which handles non-contiguity)
    C_ref = torch.bmm(A_noncontig, B_noncontig)
    # Call our kernel using the non-contiguous tensor
    C = my_module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    # The kernel does not handle non-contiguity so the results may differ.
    # We check that the result is not equal to the reference.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct results with non-contiguous inputs!"

# Issue 3: Test that invalid dimensions (that pass basic TORCH_CHECK but break the kernel assumptions) produce errors.
def test_invalid_dimensions():
    my_module = build_kernel()
    # Create tensors with mismatched inner dimensions that still pass some basic checks
    batch_size = 2
    M, K1, K2, N = 4, 8, 7, 3  # K dimensions differ
    A = torch.randn(batch_size, M, K1, dtype=torch.float32, device='cuda')
    B = torch.randn(batch_size, K2, N, dtype=torch.float32, device='cuda')
    # Our kernel checks A.size(2)==B.size(1) so we must bypass this check.
    # For testing, we simulate a scenario where the dimensions are set manually wrong
    # by directly calling the underlying kernel function via a wrapper that ignores the TORCH_CHECK.
    # Here, one may expect the kernel to produce wrong results or even fail.
    # Since we cannot directly bypass TORCH_CHECK easily,
    # we check that the forward function raises a RuntimeError due to the TORCH_CHECK.
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
