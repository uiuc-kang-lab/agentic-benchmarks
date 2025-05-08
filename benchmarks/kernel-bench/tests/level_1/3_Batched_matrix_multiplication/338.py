
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel is hard-coded for float32. When a non-float32 tensor (e.g., float64) is passed, the kernel may compute incorrect results.
def test_input_tensor_type():
    cuda_module = build_kernel()
    batch_size, m, k, n = 10, 32, 32, 32
    A = torch.randn(batch_size, m, k, device='cuda', dtype=torch.float64)
    B = torch.randn(batch_size, k, n, device='cuda', dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # We expect that launching the kernel with non-float32 will trigger an error
        _ = cuda_module.forward(A, B)

# Issue 2: The kernel assumes contiguous memory. When non-contiguous inputs are passed, the result may be incorrect.
def test_non_contiguous_inputs():
    cuda_module = build_kernel()
    batch_size, m, k, n = 8, 33, 31, 33  # use dimensions that may force non-contiguity when transposed
    A = torch.randn(batch_size, m, k, device='cuda', dtype=torch.float32)
    B = torch.randn(batch_size, k, n, device='cuda', dtype=torch.float32)
    # Create non-contiguous tensors via transpose
    A_noncontig = A.transpose(1, 2)
    B_noncontig = B.transpose(1, 2)
    # Although the kernel does not check for non-contiguity, the results are expected to be incorrect.
    C = cuda_module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    C_ref = torch.bmm(A_noncontig, B_noncontig)
    # The difference is likely not within tolerance
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel should not work correctly on non-contiguous tensors."

# Issue 3 & 4: Using constant memory for dimensions and assuming TILE_SIZE divides the dimensions cleanly.
# When dimensions are not multiples of TILE_SIZE, the kernel relies on padded loads and fully unrolled loops.
def test_non_multiple_tile_dimensions():
    cuda_module = build_kernel()
    batch_size, m, k, n = 7, 45, 50, 47  # dimensions not multiples of TILE_SIZE (32)
    A = torch.randn(batch_size, m, k, device='cuda', dtype=torch.float32)
    B = torch.randn(batch_size, k, n, device='cuda', dtype=torch.float32)
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.bmm(A, B)
    # The kernel might produce inaccuracies due to how it unrolls TILE_SIZE steps
    assert not torch.allclose(C, C_ref, atol=1e-4), "Kernel did not trigger an issue with non-multiple-of-TILE_SIZE dimensions as expected."
