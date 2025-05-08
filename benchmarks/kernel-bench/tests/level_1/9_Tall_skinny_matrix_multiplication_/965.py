
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper to build and load the CUDA extension
def build_kernel():
    # Assuming kernel.cu is in the current working directory
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32. Test with non-float32 (e.g. float64) and expect it to fail or produce mismatch.
def test_wrong_dtype():
    cuda_module = build_kernel()
    N = 64
    # Create inputs as double. The underlying kernel uses float; a mismatch is expected.
    A = torch.randn(128, N, device="cuda", dtype=torch.float64)
    B = torch.randn(N, 128, device="cuda", dtype=torch.float64)
    with pytest.raises(Exception):
        # We expect the kernel to either throw or produce results far off from torch.matmul.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Non-contiguous tensor input. The kernel assumes contiguous memory.
def test_non_contiguous_tensor():
    cuda_module = build_kernel()
    N = 32
    # Create a contiguous tensor then take its transpose to get a non-contiguous tensor.
    A_contig = torch.randn(128, N, device="cuda", dtype=torch.float32)
    B_contig = torch.randn(N, 128, device="cuda", dtype=torch.float32)
    A = A_contig.t()  # now non-contiguous and shape (N, 128)
    B = B_contig  # keeping B contiguous
    with pytest.raises(Exception):
        # Depending on how the kernel is used, this misalignment should cause an error or result in a mismatch.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Incorrect handling of transposed matrices.
# Since the kernel internally sets transA/transB to false and uses the raw strides,
# passing a transposed tensor (which is non-contiguous and whose stride differs)
# will yield incorrect results.
def test_incorrect_transpose_handling():
    cuda_module = build_kernel()
    N = 16
    # Create contiguous tensors
    A = torch.randn(128, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, 128, device="cuda", dtype=torch.float32)
    # Manually create a transposed version.
    A_t = A.t().contiguous()  # Forcing contiguity after transpose, but the expected strides for a transposed operation are not used.
    # When calling forward, the C++ kernel still uses A.size(0) and A.stride(0) from the passed tensor,
    # so the results will not match torch.matmul on transposed data.
    C_kernel = cuda_module.forward(A_t, B)
    torch.cuda.synchronize()
    # The correct result would be to compute (A_t * B) which is equivalent to torch.matmul(A.t(), B)
    C_ref = torch.matmul(A_t, B)
    # We expect a mismatch because the kernel does not correctly handle transposed layouts.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), \
        "Kernel unexpectedly computed correct result for transposed input despite known indexing issues."

# Issue 4: Hardcoded BLOCK_SIZE and NUM_STREAMS may lead to problems when dimensions are not divisible by BLOCK_SIZE.
# Create an input matrix where one or more dimensions is not a multiple of BLOCK_SIZE.
def test_dimension_not_divisible_by_block_size():
    cuda_module = build_kernel()
    # BLOCK_SIZE is defined as 16, so choose dimensions that are not multiples of 16.
    M, K, N = 70, 33, 45
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C_kernel = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Allowing for some numerical tolerance, but the results are expected to be wrong if edge cases aren’t handled properly.
    # Here, we force an assertion failure if the outputs are accidentally matching.
    if torch.allclose(C_kernel, C_ref, atol=1e-5):
        pytest.fail("Kernel produced correct output even though input dimensions are not divisible by BLOCK_SIZE. Expected miscalculation due to load imbalance or padding issue.")

if __name__ == "__main__":
    pytest.main([__file__])
