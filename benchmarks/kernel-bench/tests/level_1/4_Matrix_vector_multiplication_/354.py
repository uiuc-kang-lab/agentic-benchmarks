
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Using a tensor with dimensions that could potentially exceed 32-bit indexing
# NOTE: It is not generally feasible to allocate a giant tensor in a test.
# Instead, we simulate the scenario by checking for a warning or error message.
def test_large_tensor_indexing():
    my_module = build_kernel()
    # We simulate a large-K scenario by reporting the K argument.
    # Although we cannot really allocate a tensor that exceeds 32-bit indexing,
    # the idea is to call the function and check that it is documented to fail (or warn)
    # in a real large tensor use-case.
    M = 16
    # Pick a large K value - in testing we do not really allocate such a tensor.
    # Instead, use a size that is borderline (e.g., > 2**20) for testing purposes.
    # (For real large tensors one would need 64-bit indexing.)
    K = 2**21  
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32)
    with pytest.warns(UserWarning) as record:
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
    # Check that a warning was raised (in a proper implementation, one might
    # want to warn the user about the indexing limitations).
    # Here we simply check that the kernel runs and output shape is correct.
    assert C.shape == (M, 1)

# Test 2: Launch with an input that forces an incomplete warp.
# The current kernel launch always uses block size 256 (a multiple of 32),
# so we simulate the impact by using a very small K relative to the block size.
def test_incomplete_warp_reduction():
    my_module = build_kernel()
    # Create a tensor where K is much smaller than the block size,
    # so that many threads in the block will not participate in the accumulation.
    M = 1
    K = 10  # K < blockDim.x (256) forces many threads to do no work; 
            # if the reduction is not correctly accounting for inactive threads,
            # the result may be wrong.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # Provide B as shape (K,) so that after view it becomes a 1D tensor.
    B = torch.randn(K, device="cuda", dtype=torch.float32)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Calculate the reference output using torch.matmul.
    C_ref = torch.matmul(A, B.view(-1, 1))
    # The outputs may differ if the reduction did not correctly ignore inactive threads.
    assert torch.allclose(C, C_ref, atol=1e-5), \
        f"Output mismatch for incomplete warp case. Kernel: {C}, reference: {C_ref}"

# Test 3: Pass half precision (float16) inputs to trigger the unsupported type path.
def test_half_precision_unsupported():
    my_module = build_kernel()
    M, K = 16, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.half)
    B = torch.randn(K, 1, device="cuda", dtype=torch.half)
    with pytest.raises(RuntimeError):
        # The dispatch macro only supports float and double so float16 should raise an error.
        C = my_module.forward(A, B)

if __name__ == "__main__":
    pytest.main([__file__])
