
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue with incompatible block configuration assumption.
# Although the kernel launch configuration is hardcoded in the C++ function,
# we simulate a “general case” by calling the kernel with a batched matrix as input.
# The kernel expects a 2D A and a 1D B (after view) resulting in C of shape (M,1).
# Passing a batched tensor (3D tensor A) should trigger an indexing error.
def test_batched_input_error():
    my_module = build_kernel()
    # Create batched inputs
    batch = 4
    M, K = 16, 1024
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, 1, device="cuda", dtype=torch.float32)
    
    # Since the C++/CUDA wrapper expects A to be 2-dimensional and B to be convertible to 1D,
    # we expect a runtime error (or TORCH_CHECK failure) when passing a batched tensor.
    with pytest.raises(RuntimeError):
        _ = my_module.forward(A, B)

# Test 2: Trigger issue by passing a B tensor that is not a vector.
# The kernel expects that B.numel() equals the number of columns in A.
def test_invalid_B_shape():
    my_module = build_kernel()
    M, K = 32, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # Provide B with shape (K, 2) which is not valid as B.numel() != K after flattening.
    B = torch.randn(K, 2, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        _ = my_module.forward(A, B)

# Note:
# It is difficult to force the kernel to run with a blockDim.x that is not a multiple of 32,
# because the kernel launch configuration is hardcoded inside the C++ wrapper.
# However, issue 1 (incorrect row indexing when blockDim.x is not a multiple of 32)
# remains a design flaw that would manifest if the kernel were launched differently.
#
# Test 3: (Simulated) Check behavior on very large tensors that may exceed 32-bit indexing.
# We cannot realistically allocate more than 2^31-1 elements on typical GPUs,
# but we simulate the test by mocking a scenario where the output tensor size is extremely large.
def test_large_dimension_warning():
    my_module = build_kernel()
    
    # Create an input matrix A whose number of elements exceeds 2^31-1 is not feasible to allocate.
    # Instead, we simulate the check by creating a tensor with a huge second dimension.
    # This test is expected to fail or produce an error if the kernel does not handle large dimensions.
    M = 1
    # We'll use a large K value; note that 2**31 ~ 2.1e9, so we use a value to simulate the edge.
    K = 2**31 // 2  # This is still huge and cannot really be allocated but serves as a conceptual test.
    
    # We simulate the test by catching a RuntimeError from TORCH_CHECK (or allocation failure).
    A = torch.empty(M, K, device="cuda", dtype=torch.float32)
    B = torch.empty(K, 1, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        _ = my_module.forward(A, B)
