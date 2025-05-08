
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to load and compile the CUDA kernel module from kernel.cu.
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
    )

# Issue 1: Kernel assumes float32 input.
# We trigger it by sending in double tensors.
def test_float32_required():
    cuda_module = build_kernel()
    # Create double precision tensors.
    M, K, N = 128, 32, 128
    A = torch.randn(M, K, dtype=torch.double, device="cuda")
    B = torch.randn(K, N, dtype=torch.double, device="cuda")
    
    # Although PyTorch checks in the C++ code only for .is_cuda() and .is_contiguous(),
    # using a non-float32 tensor will lead to misinterpretation of data in the kernel.
    # We expect the computation to produce results that do NOT match the reference.
    C_kernel = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # The error is subtle: the kernel will run but compute with data of the wrong type.
    # So we check that the outputs are not equal.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-3), (
        "Kernel unexpectedly produced correct results for non-float32 inputs."
    )

# Issue 2: Kernel does not check that the inner dimensions match.
# We trigger it by giving mismatched inner dimensions.
def test_incompatible_dimensions():
    cuda_module = build_kernel()
    # In this test, A and B have mismatched inner dimensions, e.g. A is (M,K) and B is (K+1, N)
    M, K, N = 128, 32, 128
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    # Create B with wrong dimensions such that B.rows != A.cols.
    B = torch.randn(K + 1, N, dtype=torch.float32, device="cuda")
    
    # torch.matmul would raise an error, but our kernel code uses B.size(1) from B assuming
    # the inner dimension is A.size(1). This mismatch will lead to an erroneous launch.
    # We wrap in pytest.raises to capture any exception.
    with pytest.raises(Exception):
        C_kernel = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Kernel supports only 2D matrix multiplication.
# We test it by providing a batched tensor (3D tensor) that represents batched matrix multiplication.
def test_batched_input():
    cuda_module = build_kernel()
    # Create a batched matrix multiplication scenario.
    # For example, A is (B, M, K) and B is (B, K, N) where B is the batch size.
    batch_size, M, K, N = 4, 64, 32, 64
    A = torch.randn(batch_size, M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(batch_size, K, N, dtype=torch.float32, device="cuda")
    
    # The current kernel expects 2D tensors. Thus, when a 3D tensor is passed,
    # it may not only compute the wrong result but can launch with wrong dimensions.
    with pytest.raises(Exception):
        C_kernel = cuda_module.forward(A, B)
        torch.cuda.synchronize()
