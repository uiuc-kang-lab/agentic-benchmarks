
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Passing a tensor with invalid dimensions for A (e.g., 3D tensor) should error.
def test_invalid_input_dimensions():
    module = build_kernel()
    # A is 3D instead of 2D. B is 1D (flattened).
    A = torch.randn(10, 20, 30, device="cuda", dtype=torch.float32)
    # B is generated to have 20*? but kernel expects B.numel() == A.size(1), so we pass a 1D tensor with wrong size.
    B = torch.randn(20, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception):
        C = module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Batched operations are not supported.
def test_batched_input():
    module = build_kernel()
    # Create batched A with shape (batch, M, K) and corresponding B with shape (batch, K, 1)
    batch = 4
    M = 16
    K = 64
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, 1, device="cuda", dtype=torch.float32)
    # The extension expects A to be 2D and B to be a vector. This batched input should raise an error.
    with pytest.raises(Exception):
        C = module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Missing CUDA error checking may hide failure. We trigger an intentional device mismatch.
def test_device_mismatch():
    module = build_kernel()
    M = 32
    K = 128
    # A on CUDA, but B on CPU. The kernel checks for A and B to be CUDA tensors.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, 1, device="cpu", dtype=torch.float32)
    with pytest.raises(Exception):
        C = module.forward(A, B)
        torch.cuda.synchronize()

# Issue 4: Non-floating point types are not supported.
def test_non_floating_point_types():
    module = build_kernel()
    M = 16
    K = 32
    # Create integer tensors
    A = torch.randint(0, 10, (M, K), device="cuda", dtype=torch.int32)
    B = torch.randint(0, 10, (K, 1), device="cuda", dtype=torch.int32)
    with pytest.raises(Exception):
        C = module.forward(A, B)
        torch.cuda.synchronize()

# Issue 5: Fixed block size assumption may lead to inefficiencies or potential issues when K is very small.
def test_small_K():
    module = build_kernel()
    M = 10
    # Choose K much smaller than blockDim.x (which is hard-coded to 256 in the kernel)
    K = 8
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32)
    # Although the kernel might still produce the correct result, the reduction logic may be suboptimal.
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We allow for a small numerical tolerance.
    assert torch.allclose(C, C_ref, atol=1e-5), "Kernel output does not match torch.matmul for small K."

if __name__ == "__main__":
    pytest.main([__file__])
