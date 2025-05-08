
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility function to build the CUDA extension from kernel.cu.
def build_kernel():
    # Ensure the kernel file exists in the current directory.
    kernel_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    if not os.path.exists(kernel_file):
        raise FileNotFoundError(f"kernel.cu not found at {kernel_file}")
    cuda_module = load(
        name="test_module",
        sources=[kernel_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Unused vectorized load variables in linearNoDivKernel.
# Test: Use small matrix sizes to force the use of linearNoDivKernel and verify correctness.
def test_linear_kernel_correctness():
    my_module = build_kernel()
    M = 100    # small so that M and N are below THRESHOLD_SIZE
    K = 200
    N = 100
    # Create contiguous inputs of type float32.
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Expected result is A^T * B : shape (M, N)
    C_ref = torch.matmul(A.t(), B)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    assert torch.allclose(C, C_ref, atol=1e-5), \
        f"Linear kernel output mismatch. Max error: {(C-C_ref).abs().max().item()}"

# Issue 2: Lack of handling for non‐contiguous input tensors.
# Test: Pass non‐contiguous tensors to the CUDA kernel and verify that the result is incorrect.
def test_noncontiguous_inputs():
    my_module = build_kernel()
    M = 128
    K = 256
    N = 128
    # Create contiguous base tensor and then take a non-contiguous slice/by transposing twice.
    A_base = torch.randn(K, M*2, device="cuda", dtype=torch.float32)
    B_base = torch.randn(K, N*2, device="cuda", dtype=torch.float32)
    # Slice to get non-contiguous tensors.
    A = A_base[:, ::2]  # This is generally non-contiguous.
    B = B_base[:, ::2]
    # Compute reference using torch.matmul on non-contiguous tensors (torch.matmul handles non-contiguous inputs)
    C_ref = torch.matmul(A.t(), B)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Expect that due to the lack of contiguity check in the kernel,
    # the result might differ from the reference.
    # We use a high tolerance so that even small error is caught.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel did not fail on non-contiguous inputs as expected."

# Issue 3: Lack of device synchronization after kernel launch.
# Test: Pass deliberately mismatched dimensions to trigger an error.
def test_kernel_dimension_mismatch():
    my_module = build_kernel()
    M = 64
    K = 128
    N = 32
    # Create tensors with mismatching dimensions:
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    # B's first dimension is intentionally wrong (should be K but we use K+1)
    B = torch.randn(K+1, N, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()  # If error is not synchronized, this may trigger it.

# Issue 4: Kernel selection logic may be suboptimal for general inputs.
# Test: Provide large matrices so that the tiled kernel is used and check output against torch.matmul.
def test_tiled_kernel_correctness():
    my_module = build_kernel()
    # Choose M and N larger than THRESHOLD_SIZE to trigger the tiledSharedLdgKernel.
    M = 1024   # > THRESHOLD_SIZE (512)
    K = 512
    N = 1024   # > THRESHOLD_SIZE (512)
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C_ref = torch.matmul(A.t(), B)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    assert torch.allclose(C, C_ref, atol=1e-5), \
        f"Tiled kernel output mismatch. Max error: {(C-C_ref).abs().max().item()}"

if __name__ == "__main__":
    pytest.main([__file__])
