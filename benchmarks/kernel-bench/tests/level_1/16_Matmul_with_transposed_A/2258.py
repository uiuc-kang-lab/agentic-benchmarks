
import os
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to build the CUDA extension module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Check for the use of unqualified "min" in the kernel source.
def test_unqualified_min_usage():
    # Read the kernel file and ensure that "std::min" is used rather than a bare "min".
    # If bare "min" is found, this signals the issue.
    kernel_src = None
    kernel_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    with open(kernel_path, "r") as f:
        kernel_src = f.read()
    # The issue is that the code uses unqualified "min". We expect to see "std::min" in
    # a corrected version. Thus, if the code does not contain "std::min", then the issue is present.
    assert "std::min" in kernel_src, (
        "Kernel uses unqualified 'min'. It should use 'std::min' for portability and to ensure compilation."
    )

# Test case 2: Trigger the problem with non-contiguous inputs.
def test_non_contiguous_input():
    # Set up dimensions that match the kernel expectations:
    # A is expected to be of shape (K, M) and B of shape (K, N).
    K = 4096
    M = 1024
    N = 2048

    # Create contiguous tensors first and then induce non-contiguity with slicing.
    # We create a larger tensor and then select every other column to force non-contiguous memory.
    A_full = torch.randn(K, M * 2, device="cuda", dtype=torch.float32)
    B_full = torch.randn(K, N * 2, device="cuda", dtype=torch.float32)
    # Slice to produce the required shapes. The resulting tensors are non-contiguous.
    A_noncontig = A_full[:, ::2]  # shape (K, M)
    B_noncontig = B_full[:, ::2]  # shape (K, N)
    assert not A_noncontig.is_contiguous(), "Expected A_noncontig to be non-contiguous."
    assert not B_noncontig.is_contiguous(), "Expected B_noncontig to be non-contiguous."

    # Load the CUDA kernel.
    module = build_kernel()

    # Run the kernel: note that the kernel computes C = A.T * B.
    C_kernel = module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()

    # Compute the reference result using PyTorch's matmul.  Since the model does A.T, we do likewise.
    C_ref = torch.matmul(A_noncontig.t(), B_noncontig)

    # Because the kernel does not handle non-contiguous inputs correctly, we expect a discrepancy.
    # Thus, the kernel output should differ (i.e. not be allclose) from the correct result.
    # (If they were the same, then the kernel would be handling non-contiguity.)
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Kernel output is correct even with non-contiguous inputs, "
        "but the implementation does not handle non-contiguity explicitly."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
