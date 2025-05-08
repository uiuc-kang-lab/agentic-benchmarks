
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure the proper paths are used.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(this_dir, "kernel.cu")
    cuda_module = load(
        name="kernel_module",
        sources=[cuda_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test Case 1: Non-Contiguous Input Test
def test_non_contiguous_input():
    # Create a contiguous tensor then make it non-contiguous by transposing
    M, N = 1024, 512
    # Create a contiguous tensor in float32 on CUDA
    A = torch.randn(M, N, device="cuda", dtype=torch.float32)
    # Create a non-contiguous tensor view (for instance, by transposing)
    A_non_contig = A.t()  # This makes the tensor non-contiguous
    s = 3.14

    my_kernel = build_kernel()
    
    # We expect that the kernel may produce wrong results or error out because it directly manipulates
    # the raw data pointer assuming contiguity. Here, we check that the output does NOT match the expected.
    # For a correct general implementation, one might automatically call .contiguous().
    C = my_kernel.forward(A_non_contig, s)

    # Compute reference result (forcing contiguous behavior in the reference)
    # Here, we apply the operation on the contiguous version of A_non_contig.
    C_ref = A_non_contig.contiguous() * s

    # The test will fail if the kernel handled non-contiguous inputs correctly.
    # Otherwise, we expect a discrepancy.
    assert not torch.allclose(C, C_ref), (
        "Kernel unexpectedly handled non-contiguous inputs correctly. "
        "It should yield an error or produce incorrect results for non-contiguous tensors."
    )

# Test Case 2: Unsupported Data Type Test
def test_unsupported_dtype():
    M, N = 1024, 512
    # Create a tensor using double precision data (float64) on CUDA.
    A_double = torch.randn(M, N, device="cuda", dtype=torch.float64)
    s = 3.14

    my_kernel = build_kernel()

    with pytest.raises(RuntimeError, match="Input tensor A must be of type float."):
        # This should raise an error because the TORCH_CHECK in the host function expects torch::kFloat.
        my_kernel.forward(A_double, s)
