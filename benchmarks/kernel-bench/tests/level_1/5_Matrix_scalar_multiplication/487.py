
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Note: verbose flag set to True for debugging build messages.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="session")
def kernel_module():
    return build_kernel()

def test_non_contiguous_tensor(kernel_module):
    # Create a tensor and then make it noncontiguous by transposing.
    # The CUDA kernel does not account for arbitrary strides.
    M, N = 64, 32
    A = torch.randn(M, N, device='cuda', dtype=torch.float32)
    A_t = A.t()  # transpose -> noncontiguous
    s = 2.0

    # Call the CUDA kernel. Since it uses A.data_ptr and assumes contiguous data,
    # the result should be incorrect.
    C = kernel_module.forward(A_t, s)
    # To get the expected result using PyTorch, we force contiguous.
    C_ref = A_t.contiguous() * s

    # Check that the outputs differ.
    # We expect the kernel to miscompute, so they should NOT be all close.
    with pytest.raises(AssertionError):
        assert torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly computed correct result for noncontiguous tensor."

def test_small_tensor_no_vector_groups(kernel_module):
    # Create a very small tensor with fewer than 4 elements.
    # The grid dimension will be computed as zero, so the kernel will not launch any threads
    # to process the remainder elements.
    A = torch.tensor([1.0, 2.0, 3.0], device='cuda', dtype=torch.float32)
    s = 3.0
    C = kernel_module.forward(A, s)
    # Expected result using PyTorch's multiplication
    C_ref = A * s

    # Since the kernel launch results in no threads processing the remainder,
    # the output will remain uninitialized (or hold garbage values).
    # Test that the computed result is not equal to the reference.
    with pytest.raises(AssertionError):
        assert torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly computed correct result for a small tensor with less than 4 elements."

if __name__ == "__main__":
    pytest.main([__file__])
