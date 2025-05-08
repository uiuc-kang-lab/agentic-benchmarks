
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the kernel from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue with non-contiguous tensor.
# Here we create a tensor via a transpose which makes it non-contiguous.
# The kernel does no check for contiguity so the kernel will use its vectorized branch,
# leading to an incorrect result compared to torch.mul.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor then transpose to make it non-contiguous.
    M, N = 256, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float32)
    A_non_contig = A.t()  # non-contiguous view
    s = 3.14

    # Run our custom kernel
    C_kernel = my_module.forward(A_non_contig, s)
    # Run the correct multiplication using PyTorch (which handles striding correctly)
    C_ref = A_non_contig * s

    torch.cuda.synchronize()
    # We expect the results to differ because the kernel does not account for non-contiguous layouts.
    # Here we check that they are not almost equal.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Test failed: The kernel computed the same result on a non-contiguous tensor even though it should not."
    )

# Test 2: Trigger issue with misaligned tensors.
# By slicing off one element from a contiguous tensor, the new tensor will be contiguous 
# but its data_ptr() is mis-aligned for float4 accesses.
def test_misaligned_tensor():
    my_module = build_kernel()
    # Create a 1D tensor with a size that is a multiple of 4 plus one extra element.
    size = 1025  # 1024 is a multiple of 4; 1025 forces an offset in the slice.
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    # Slice from element 1 to get a tensor that is contiguous but misaligned.
    A_misaligned = A.narrow(0, 1, size - 1)
    s = 2.71

    # Run our custom kernel
    C_kernel = my_module.forward(A_misaligned, s)
    # Run the correct multiplication using PyTorch
    C_ref = A_misaligned * s

    torch.cuda.synchronize()
    # Because the kernel’s alignment check only looks at A.data_ptr() and may not correctly handle the misaligned view,
    # we expect the result to be incorrect.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), (
        "Test failed: The kernel output on a misaligned tensor matches the expected result, though it should not."
    )

# Test 3: Trigger issue with wrong input tensor type.
# The kernel expects a float tensor, so passing a tensor of a different type should raise an error.
def test_wrong_dtype_tensor():
    my_module = build_kernel()
    M, N = 128, 128
    # Create a double (float64) tensor which should not be accepted.
    A = torch.randn(M, N, device="cuda", dtype=torch.float64)
    s = 1.23
    with pytest.raises(RuntimeError, match="Input tensor A must be of type float."):
        my_module.forward(A, s)

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
