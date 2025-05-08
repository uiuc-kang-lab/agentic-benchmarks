
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_dtype():
    """
    Test that passing a tensor of type other than float32 (here float64) raises an error.
    Issue 1: The kernel assumes float32 and does not check the tensor datatype.
    """
    my_module = build_kernel()
    # Create a float64 tensor on CUDA
    x = torch.randn(1024, dtype=torch.float64, device='cuda')
    negative_slope = 0.01
    # Expect the kernel to raise an error because CHECK_INPUT only checks device and contiguity,
    # leaving the float32 assumption unchecked.
    with pytest.raises(RuntimeError):
        # The following call should misinterpret the pointer data since the tensor is not float32.
        _ = my_module.forward(x, negative_slope)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_launch_error_checking():
    """
    Test for kernel launch issues.
    Issue 2: The kernel does not perform error checking after the kernel launch.
    This test deliberately creates a non-contiguous tensor to ensure the CHECK_INPUT macro triggers an error.
    """
    my_module = build_kernel()
    # Create a non-contiguous tensor by transposing a 2D tensor
    x = torch.randn(32, 64, device='cuda', dtype=torch.float32).t()  # .t() returns a non-contiguous tensor
    negative_slope = 0.01
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x, negative_slope)
