
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from the kernel.cu file.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_input_tensor_type_error():
    """
    Test that passing a tensor of wrong dtype (non-float32) raises an error.
    This is expected because the kernel assumes x.data_ptr<float>().
    """
    cuda_module = build_kernel()
    # Create a double tensor instead of a float tensor.
    x = torch.randn(16, 16384, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError):
        # Expect the kernel to error out when accessing data_ptr<float>()
        y = cuda_module.forward(x)
        # Force synchronization to catch errors
        torch.cuda.synchronize()

def test_non_contiguous_tensor():
    """
    Test that passing a non-contiguous tensor causes issues.
    The kernel assumes contiguous memory and may produce incorrect results or error.
    """
    cuda_module = build_kernel()
    # Create a contiguous tensor then make it non-contiguous by transposing
    x = torch.randn(16, 16384, dtype=torch.float32, device="cuda")
    # Create a non-contiguous view, e.g., by transposing a 2D tensor
    x_noncontig = x.t()  # This view has different strides and is non-contiguous
    # Even though the input is non-contiguous, the CUDA kernel uses data_ptr() directly,
    # which may result in incorrect output or a runtime error.
    # We are testing that this situation is not handled properly by the kernel.
    with pytest.raises(Exception):
        y = cuda_module.forward(x_noncontig)
        torch.cuda.synchronize()

def test_kernel_launch_error_checking():
    """
    Test to simulate conditions that might trigger a kernel launch error
    and check that the lack of error checking does not catch it.
    For example, we can pass an empty tensor (numel==0) if the kernel doesn't check.
    """
    cuda_module = build_kernel()
    # Passing an empty tensor should ideally work, but if there is an error
    # in launch error checking, it might go unnoticed.
    x = torch.empty(0, dtype=torch.float32, device="cuda")
    # Since an empty tensor is a valid input, we expect no error here.
    # If an error is raised, it indicates that the kernel is mishandling edge cases.
    y = cuda_module.forward(x)
    torch.cuda.synchronize()
    assert y.numel() == 0, "Output tensor should be empty when input is empty."
