
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="swish_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to compute swish activation in Python.
def swish_python(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

# Test case for Issue #3:
# Passing a non-float32 tensor (e.g., double) should cause erroneous results.
def test_non_float_input():
    my_module = build_kernel()
    # Create a tensor with dtype double on CUDA.
    x = torch.randn(32, 32, dtype=torch.double, device='cuda')
    # We expect the kernel to misinterpret the data as float,
    # so the resulting output won't match the Python reference.
    y_kernel = my_module.forward(x)
    y_ref = swish_python(x)
    # The differences are expected because of wrong type interpretation.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), \
        "Kernel output unexpectedly matches reference output on non-float tensor!"

# Test case for ensuring that a CPU tensor is rejected.
def test_cpu_input():
    my_module = build_kernel()
    x = torch.randn(32, 32, device='cpu', dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA"):
        _ = my_module.forward(x)

# The following test simulates a huge tensor to trigger issue #1 and issue #2.
# It is marked as expected to fail in typical testing environments.
@pytest.mark.xfail(reason="Test for large tensor indexing and fmin conversion. Not practical on most systems.")
def test_large_tensor_index_overflow():
    my_module = build_kernel()
    # Attempt to create a very large 1D tensor. Note: This tensor is extremely large
    # and may not be actually allocatable on most devices.
    n = (2**31) + 1000  # This size is chosen to trigger potential 32-bit index issues.
    try:
        x = torch.randn(n, dtype=torch.float32, device='cuda')
    except RuntimeError as e:
        pytest.skip("Skipping test_large_tensor_index_overflow due to memory allocation issues on the device.")
    y_kernel = my_module.forward(x)
    y_ref = swish_python(x)
    # For a correct kernel, these results should be nearly identical.
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), \
        "Kernel output incorrect for large tensor index range!"

# Test case for Issue #4 by simulating a failure in a CUDA runtime call.
# We monkeypatch cudaStreamCreate to simulate an error.
def test_cuda_stream_error(monkeypatch):
    my_module = build_kernel()

    # Save the original cudaStreamCreate function from cuda_runtime
    import ctypes
    libcudart = ctypes.CDLL("libcudart.so")
    original_cudaStreamCreate = libcudart.cudaStreamCreate

    # Define a fake cudaStreamCreate that returns an error code.
    def fake_cudaStreamCreate(stream_ptr):
        return 1  # non-zero indicates an error in CUDA API

    # Monkey patch cudaStreamCreate
    monkeypatch.setattr(libcudart, "cudaStreamCreate", fake_cudaStreamCreate)

    x = torch.randn(32, 32, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x)
    
    # Restore the original function to avoid affecting other tests.
    monkeypatch.setattr(libcudart, "cudaStreamCreate", original_cudaStreamCreate)
