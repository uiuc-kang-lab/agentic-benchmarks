
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="l1norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Non-float input (issue 2)
def test_non_float_input():
    # Create a double tensor on CUDA.
    x = torch.randn(16, 16384, dtype=torch.double, device="cuda")
    kernel = build_kernel()
    # Since the kernel always interprets the raw pointer as float,
    # this test is designed to cause either a crash or wrong values.
    with pytest.raises(RuntimeError):
        # We expect a runtime error because the memory reinterpretation
        # of double as float will likely lead to an illegal memory access or assertion.
        _ = kernel.forward(x)
    torch.cuda.synchronize()

# Test 2: Invalid tensor dimension (issue 1)
def test_invalid_dimension():
    # Create a 3D tensor even though the kernel expects a 2D tensor.
    x = torch.randn(4, 16, 16384, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        _ = kernel.forward(x)
    # Check that the error message indicates the dimension check failure.
    assert "Expected 2D tensor" in str(excinfo.value)
    torch.cuda.synchronize()

# Test 3: Extremely large batch size (issue 3)
def test_excessively_large_batch():
    # Many GPUs have a limit on the maximum grid dimension (number of blocks).
    # Here we try to force a kernel launch with a batch size that may exceed that limit.
    # NOTE: This test allocates a tensor with small row size to avoid huge memory usage.
    # Adjust the batch_size below if necessary.
    max_grid_size = 70000  # This is an arbitrary number chosen to be near common limits.
    x = torch.randn(max_grid_size, 32, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # The launch configuration uses one block per row so if max_grid_size is beyond
        # the device capability, this should raise an error.
        _ = kernel.forward(x)
    torch.cuda.synchronize()
