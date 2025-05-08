
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Rebuild the extension every time to pick up changes.
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Helper function for Swish using PyTorch
def torch_swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

@pytest.fixture(scope="module")
def cuda_kernel_module():
    return build_kernel()

def test_wrong_dtype(cuda_kernel_module):
    # Issue 1: The kernel only supports float32. Passing float64 should trigger an error
    # or produce incorrect results.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # Expect a runtime error because our kernel accesses float pointer.
        _ = cuda_kernel_module.forward(x)
        
def test_non_contiguous_input(cuda_kernel_module):
    # Issue 2: The kernel assumes contiguous memory.
    # Create a tensor and then a non contiguous view.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    x_t = x.transpose(0, 1)  # Now non contiguous
    y = cuda_kernel_module.forward(x_t)
    # Compute the expected output using torch where non-contiguous is handled correctly.
    y_expected = torch_swish(x_t)
    # The faulty kernel will process the data as if it were contiguous.
    # This test should fail because the output won’t match the expected result.
    assert not torch.allclose(y, y_expected, atol=1e-5), (
        "Kernel output for non-contiguous tensor unexpectedly matches the expected output."
    )

def test_large_tensor_index_overflow(cuda_kernel_module):
    # Issue 3: The kernel uses 32-bit indexing.
    # We simulate this by checking behavior when tensor's numel() is very high.
    # Note: Allocating >INT_MAX elements is not feasible. Instead, we simulate
    # the issue by creating a tensor with a shape such that the number of elements is near the 32-bit limit.
    max_int32 = 2**31 - 1
    # We do not actually allocate such huge tensor; instead, we mimic the case by manipulating the shape.
    # Here, we create a tensor with an "artificial" flattened size exceeding INT_MAX by using a custom wrapper.
    # In practice, a tensor with more than INT_MAX elements would be too large to allocate,
    # so we check that the kernel dispatches based on int indices.
    n = max_int32 + 100  # An artificial number > INT_MAX
    # Allocate a tensor with n elements if possible, else skip the test.
    try:
        # This allocation might fail on many GPUs; if so, skip the test.
        x = torch.randn(n, device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Skipping large tensor test; unable to allocate tensor with >INT_MAX elements.")
        
    y = cuda_kernel_module.forward(x)
    y_expected = torch_swish(x)
    # There is a potential for indexing overflow so the output might diverge.
    assert not torch.allclose(y, y_expected, atol=1e-5), (
        "Kernel output for large tensor (simulate index overflow) unexpectedly matches the expected output."
    )

def test_missing_kernel_error_check(cuda_kernel_module):
    # Issue 4: The kernel does not check errors.
    # We trigger a kernel launch error by providing a tensor that is on CPU.
    x_cpu = torch.randn(1024, 1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = cuda_kernel_module.forward(x_cpu)

# Optionally, one might check for performance issues pertaining to using expf vs __expf.
# However, this is not straightforward to test with unit tests and is better measured with profiling.
