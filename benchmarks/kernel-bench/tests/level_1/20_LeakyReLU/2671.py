
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA kernel module.
def build_kernel():
    cuda_module = load(
        name="leaky_relu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue for non-float32 input.
def test_non_float_input():
    my_module = build_kernel()
    # Create a double tensor on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    negative_slope = 0.01
    with pytest.raises(RuntimeError) as excinfo:
        # This call will pass the tensor to the kernel, which then incorrectly
        # casts its pointer to float*. This should lead to a runtime error.
        my_module.forward(x, negative_slope)
    assert "must be a CUDA tensor" not in str(excinfo.value), "Unexpected CHECK_CUDA error message"
    # If no error, the result will be incorrect as the kernel expects float.

# Test 2: Trigger issue for non-contiguous input.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32).t()
    negative_slope = 0.01
    with pytest.raises(RuntimeError) as excinfo:
        # The CHECK_INPUT macro should trigger an error because x is non-contiguous.
        my_module.forward(x, negative_slope)
    assert "must be contiguous" in str(excinfo.value)

# Test 3: Test the shared memory branch by using an input with number of elements >= SHARED_MEMORY_THRESHOLD.
def test_shared_memory_branch():
    my_module = build_kernel()
    # Create a large tensor whose total number of elements is >= 1M.
    n = 1048576  # exactly the threshold, can also go above
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    negative_slope = 0.01
    # Call the kernel; it uses the shared memory branch.
    out = my_module.forward(x, negative_slope)
    # Validate result by comparing to PyTorch's own implementation.
    out_ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    torch.cuda.synchronize()
    assert torch.allclose(out, out_ref, atol=1e-5), "Shared memory branch output does not match torch.nn.functional.leaky_relu"

# Test 4: Test the non-shared memory branch using a small tensor.
def test_non_shared_memory_branch():
    my_module = build_kernel()
    # Use a tensor with less than threshold elements.
    n = 1000
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    negative_slope = 0.01
    out = my_module.forward(x, negative_slope)
    out_ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    torch.cuda.synchronize()
    assert torch.allclose(out, out_ref, atol=1e-5), "Non-shared memory branch output does not match torch.nn.functional.leaky_relu"
