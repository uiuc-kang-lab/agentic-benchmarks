
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension (loading kernel.cu)
def build_kernel():
    module = load(
        name="custom_maxpool",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test case 1: Trigger the device max() issue by attempting to run the kernel with an unsupported floating point type.
# The kernel dispatch macro AT_DISPATCH_FLOATING_TYPES only covers float and double.
# Passing half precision (float16) should result in a runtime or compilation error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_invalid_dtype_float16():
    # Create a tensor with float16 type and valid shape.
    x = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float16)
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect a failure because float16 is not handled by AT_DISPATCH_FLOATING_TYPES.
        out = mod.forward(x, kernel_size, stride, padding, dilation)
        # Force synchronization to catch asynchronous failures.
        torch.cuda.synchronize()

# Test case 2: Trigger the lack of kernel launch error checking.
# We intentionally provide pooling parameters that cause the computed output dimensions to be negative,
# leading to an invalid launch configuration.
# For instance, using a very small input with dilation and kernel size parameters that over‐stretch the effective window.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_invalid_output_dimensions():
    # Create a very small input tensor that will lead to negative output dimensions.
    x = torch.randn(1, 1, 1, 1, device="cuda", dtype=torch.float32)
    # Choose parameters to force negative output size.
    # Example: dilation is high relative to input size and no padding.
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 2  # Effective kernel size becomes 2*(3-1)+1 = 5, which is larger than input.
    mod = build_kernel()
    # The kernel launch is expected to be misconfigured.
    # Depending on the runtime, this may surface as an error on cudaDeviceSynchronize.
    out = mod.forward(x, kernel_size, stride, padding, dilation)
    with pytest.raises(RuntimeError):
        torch.cuda.synchronize()

# Test case 3: Trigger an issue due to unsupported input type (e.g. integer types).
# The kernel is only dispatched for floating point types.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unsupported_integer_type():
    x = torch.randint(0, 10, (16, 32, 128, 128), device="cuda", dtype=torch.int32)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        out = mod.forward(x, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()
