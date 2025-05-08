
import math
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    # Note: this will compile kernel.cu from the current directory.
    cuda_module = load(
        name="max_pool3d_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Non-cubic (tuple) pooling parameters.
# Since the kernel wrapper expects ints, passing a tuple for kernel_size should raise a TypeError.
def test_non_cubic_pooling():
    mod = build_kernel()
    # Create a dummy input tensor on CUDA.
    x = torch.randn(2, 3, 16, 16, 16, device='cuda', dtype=torch.float32)
    kernel_size = (3, 3, 3)  # non-cubic parameter (as tuple) is not supported.
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    return_indices = False
    ceil_mode = False

    with pytest.raises(TypeError):
        # Attempt to call the forward function with tuple parameters.
        # The extension binding expects ints so a tuple will trigger a type conversion error.
        mod.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

# Test case 2: Half precision input.
# Since the kernel dispatches only for floating types that support infinity (float/double),
# using a half tensor should trigger an error.
def test_half_precision_input():
    mod = build_kernel()
    x = torch.randn(2, 3, 16, 16, 16, device='cuda', dtype=torch.float16)
    # Using valid int parameters.
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False

    with pytest.raises(RuntimeError):
        # Expect the extension to fail dispatching on half precision (or at least not compute correctly)
        mod.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

# Test case 3: Lack of CUDA error checking.
# We force an invalid configuration (e.g. an extremely large dilation) so that memory accesses become invalid.
# In a properly error-checked kernel the error would be caught.
def test_invalid_parameters_for_cuda_error():
    mod = build_kernel()
    # Use a moderately sized input.
    x = torch.randn(2, 3, 16, 16, 16, device='cuda', dtype=torch.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 9999  # unrealistic dilation - will yield output dimensions that are nonsensical.
    return_indices = False
    ceil_mode = False

    # In the current implementation, the kernel does not do any error-checking so launching with nonsensical
    # parameters might result in device-side error. We force a synchronization and expect a CUDA error.
    with pytest.raises(RuntimeError):
        out = mod.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        torch.cuda.synchronize()  # Force host to check for asynchronous errors

# Test case 4: Missing <cmath> inclusion.
# This is a compile-time issue that cannot be caught by runtime tests;
# however we include a dummy test to remind maintainers that build failure is expected in some setups.
def test_cmath_inclusion_warning():
    # This test always passes. It is here only to indicate that if your build fails
    # it might be because <cmath> is not included.
    assert math.ceil(3.2) == 4

if __name__ == "__main__":
    pytest.main([__file__])
