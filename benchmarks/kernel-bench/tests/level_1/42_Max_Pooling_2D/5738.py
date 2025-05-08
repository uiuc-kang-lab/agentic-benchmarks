
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# We assume kernel.cu is in the current directory.
def build_kernel():
    return load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Using device std::numeric_limits with a type (half) that may not be supported.
def test_half_precision_input():
    # This test uses half precision. The kernel uses std::numeric_limits on scalar_t.
    # For torch.float16, std::numeric_limits may not be defined, triggering a runtime error.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel_module = build_kernel()
    batch_size = 2
    channels = 3
    height = 16
    width = 16
    # Using half precision input.
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float16)
    # We expect the kernel to fail (or produce an error) because of the use of std::numeric_limits.
    with pytest.raises(RuntimeError):
        # Depending on the extent of the device error, this may raise an error.
        kernel_module.forward(x, 2, 2, 1, 1)
    torch.cuda.synchronize()

# Issue 2: The unqualified use of max may be ambiguous in device code.
def test_integer_input():
    # The AT_DISPATCH_FLOATING_TYPES macro only supports floating types.
    # Passing an integer tensor should trigger an error.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel_module = build_kernel()
    batch_size = 2
    channels = 3
    height = 16
    width = 16
    # Create an integer tensor.
    x = torch.randint(0, 100, (batch_size, channels, height, width), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        kernel_module.forward(x, 2, 2, 1, 1)
    torch.cuda.synchronize()

# Issue 3: The kernel “specializes” for kernel_size == 2 and 3 but uses a generic branch for others.
# We test with a kernel_size that is not 2 or 3 (e.g. 7) and compare against PyTorch’s nn.MaxPool2d.
def test_generic_kernel_size():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel_size = 7
    stride = 2
    padding = 3
    dilation = 1
    kernel_module = build_kernel()
    batch_size = 2
    channels = 3
    height = 32
    width = 32
    # Input tensor.
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    # Output from our custom kernel.
    out_kernel = kernel_module.forward(x, kernel_size, stride, padding, dilation)
    # Output from PyTorch’s built-in max pooling.
    maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out_torch = maxpool(x)
    # They may not match exactly if there is an implementation bug.
    assert torch.allclose(out_kernel, out_torch, atol=1e-4), \
        f"Kernel output does not match PyTorch output for kernel_size {kernel_size}."

# Issue 4: There is no error checking after the kernel launch.
# We simulate an invalid configuration (for example, a negative padding) that should be caught by the implementation.
def test_invalid_configuration():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel_module = build_kernel()
    batch_size = 2
    channels = 3
    height = 16
    width = 16
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    # Negative padding is invalid.
    with pytest.raises(RuntimeError):
        kernel_module.forward(x, 3, 1, -1, 1)
    torch.cuda.synchronize()

# Issue 5: Using input.type() in the dispatch macro may be deprecated and disallow non-floating types.
# We check that a warning or error is raised when using an unsupported type (e.g., integer) and that the value
# of the built-in type name is reported.
def test_deprecated_dispatch_usage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel_module = build_kernel()
    batch_size = 2
    channels = 3
    height = 16
    width = 16
    # Here we again pass an integer tensor.
    x = torch.randint(0, 100, (batch_size, channels, height, width), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        kernel_module.forward(x, 3, 1, 1, 1)
    torch.cuda.synchronize()
