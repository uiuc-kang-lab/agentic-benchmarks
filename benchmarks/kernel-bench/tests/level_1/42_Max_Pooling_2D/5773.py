
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper function to compile the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="test_maxpool",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2: Test using double precision input
# This test verifies that the maximum operation is computed correctly using a double tensor.
def test_double_precision():
    cuda_mod = build_kernel()
    # Use double-precision input.
    batch_size, channels, height, width = 2, 3, 16, 16
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1

    # Create a random input tensor in double precision on CUDA.
    input_tensor = torch.randn(batch_size, channels, height, width, dtype=torch.double, device="cuda")
    # Compute expected output using PyTorch's built-in max_pool2d
    expected = F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    # Compute output using the custom CUDA kernel
    out = cuda_mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=1e-5), f"Double precision test failed. Max diff: {(out - expected).abs().max().item()}"

# Issue 3: Test missing kernel launch error checking.
# We simulate error detection by deliberately providing an incorrectly shaped input (e.g. missing dimensions)
def test_invalid_input_shape():
    cuda_mod = build_kernel()
    # Create an input tensor with wrong dimensionality 
    # (the kernel expects 4D: (batch, channel, height, width))
    input_tensor = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    with pytest.raises(RuntimeError):
        # This call should fail given the input shape conflict.
        out = cuda_mod.forward(input_tensor, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 4: Test limited type support by providing a half-precision input.
def test_half_precision_not_supported():
    cuda_mod = build_kernel()
    batch_size, channels, height, width = 2, 3, 16, 16
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1
    input_tensor = torch.randn(batch_size, channels, height, width, dtype=torch.half, device="cuda")
    with pytest.raises(RuntimeError):
        # Expect failure because half type is not dispatched in the kernel.
        out = cuda_mod.forward(input_tensor, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 5 & 6: Test non-square pooling behavior and large batch*channel in z direction.
# For issue 5, although the kernel only supports square pooling window,
# the test verifies that rectangular pooling parameters (when interpreted as square) lead to a discrepancy.
def test_non_square_kernel_parameter():
    cuda_mod = build_kernel()
    batch_size, channels, height, width = 2, 3, 32, 32
    # Intentionally use values that would differ if interpreted as rectangular.
    # Since our kernel supports only an int for kernel_size,
    # we simulate a scenario where a user might expect a non-square pooling.
    kernel_size = 3  # square window but user expectation might be (3, 2) - not supported.
    stride = 2
    padding = 1
    dilation = 1

    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    expected = F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out = cuda_mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # Here we simply compare the results. A mismatch indicates that the kernel cannot handle alternate rectangular cases.
    assert torch.allclose(out, expected, atol=1e-5), f"Non-square kernel parameter test. Max diff: {(out - expected).abs().max().item()}"

# Issue 6: Test large combined batch and channel dimension.
# This test pushes the grid configuration. Even though a fixed blockDim.z is used, the result should be correct.
def test_large_batch_channel():
    cuda_mod = build_kernel()
    # Use larger batch size and channel count.
    batch_size, channels, height, width = 32, 64, 32, 32
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    expected = F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out = cuda_mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    assert torch.allclose(out, expected, atol=1e-5), f"Large batch/channel test failed. Max diff: {(out - expected).abs().max().item()}"
