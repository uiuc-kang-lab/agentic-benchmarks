
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test for input tensor type mismatch (non-float32 input should trigger an error)
def test_dtype_mismatch():
    module = build_kernel()
    batch_size = 2
    in_channels = 4
    length = 10
    # Create double-precision (float64) input, weight and bias.
    x = torch.randn(batch_size, in_channels, length, dtype=torch.float64, device="cuda")
    weight = torch.randn(in_channels, 5, 3, dtype=torch.float64, device="cuda")
    bias = torch.randn(5, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError):
        # This call should fail when the kernel (which only supports float32) is invoked.
        module.forward(x, weight, bias, 1, 0, 1)

# Issue 2: Test for shared memory overuse due to redundant per-block load of weights.
# The test creates a weight tensor so large that the per-block shared memory allocation exceeds available limits.
def test_shared_memory_overuse():
    module = build_kernel()
    batch_size = 2
    in_channels = 256
    out_channels = 256
    kernel_size = 15  # Large kernel size to force enormous shared memory usage.
    length = 20
    # The shared memory per block will be: in_channels*out_channels*kernel_size*sizeof(float)
    # which is likely to exceed the device's available shared memory.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        output = module.forward(x, weight, bias, 1, 0, 1)
        # Ensure that any asynchronous error is reported.
        torch.cuda.synchronize()

# Issue 3: Test for potential incorrect handling of negative indices due to modulo operation.
# We set parameters so that for some kernel positions, i_pos becomes negative.
def test_negative_index_handling():
    module = build_kernel()
    batch_size = 1
    in_channels = 2
    out_channels = 2
    kernel_size = 3
    length = 5
    stride = 1
    padding = 0
    dilation = 3  # With dilation 3 and kernel_size 3, some computed i_pos will be negative.
    
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Compute expected output using PyTorch's native ConvTranspose1d.
    conv_transpose = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=True
    )
    conv_transpose.weight.data.copy_(weight)
    conv_transpose.bias.data.copy_(bias)
    expected = conv_transpose(x)
    
    output = module.forward(x, weight, bias, stride, padding, dilation)
    torch.cuda.synchronize()
    
    # We expect a discrepancy due to the modulo-driven mishandling of negative indices.
    # The test asserts that the kernel's output does not match the expected correct behavior.
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Kernel output unexpectedly matches expected result despite potential negative index handling issues."
