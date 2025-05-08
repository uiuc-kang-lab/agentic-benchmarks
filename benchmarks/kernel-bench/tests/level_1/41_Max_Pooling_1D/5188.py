
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Utility to build the CUDA extension.
def build_kernel():
    # Force rebuild to use the local kernel.cu file.
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# --------------------------
# Test 1: Passing non-float32 input should trigger an error
# (Issue 1)
def test_input_type():
    cuda_module = build_kernel()
    # Create an input tensor of type float64 intentionally.
    batch_size, channels, length = 2, 4, 16
    x = torch.randn(batch_size, channels, length, dtype=torch.float64, device='cuda')
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False

    with pytest.raises(RuntimeError):
        # This should fail because the kernel code calls x.data_ptr<float>()
        cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)

# --------------------------
# Test 2: Concurrent kernel launches with different configurations might interfere via constant memory
# (Issue 2)
def test_concurrent_kernel_calls():
    cuda_module = build_kernel()
    batch_size, channels, length = 2, 4, 32
    x = torch.randn(batch_size, channels, length, dtype=torch.float32, device='cuda')

    # Use two different configurations
    kernel_size1 = 3
    stride1 = 1
    padding1 = 1
    dilation1 = 1

    kernel_size2 = 5
    stride2 = 2
    padding2 = 2
    dilation2 = 2

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # Launch first kernel call in stream1
    with torch.cuda.stream(stream1):
        out1 = cuda_module.forward(x, kernel_size1, stride1, padding1, dilation1, False)
    # Launch second kernel call in stream2
    with torch.cuda.stream(stream2):
        out2 = cuda_module.forward(x, kernel_size2, stride2, padding2, dilation2, False)
    # Synchronize streams
    stream1.synchronize()
    stream2.synchronize()

    # If constant memory is overwritten by concurrent calls,
    # the outputs may be incorrect.
    # We check that the output shapes correspond to the expected configurations.
    output_length1 = ((length + 2*padding1 - dilation1*(kernel_size1-1) - 1) // stride1) + 1
    output_length2 = ((length + 2*padding2 - dilation2*(kernel_size2-1) - 1) // stride2) + 1

    assert out1.shape == (batch_size, channels, output_length1), f"Output shape mismatch for first kernel call, got {out1.shape}"
    assert out2.shape == (batch_size, channels, output_length2), f"Output shape mismatch for second kernel call, got {out2.shape}"

# --------------------------
# Test 3: When return_indices is true, the kernel returns concatenated output instead of a tuple
# (Issue 3)
def test_return_indices_format():
    cuda_module = build_kernel()
    batch_size, channels, length = 2, 4, 32
    x = torch.randn(batch_size, channels, length, dtype=torch.float32, device='cuda')
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = True

    out = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    
    # Expecting typical behavior: a tuple (output, indices). However our kernel returns
    # a concatenated tensor if indices are requested.
    # We trigger this issue by checking if the last dimension is doubled.
    output_length = ((length + 2*padding - dilation*(kernel_size-1) - 1) // stride) + 1

    # The concatenated tensor should have shape:
    # (batch_size, channels, output_length * 2)
    expected_shape = (batch_size, channels, output_length * 2)
    assert out.shape == expected_shape, f"Return indices tensor shape incorrect, expected {expected_shape} but got {out.shape}"

# --------------------------
# Test 4: Check that the device constant -INFINITY is correctly defined
# (Issue 4)
def test_negative_infinity():
    # This test creates an input tensor whose all values are very negative,
    # so that the max pooling should always pick one of these very negative numbers.
    # If -INFINITY is not set correctly, the result might be incorrect.
    cuda_module = build_kernel()
    batch_size, channels, length = 1, 1, 10
    x = (torch.randn(batch_size, channels, length, dtype=torch.float32, device='cuda') - 1000)
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False

    out = cuda_module.forward(x, kernel_size, stride, padding, dilation, return_indices)

    # Since the window always has finite numbers (even if very low),
    # the output value must be greater than -1e30 (a sanity check for -INFINITY mishandling).
    assert (out > -1e30).all(), "Kernel may be mishandling -INFINITY or initial value in max pooling."

if __name__ == '__main__':
    pytest.main([__file__])
