
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper function to compute average pooling in Python (count_include_pad=True)
def avg_pool1d_reference(x, kernel_size, stride, padding):
    # x shape: (batch_size, channels, input_length)
    batch_size, channels, input_length = x.shape
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    y = torch.empty((batch_size, channels, output_length), device=x.device, dtype=x.dtype)
    for b in range(batch_size):
        for c in range(channels):
            for o in range(output_length):
                sum_val = 0.0
                for k in range(kernel_size):
                    pos = o * stride + k - padding
                    if 0 <= pos < input_length:
                        sum_val += x[b, c, pos].item()
                    else:
                        sum_val += 0.0
                y[b, c, o] = sum_val / kernel_size
    return y

# Test case for issue 1: Kernel dtype limitation (only float32 is supported)
def test_dtype_issue():
    # Create a tensor with double precision
    batch_size, channels, input_length = 2, 3, 16
    x = torch.randn(batch_size, channels, input_length, device="cuda", dtype=torch.float64)
    kernel_size = 3
    stride = 1
    padding = 1
    cuda_module = build_kernel()
    # Expect the kernel to either throw an error or produce incorrect result.
    with pytest.raises(RuntimeError):
        # Try invoking the CUDA kernel (it will call data_ptr<float>(), so mismatch in dtype should cause an error)
        _ = cuda_module.forward(x, kernel_size, stride, padding)

# Test case for issue 2: Non-contiguous tensor handling
def test_noncontiguous_input():
    batch_size, channels, input_length = 4, 8, 32
    kernel_size = 4
    stride = 2
    padding = 1
    # Create a contiguous tensor and then make it non-contiguous by transposing a couple of dimensions and back
    x = torch.randn(batch_size, channels, input_length, device="cuda", dtype=torch.float32)
    # Force non-contiguity: for example, by transposing channel and input_length dimensions and then transposing back
    x_noncontig = x.transpose(1, 2).transpose(1, 2)
    assert not x_noncontig.is_contiguous(), "Test setup error: tensor is contiguous."
    
    cuda_module = build_kernel()
    
    # Run the custom kernel on non-contiguous input
    y_custom = cuda_module.forward(x_noncontig, kernel_size, stride, padding)
    # Run the reference using PyTorch default pooling (which requires contiguous input, so we use x)
    y_ref = torch.nn.functional.avg_pool1d(x, kernel_size=kernel_size, stride=stride, padding=padding)
    
    # The custom kernel is not designed to handle non-contiguous tensors correctly.
    # Thus, the output is expected to be different from the reference.
    if torch.allclose(y_custom, y_ref, atol=1e-5):
        pytest.fail("Kernel unexpectedly produced correct results for non-contiguous input.")

# Test case for issue 3: Lack of count_include_pad flexibility
def test_count_include_pad_issue():
    batch_size, channels, input_length = 2, 3, 20
    kernel_size = 5
    stride = 2
    padding = 2
    # Create a tensor with float32, and then use non-default behavior of count_include_pad=False
    # PyTorch's nn.AvgPool1d supports count_include_pad=False but our kernel always divides by kernel_size.
    x = torch.randn(batch_size, channels, input_length, device="cuda", dtype=torch.float32)
    
    # Expected output using count_include_pad=False (manually compute)
    # We simulate what torch.nn.AvgPool1d would do when count_include_pad is False.
    def avg_pool1d_count_exclude(x, kernel_size, stride, padding):
        batch_size, channels, input_length = x.shape
        output_length = (input_length + 2 * padding - kernel_size) // stride + 1
        y = torch.empty((batch_size, channels, output_length), device=x.device, dtype=x.dtype)
        for b in range(batch_size):
            for c in range(channels):
                for o in range(output_length):
                    sum_val = 0.0
                    count = 0
                    for k in range(kernel_size):
                        pos = o * stride + k - padding
                        if 0 <= pos < input_length:
                            sum_val += x[b, c, pos].item()
                            count += 1
                    # Avoid division by zero for unusual cases.
                    y[b, c, o] = sum_val / count if count > 0 else 0.0
        return y
    
    y_expected = avg_pool1d_count_exclude(x, kernel_size, stride, padding)
    cuda_module = build_kernel()
    # Our kernel only supports count_include_pad=True behavior.
    y_custom = cuda_module.forward(x, kernel_size, stride, padding)
    
    # The outputs should differ because our kernel divides by kernel_size regardless of count.
    if torch.allclose(y_custom, y_expected, atol=1e-5):
        pytest.fail("Kernel unexpectedly produced results matching count_include_pad=False behavior.")
