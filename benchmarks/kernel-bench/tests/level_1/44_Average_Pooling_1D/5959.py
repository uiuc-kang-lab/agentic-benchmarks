
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="manual_unroll_avg_pool1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue for unsupported data type (non–float32)
def test_dtype_support():
    # Create a tensor of type float64. The kernel expects float32.
    x = torch.randn(16, 32, 128, device="cuda", dtype=torch.float64)
    kernel_size, stride, padding = 3, 1, 0
    module = build_kernel()
    with pytest.raises((RuntimeError, AssertionError)):
        # This should either produce an error or give an incorrect result so we catch the exception.
        _ = module.forward(x, kernel_size, stride, padding)
    # Note: behavior is undefined; we expect an error due to reinterpretation of double data as float.

# Test 2: Trigger issue with large grid dimensions (exceeding CUDA grid limits)
def test_large_grid_dims():
    # Set in_channels large enough to cause grid.y > 65,535 (e.g., 70000 channels)
    batch_size = 2
    in_channels = 70000
    input_length = 20   # small input_length so that output_length is also small
    kernel_size = 3
    stride = 1
    padding = 1
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Launching the kernel should fail due to grid dimension limitations.
        _ = module.forward(x, kernel_size, stride, padding)
    # If the kernel does not properly use grid-stride loop for channels,
    # the grid dimension in y may exceed the device limit.

# Test 3: Trigger issue with lack of kernel launch error checking.
def test_kernel_launch_error_checking():
    # Forcing a kernel launch error by passing an input with invalid dimensionality.
    # The kernel requires a 3D tensor.
    x = torch.randn(16, 128, device="cuda", dtype=torch.float32)  # 2D tensor instead of 3D
    kernel_size, stride, padding = 3, 1, 0
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(x, kernel_size, stride, padding)
    # The TORCH_CHECK in the host function should trigger an error.

# Test 4: Trigger issue with non–contiguous input memory.
def test_non_contiguous_input():
    # Create a contiguous tensor and then create a non-contiguous version via transpose.
    x = torch.randn(16, 32, 128, device="cuda", dtype=torch.float32)
    # Make input non-contiguous by permuting dimensions and then reverting order.
    x_noncontig = x.transpose(0, 1).transpose(0, 1)
    # Confirm that the tensor is non-contiguous.
    assert not x_noncontig.is_contiguous(), "Tensor should be non-contiguous for this test."
    
    kernel_size, stride, padding = 3, 1, 1
    module = build_kernel()

    # Compute output with the kernel (which does not check contiguity)
    out_kernel = module.forward(x_noncontig, kernel_size, stride, padding)
    # Compute expected output using PyTorch's AvgPool1d on a contiguous tensor.
    avg_pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    out_ref = avg_pool(x_noncontig.contiguous())
    # The outputs may differ because the kernel assumed contiguous layout.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), "Non-contiguous input did not trigger an incorrect result as expected."

# Test 5: Validate behavior with kernel sizes not specially unrolled (e.g. kernel_size != 3 or 5)
def test_generic_kernel_size():
    # Use a kernel size that is not 3 or 5.
    batch_size = 16
    in_channels = 32
    input_length = 128
    kernel_size = 4  # generic case
    stride = 2
    padding = 1

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    out_kernel = module.forward(x, kernel_size, stride, padding)
    avg_pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    out_ref = avg_pool(x)
    
    # Here we expect the results to be similar. A significant difference might indicate that
    # the generic loop (without manual unrolling) is not handling the kernel correctly.
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), "Output from generic kernel size does not match PyTorch's AvgPool1d."

