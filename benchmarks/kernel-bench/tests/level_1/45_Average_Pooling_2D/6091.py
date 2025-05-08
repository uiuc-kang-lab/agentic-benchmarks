
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build/load the custom CUDA kernel extension.
def build_kernel():
    cuda_module = load(
        name="custom_avgpool_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Test that half-precision input (float16) is not supported.
def test_half_precision_not_supported():
    my_module = build_kernel()
    # Create a half-precision tensor on CUDA.
    x = torch.randn(2, 2, 8, 8, device="cuda", dtype=torch.float16)
    kernel_size = 3
    stride = 3
    padding = 1
    with pytest.raises(RuntimeError):
        # Expect a runtime error because half precision isn't dispatched.
        out = my_module.forward(x, kernel_size, stride, padding)

# Issue 2: Test that non-square pooling parameters (i.e. kernel_size and stride as tuples)
# are not supported because the kernel accepts only int.
def test_non_square_pooling_not_supported():
    my_module = build_kernel()
    x = torch.randn(2, 2, 16, 16, device="cuda", dtype=torch.float32)
    # Attempt to pass non-square kernel parameters (as tuples) should be rejected by pybind11 conversion.
    kernel_size = (3, 2)
    stride = (3, 2)
    padding = (1, 0)
    with pytest.raises(TypeError):
        out = my_module.forward(x, kernel_size, stride, padding)

# Issue 3: Test that supplying a dilation parameter is not supported.
def test_dilation_not_supported():
    my_module = build_kernel()
    x = torch.randn(2, 2, 16, 16, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 3
    padding = 1
    # The extension forward function does not expect a dilation parameter.
    with pytest.raises(TypeError):
        # Explicit keyword argument 'dilation' should cause an error.
        out = my_module.forward(x, kernel_size, stride, padding, dilation=2)

# Issue 4: Test that a zero stride is not validated and leads to an error.
def test_zero_stride():
    my_module = build_kernel()
    x = torch.randn(2, 2, 16, 16, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 0  # Invalid stride should trigger an error (likely division by zero in host calculation)
    padding = 1
    with pytest.raises(Exception):
        out = my_module.forward(x, kernel_size, stride, padding)

# Issue 5: Test the inflexible division in pooling: For positions where the pooling window is clipped
# (e.g. when padding makes part of the pooling window fall outside the input), the kernel always divides 
# by kernel_size*kernel_size. When using a PyTorch AvgPool2d with count_include_pad=False, the result would differ.
def test_count_include_pad_inflexibility():
    # This test creates a scenario where pooling windows at the border are partially out-of-bound.
    # The custom kernel always divides by (kernel_size*kernel_size) (i.e. count_include_pad=True),
    # while nn.AvgPool2d(count_include_pad=False) divides by the number of valid elements.
    my_module = build_kernel()
    # Use a small input with padding so that border windows are clipped.
    x = torch.tensor([[[[1.0, 2.0],
                        [3.0, 4.0]]]], device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 1
    padding = 1
    # Use PyTorch's AvgPool2d with count_include_pad=False
    pool_layer = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=False)
    ref_out = pool_layer(x)
    # Get custom kernel output (which implicitly does count_include_pad=True division)
    custom_out = my_module.forward(x, kernel_size, stride, padding)
    # For the top-left corner, the pooling window is missing one row and one column.
    # Hence, ref_out[0,0,0,0] is computed as sum/(# valid elements), while custom_out divides by kernel_size**2.
    # They should differ.
    assert not torch.allclose(custom_out, ref_out), (
        "Custom kernel output should differ from a count_include_pad=False pooling result."
    )
