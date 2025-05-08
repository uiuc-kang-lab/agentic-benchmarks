
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu.
# It is assumed that kernel.cu is in the same directory as this test file.
def build_kernel():
    module = load(
        name="custom_maxpool",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Loop unrolling using a runtime kernel_size.
# Use a relatively large and non‐standard kernel size so that loop unrolling
# might not be optimized as expected. The custom kernel output is compared against
# PyTorch's nn.MaxPool2d output. If there is a miscompilation due to unroll,
# the outputs could differ.
def test_loop_unroll_issue():
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 3, 20, 20
    # Use kernel parameters where kernel_size is far from a typical constant.
    kernel_size = 10
    stride = 3
    padding = 2
    dilation = 2
    # Create an input tensor
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    
    # Get reference output from PyTorch built-in MaxPool2d
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    ref_output = pool(input_tensor)
    
    # Get output from our custom CUDA kernel via the built extension.
    mod = build_kernel()
    custom_output = mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    
    # It is expected that the two outputs match.
    # If there is an issue with the unrolling logic, they will differ.
    assert torch.allclose(custom_output, ref_output, atol=1e-4), \
        f"Loop unroll issue: custom output doesn't match reference (max diff = {(custom_output - ref_output).abs().max().item()})"

# Issue 2: Using std::numeric_limits<scalar_t>::infinity() on device.
# This may fail or produce unexpected results. Here, we try using an input in half precision.
def test_device_infinity_issue():
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 3, 20, 20
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    # Create a half precision input; note that our AT_DISPATCH_FLOATING_TYPES does not cover half.
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float16)
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel does not dispatch half.
        _ = mod.forward(input_tensor, kernel_size, stride, padding, dilation)

# Issue 3: Ambiguous use of max() on device.
# Using input containing NaN values may trigger unexpected behavior
# if the device max function is not handling floating point comparisons as expected.
def test_max_function_issue():
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 3, 10, 10
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    # Create input with a NaN inserted in each pooling window in a predictable way.
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    # Set one element arbitrarily to NaN in a window
    input_tensor[:, :, 5, 5] = float('nan')
    
    # Get reference output using PyTorch built-in with max pooling.
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    ref_output = pool(input_tensor)
    
    mod = build_kernel()
    custom_output = mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    
    # If the max function in the kernel is misbehaving the NaN might not be propagated correctly.
    nan_in_ref = torch.isnan(ref_output).any()
    nan_in_custom = torch.isnan(custom_output).any()
    
    # We expect consistent behavior: either both propagate NaN or not.
    assert nan_in_custom == nan_in_ref, \
        "Max function issue: Differing NaN propagation between custom kernel and PyTorch's maxpool."

# Issue 4: Kernel supports only float32/double.
# Pass an integer tensor to trigger an error.
def test_integer_input_issue():
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 3, 10, 10
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    # Create integer input tensor.
    input_tensor = torch.randint(0, 10, (batch_size, channels, height, width), device="cuda", dtype=torch.int32)
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        _ = mod.forward(input_tensor, kernel_size, stride, padding, dilation)

# Issue 5: Assumption of contiguous NCHW layout.
# Provide a non–contiguous tensor (e.g. by transposing) to see if the kernel produces wrong results.
def test_non_contiguous_input_issue():
    torch.manual_seed(42)
    batch_size, channels, height, width = 2, 3, 20, 20
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    # Create a contiguous input tensor then force a non-contiguous layout by transposing channels.
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    non_contig = x.transpose(1, 2)  # now shape becomes (batch_size, height, channels, width) and non contiguous
    
    mod = build_kernel()
    # Our custom kernel expects a layout of NCHW, but we pass a non contiguous tensor.
    custom_output = mod.forward(non_contig, kernel_size, stride, padding, dilation)
    
    # Compute reference using PyTorch max pooling on a contiguous tensor.
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    # Force non contiguous input into the expected (contiguous) layout by transposing back.
    ref_output = pool(non_contig.transpose(1,2))
    
    # The results are likely to differ because the indexing in the kernel does not handle non–contiguous memory.
    # We deliberately check that the outputs do NOT match.
    if torch.allclose(custom_output, ref_output, atol=1e-4):
        pytest.fail("Non contiguous input issue: custom kernel output matches reference despite non contiguous layout.")

