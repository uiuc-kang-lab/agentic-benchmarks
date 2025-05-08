
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_conv2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Utility: a simple convolution wrapper that calls our CUDA kernel.
def run_conv2d(x, weight, bias, stride, padding, dilation, groups):
    module = build_kernel()
    # Our kernel's forward signature:
    # forward(x, weight, optional(bias), stride, padding, dilation, groups)
    return module.forward(x, weight, bias, stride, padding, dilation, groups)

# -----------------------------------------------------------------------------
# Test Case for Issue 1: Non-float32 tensor passed.
# We construct tensors in double precision. Even though they are CUDA and contiguous,
# the kernel always casts pointers to float*, so the computation will be garbage.
# We compare against PyTorch's convolution output in float32. They should not match.
def test_non_float_input():
    # Create inputs in float64 (double) precision.
    batch_size = 2
    in_channels = 3
    out_channels = 8
    in_height = 16
    in_width = 16
    kernel_size = (3, 5)
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    
    # Create double precision inputs and weight.
    x = torch.randn(batch_size, in_channels, in_height, in_width, dtype=torch.float64, device="cuda")
    # Weight shape as (out_channels, in_channels/groups, kH, kW)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float64, device="cuda")
    bias = None  # or torch.randn(out_channels, dtype=torch.float64, device="cuda")
    
    # Run our custom CUDA kernel.
    out = run_conv2d(x, weight, bias, stride, padding, dilation, groups)
    
    # Compare to a reference conv2d (but first cast inputs to float32).
    model = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                            dilation=dilation, groups=groups, bias=False).cuda()
    # Copy weights from our weight cast to float32.
    with torch.no_grad():
        model.weight.copy_(weight.float())
    
    ref = model(x.float())
    # Because our CUDA kernel interpreted the double data pointer as float data,
    # the results will be unpredictable. We assert that the outputs are not close.
    assert not torch.allclose(out, ref, atol=1e-3), (
        "Kernel accepted non-float32 inputs with correct result; "
        "expected undefined behavior due to missing type checks."
    )

# -----------------------------------------------------------------------------
# Test Case for Issue 2: Non-contiguous tensor input.
# Create a non-contiguous tensor (for instance, via transposition) and verify that the CHECK_CONTIGUOUS macro fails.
def test_non_contiguous_input():
    batch_size = 2
    in_channels = 3
    out_channels = 8
    in_height = 16
    in_width = 16
    kernel_size = (3, 3)
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    # Make x non-contiguous.
    x = x.transpose(2, 3)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1],
                         device="cuda", dtype=torch.float32)
    bias = None
    
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _ = run_conv2d(x, weight, bias, stride, padding, dilation, groups)

# -----------------------------------------------------------------------------
# Test Case for Issue 3: Excessively large out_channels leading to grid dimension overflow.
# We construct weight and bias tensors such that out_channels is larger than CUDA gridDim.z maximum.
# This should trigger a kernel launch failure.
def test_large_out_channels():
    batch_size = 1
    in_channels = 3
    # Set out_channels to a value greater than the typical 65535 maximum grid dimension in z.
    out_channels = 70000
    in_height = 32
    in_width = 32
    kernel_size = (3, 3)
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1],
                         device="cuda", dtype=torch.float32)
    bias = None

    # We expect the kernel launch or subsequent synchronization to fail.
    with pytest.raises(RuntimeError, match="CUDA kernel failed"):
        _ = run_conv2d(x, weight, bias, stride, padding, dilation, groups)

# -----------------------------------------------------------------------------
# Test Case for Issue 4: Excessive number of CUDA streams created for a high batch size.
# Launch the kernel with a very high batch size to stress the stream creation.
# While this might not always crash, in constrained environments it may lead to errors.
def test_high_batch_size_streams():
    # Use a relatively high batch size (this number may be adjusted depending on the available GPU resources).
    batch_size = 1024  # High batch size that may stress the stream creation.
    in_channels = 3
    out_channels = 16
    in_height = 32
    in_width = 32
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1],
                         device="cuda", dtype=torch.float32)
    bias = None
    # The expectation is that, with a very high batch size, resource exhaustion or performance degradation
    # might trigger a CUDA error during stream creation or kernel launch.
    with pytest.raises(RuntimeError, match="CUDA kernel failed"):
        _ = run_conv2d(x, weight, bias, stride, padding, dilation, groups)
