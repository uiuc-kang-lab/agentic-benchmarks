
import torch
import pytest
from torch.nn.functional import conv2d
from torch.utils.cpp_extension import load

# Helper to compile the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="depthwise_conv2d_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger the shared memory indexing inconsistency.
# Use kernel parameters such that tile_w != tile_w_padded.
# The output from our custom kernel is compared with PyTorch's conv2d result.
def test_shared_memory_indexing():
    # Parameters chosen to trigger the padded tile discrepancy
    batch_size = 2
    in_channels = 4
    height = 31
    width = 31
    kernel_size_h = 3
    # Use an asymmetric kernel width that is not a multiple of 4, so that padding is applied
    kernel_size_w = 5  
    stride = 1
    padding = 1
    dilation = 1
    groups = in_channels  # depthwise

    # Create input and weight tensors
    torch.manual_seed(0)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size_h, kernel_size_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Reference output using PyTorch's conv2d (depthwise)
    ref_out = conv2d(x, weight, bias, stride, padding, dilation, groups)

    # Build and call the custom kernel
    mod = build_kernel()
    out = mod.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)
    torch.cuda.synchronize()
    
    # The mismatch is expected because of shared memory indexing issues.
    assert not torch.allclose(out, ref_out, atol=1e-4), "Test should trigger shared memory indexing issue."

# Test 2: Trigger unsafe vectorized loads via misaligned data.
# We simulate misalignment by slicing the input tensor to create a non-aligned storage.
def test_vectorized_memory_load_alignment():
    batch_size = 2
    in_channels = 3
    height = 32
    width = 32
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = in_channels

    torch.manual_seed(0)
    # Create a larger tensor and then slice to force misalignment.
    x_full = torch.randn(batch_size, in_channels, height + 1, width + 1, device="cuda", dtype=torch.float32)
    x = x_full[:, :, 1:, 1:].clone()  # Removing one row and one column may result in misaligned memory.
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    ref_out = conv2d(x, weight, bias, stride, padding, dilation, groups)
    mod = build_kernel()
    out = mod.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)
    torch.cuda.synchronize()
    
    # The misaligned memory loads can cause incorrect results.
    assert not torch.allclose(out, ref_out, atol=1e-4), "Test should trigger float4 load alignment issue."

# Test 3: Trigger issues from rigid tiling assumptions.
# Use input dimensions which are not multiples of TILE_SIZE to force boundary conditions.
def test_rigid_tiling_assumptions():
    batch_size = 1
    in_channels = 3
    height = 45  # non-multiple of TILE_SIZE (16)
    width = 37   # non-multiple of TILE_SIZE (16)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    groups = in_channels

    torch.manual_seed(42)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    ref_out = conv2d(x, weight, bias, stride, padding, dilation, groups)
    mod = build_kernel()
    out = mod.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)
    torch.cuda.synchronize()
    
    # Expect differences due to tile mapping issues
    assert not torch.allclose(out, ref_out, atol=1e-4), "Test should trigger tiling boundary issue."

# Test 4: Trigger problems with hardcoded batch/channel mapping.
# Use a grouping arrangement that is not depthwise.
def test_hardcoded_batch_channel_mapping():
    # Here, groups is not equal to in_channels.
    batch_size = 1
    in_channels = 4
    out_channels = 8
    height = 16
    width = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 2  # Not equal to in_channels (=> standard grouped convolution)

    torch.manual_seed(7)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # weight shape for grouped conv: (out_channels, in_channels//groups, kernel_h, kernel_w)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Using torch's conv2d as a reference:
    ref_out = conv2d(x, weight, bias, stride, padding, dilation, groups)

    mod = build_kernel()
    # The custom kernel is designed only for depthwise (groups==in_channels), so this call should produce errors or wrong results.
    out = mod.forward(x, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)
    torch.cuda.synchronize()
    
    # Expect the result to be different from the reference for non-depthwise case.
    assert not torch.allclose(out, ref_out, atol=1e-4), "Test should trigger batch/channel mapping issue."

# Test 5: Trigger potential issues with #pragma unroll on variable loop bounds.
# Using a kernel size that is not known at compile time may trigger unexpected loop unrolling.
def test_unroll_variable_loops():
    batch_size = 1
    in_channels = 3
    height = 32
    width = 32
    kernel_size_h = 3
    # Use a kernel width different from kernel height.
    kernel_size_w = 7  
    stride = 1
    padding_h = 1
    padding_w = 2
    dilation = 1
    groups = in_channels

    torch.manual_seed(99)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, 1, kernel_size_h, kernel_size_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    ref_out = conv2d(x, weight, bias, (stride, stride), (padding_h, padding_w), (dilation, dilation), groups)
    mod = build_kernel()
    out = mod.forward(x, weight, bias, stride, stride, padding_h, padding_w, dilation, dilation, groups)
    torch.cuda.synchronize()
    
    # Expect differences because of incorrect unrolling over variable loop bounds.
    assert not torch.allclose(out, ref_out, atol=1e-4), "Test should trigger unroll variable loop issue."
