
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue with non-float32 input (e.g. double precision)
def test_non_float32_dtype():
    my_module = build_kernel()
    # Create double precision input
    batch_size = 2
    in_channels = 3
    height_in = 32
    width_in = 32
    x = torch.randn(batch_size, in_channels, height_in, width_in, dtype=torch.double, device='cuda')
    # Create weight with correct shape for depthwise convolution: [in_channels, 1, k, k]
    kernel_size = 3
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.double, device='cuda')
    # Bias is optional; here we leave it as None.
    stride = 1
    padding = 0
    with pytest.raises(RuntimeError):
        # This should fail because the kernel expects float (not double).
        my_module.forward(x, weight, None, stride, padding)

# Test 2: Trigger grid dimension issue by using a large batch*channel count.
def test_exceeding_grid_z_limit():
    my_module = build_kernel()
    # We choose batch_size and channels such that batch_size * out_channels > 65535.
    # Since for depthwise conv out_channels = in_channels * channels_per_group and for our model channels_per_group=1,
    # we set in_channels large enough.
    batch_size = 1
    in_channels = 70000  # 70000 > 65535 so batch*channels will exceed the limit.
    height_in = 32
    width_in = 32
    x = torch.randn(batch_size, in_channels, height_in, width_in, device='cuda', dtype=torch.float32)
    kernel_size = 3
    # Weight shape for depthwise convolution: [in_channels, 1, kernel_size, kernel_size]
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    stride = 1
    padding = 0
    with pytest.raises(RuntimeError):
        # The kernel launch should fail because gridDim.z exceeds the allowed maximum.
        my_module.forward(x, weight, None, stride, padding)
    # Note: Depending on the GPU/hardware, this error may be caught at kernel launch or later at synchronization.

# Test 3: Trigger issue with non-depthwise (general) convolution.
def test_non_depthwise_convolution():
    my_module = build_kernel()
    # Here we simulate a case where groups != in_channels, i.e. a non-depthwise convolution.
    batch_size = 2
    in_channels = 3
    # Let channels_per_group be 2 (so out_channels should be 6), but depthwise expects channels_per_group=1.
    channels_per_group = 2
    out_channels = in_channels * channels_per_group
    height_in = 32
    width_in = 32
    x = torch.randn(batch_size, in_channels, height_in, width_in, device='cuda', dtype=torch.float32)
    kernel_size = 3
    # Weight shape that would come from a grouped convolution but not depthwise: [in_channels, channels_per_group, k, k]
    weight = torch.randn(in_channels, channels_per_group, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    stride = 1
    padding = 0
    with pytest.raises(IndexError):
        # The kernel indexing in the CUDA code is specialized for depthwise convolution (channels_per_group==1)
        # and will likely access out-of-bound memory or compute an incorrect index.
        my_module.forward(x, weight, None, stride, padding)
    # Depending on how memory errors are reported, this test might catch an IndexError or a CUDA error.

# Test 4: Trigger issue with excessive shared memory usage.
def test_excessive_shared_memory():
    my_module = build_kernel()
    batch_size = 2
    in_channels = 3
    height_in = 64
    width_in = 64
    x = torch.randn(batch_size, in_channels, height_in, width_in, device='cuda', dtype=torch.float32)
    # Use an uncommonly large kernel size to potentially exceed the allowed shared memory per block.
    kernel_size = 15
    # Weight shape for depthwise convolution: [in_channels, 1, kernel_size, kernel_size]
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    stride = 1
    padding = 0
    with pytest.raises(RuntimeError):
        # This should trigger an error due to excessive shared memory allocation.
        my_module.forward(x, weight, None, stride, padding)
    # Note: whether this error triggers can be hardware dependent; on devices with high shared memory,
    # the error might not occur.

# Test 5: Lack of error checking after kernel launch.
def test_no_kernel_launch_error_checking():
    my_module = build_kernel()
    # This test calls the forward function with valid parameters. If the kernel launch had an error,
    # proper error checking would have caught it. In this case, we assume the input is valid and expect
    # no error. This test acts as a control to show that the kernel does not report asynchronous errors.
    batch_size = 2
    in_channels = 3
    height_in = 32
    width_in = 32
    x = torch.randn(batch_size, in_channels, height_in, width_in, device='cuda', dtype=torch.float32)
    kernel_size = 3
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    stride = 1
    padding = 0
    # Even though the kernel code does not check for errors after launch,
    # a valid configuration should run without raising errors.
    output = my_module.forward(x, weight, None, stride, padding)
    torch.cuda.synchronize()
    assert output is not None
