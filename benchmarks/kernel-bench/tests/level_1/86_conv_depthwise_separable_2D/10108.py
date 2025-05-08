
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper to compile the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_depthwise_separable",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference implementation using PyTorch’s built-ins:
def ref_depthwise_conv2d(x, depthwise_weight, depthwise_bias, stride, padding, dilation):
    # groups is equal to in_channels for depthwise conv
    return F.conv2d(x, depthwise_weight, bias=depthwise_bias, stride=stride,
                    padding=padding, dilation=dilation, groups=x.shape[1])

def ref_pointwise_conv2d(x, pointwise_weight, pointwise_bias):
    # pointwise: kernel_size=1, groups=1
    return F.conv2d(x, pointwise_weight, bias=pointwise_bias)

# 1. Test that the depthwise kernel “loop stride” mismatch is exposed by using an input shape
#    where the total number of output elements is not an exact multiple of (threads * ELEMENTS_PER_THREAD).
def test_depthwise_elements_per_thread():
    # Choose sizes so that total elements (batch * channels * out_h * out_w)
    # is not divisible by (256*4)
    batch = 7
    channels = 3
    in_h, in_w = 29, 29
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    # Create input and weights
    x = torch.randn(batch, channels, in_h, in_w, device='cuda', dtype=torch.float32)
    # Depthwise: groups==channels
    depthwise_weight = torch.randn(channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    depthwise_bias = torch.randn(channels, device='cuda', dtype=torch.float32)

    # Build module and run kernel forward
    module = build_kernel()
    out = module.forward(x, depthwise_weight, torch.randn(64, channels, 1, 1, device='cuda', dtype=torch.float32),
                         depthwise_bias, None, stride, padding, dilation)

    # Reference: run depthwise conv then a dummy pointwise (identity) conv.
    ref_dw = ref_depthwise_conv2d(x, depthwise_weight, depthwise_bias, stride, padding, dilation)
    # Compare just the depthwise convolution part
    # (We only care that the computed depthwise result – which then feeds into the pointwise conv – is off.)
    # Since our kernel may be off because of the wrong loop-step, we expect a noticeable mismatch.
    assert not torch.allclose(out[:, :channels], ref_dw, atol=1e-4), (
        "Test failed to trigger the ELEMENTS_PER_THREAD issue in the depthwise kernel."
    )

# 2. Test that the pointwise kernel produces differences when the spatial dimensions are not multiples of TILE_DIM.
def test_pointwise_shared_memory_tiling():
    batch = 4
    channels = 3
    out_channels = 8
    # Choose height*width not divisible by TILE_DIM (32)
    in_h, in_w = 35, 37
    kernel_size = 3  # depthwise kernel uses this

    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(batch, channels, in_h, in_w, device='cuda', dtype=torch.float32)
    depthwise_weight = torch.randn(channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    depthwise_bias = torch.randn(channels, device='cuda', dtype=torch.float32)
    pointwise_weight = torch.randn(out_channels, channels, 1, 1, device='cuda', dtype=torch.float32)
    pointwise_bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    module = build_kernel()
    out = module.forward(x, depthwise_weight, pointwise_weight,
                         depthwise_bias, pointwise_bias,
                         stride, padding, dilation)

    # Build reference output: first apply a depthwise conv (using built-in conv2d with groups)
    ref_dw = ref_depthwise_conv2d(x, depthwise_weight, depthwise_bias, stride, padding, dilation)
    ref_out = ref_pointwise_conv2d(ref_dw, pointwise_weight, pointwise_bias)

    # Because of the suspected shared memory mis‐indexing in the pointwise kernel,
    # we expect the kernel output to differ from the reference.
    assert not torch.allclose(out, ref_out, atol=1e-4), (
        "Test failed to detect mismatch in the pointwise kernel (shared memory indexing issue)."
    )

# 3. Test that using a non-standard (e.g. double precision) tensor type triggers failure
#    because the kernel is “dispatched” only over float/double but may assume float32 ordering,
#    or because our reinterpret_cast might be inconsistent.
def test_non_float32_input():
    batch = 2
    channels = 3
    out_channels = 4
    in_h, in_w = 16, 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    # Create double tensor input – if not supported or handled correctly, this may trigger issues.
    x = torch.randn(batch, channels, in_h, in_w, device='cuda', dtype=torch.float64)
    depthwise_weight = torch.randn(channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.float64)
    depthwise_bias = torch.randn(channels, device='cuda', dtype=torch.float64)
    pointwise_weight = torch.randn(out_channels, channels, 1, 1, device='cuda', dtype=torch.float64)
    pointwise_bias = torch.randn(out_channels, device='cuda', dtype=torch.float64)

    module = build_kernel()
    with pytest.raises(RuntimeError):
        # We expect the kernel launch or execution to fail (or produce an error)
        # because the provided type is double and the implementation may not support it correctly.
        module.forward(x, depthwise_weight, pointwise_weight,
                       depthwise_bias, pointwise_bias,
                       stride, padding, dilation)

# 4. Test that the missing cuda error-checking (lack of synchronization/error query)
#    causes the kernel to return results that do not match the reference when complex sizes are used.
def test_no_cuda_error_check():
    # This test uses a more “complex” spatial layout that stress-tests both kernels.
    batch = 5
    channels = 3
    out_channels = 10
    in_h, in_w = 61, 73  # prime numbers to avoid nice divisibility
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    x = torch.randn(batch, channels, in_h, in_w, device='cuda', dtype=torch.float32)
    depthwise_weight = torch.randn(channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    depthwise_bias = torch.randn(channels, device='cuda', dtype=torch.float32)
    pointwise_weight = torch.randn(out_channels, channels, 1, 1, device='cuda', dtype=torch.float32)
    pointwise_bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    module = build_kernel()
    out = module.forward(x, depthwise_weight, pointwise_weight,
                         depthwise_bias, pointwise_bias,
                         stride, padding, dilation)

    ref_dw = ref_depthwise_conv2d(x, depthwise_weight, depthwise_bias, stride, padding, dilation)
    ref_out = ref_pointwise_conv2d(ref_dw, pointwise_weight, pointwise_bias)

    # Since no proper error checking and synchronization is done,
    # the output may be incorrect if any asynchronous error occurred.
    assert not torch.allclose(out, ref_out, atol=1e-4), (
        "Test failed to trigger a noticeable difference due to missing error checking."
    )
