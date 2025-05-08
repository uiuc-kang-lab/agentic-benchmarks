
import torch
import pytest
from torch import nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Build the kernel CUDA extension.
def build_kernel():
    cuda_module = load(
        name="custom_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper model which mimics depthwise separable convolution.
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise_weight = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size, kernel_size, device='cuda'))
        if bias:
            self.depthwise_bias = nn.Parameter(
                torch.randn(in_channels, device='cuda'))
        else:
            self.register_parameter('depthwise_bias', None)
        self.pointwise_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, 1, 1, device='cuda'))
        if bias:
            self.pointwise_bias = nn.Parameter(
                torch.randn(out_channels, device='cuda'))
        else:
            self.register_parameter('pointwise_bias', None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        # Depthwise conv, groups = in_channels:
        x = F.conv2d(x, self.depthwise_weight, self.depthwise_bias,
                     stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=x.size(1))
        # Pointwise conv:
        x = F.conv2d(x, self.pointwise_weight, self.pointwise_bias)
        return x

# Test 1: Depthwise kernel ELEMENTS_PER_THREAD mismatch.
# Use an input size that forces total_depthwise to be a number that is not a multiple
# of THREADS_PER_BLOCK * ELEMENTS_PER_THREAD.
def test_depthwise_elements_thread_mismatch():
    # Choose spatial dims so that (batch * channels * out_h * out_w) is non-multiple.
    batch = 3
    in_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    height = 17  # odd number to force output dimension irregularity
    width = 19
    
    # Build input
    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    
    # Create model and get reference output using PyTorch convs.
    model_ref = DepthwiseSeparableConv(in_channels, 8, kernel_size,
                                       stride=stride, padding=padding,
                                       dilation=dilation, bias=True).eval()
    
    # Use same weights for our custom kernel call.
    depthwise_weight = model_ref.depthwise_weight.detach()
    pointwise_weight = model_ref.pointwise_weight.detach()
    depthwise_bias = model_ref.depthwise_bias.detach() if model_ref.depthwise_bias is not None else torch.tensor([])
    pointwise_bias = model_ref.pointwise_bias.detach() if model_ref.pointwise_bias is not None else torch.tensor([])
    
    # Get reference output.
    ref_out = model_ref(x)
    
    # Run our custom CUDA kernel.
    custom_conv = build_kernel()
    # forward_wrapper(name in extension is "forward")
    custom_out = custom_conv.forward(x, depthwise_weight, pointwise_weight,
                                     depthwise_bias, pointwise_bias,
                                     stride, padding, dilation)
    torch.cuda.synchronize()
    
    # Expect differences due to mis-handled multi-element per thread loop.
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), (
        "Test failed to trigger depthwise kernel ELEMENTS_PER_THREAD issue.")

# Test 2: Depthwise kernel pragma unroll with non constant kernel size.
# Use an unusual kernel size (e.g., 5) for which unrolling may be problematic.
def test_depthwise_unroll_non_constant_kernel():
    batch = 2
    in_channels = 3
    kernel_size = 5  # Non-standard size for which unrolling may not work well
    stride = 1
    padding = 2
    dilation = 1
    height = 32
    width = 32

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    
    model_ref = DepthwiseSeparableConv(in_channels, 4, kernel_size,
                                       stride=stride, padding=padding,
                                       dilation=dilation, bias=False).eval()
    depthwise_weight = model_ref.depthwise_weight.detach()
    pointwise_weight = model_ref.pointwise_weight.detach()
    depthwise_bias = torch.tensor([], device="cuda")
    pointwise_bias = torch.tensor([], device="cuda")

    ref_out = model_ref(x)
    
    custom_conv = build_kernel()
    custom_out = custom_conv.forward(x, depthwise_weight, pointwise_weight,
                                     depthwise_bias, pointwise_bias,
                                     stride, padding, dilation)
    torch.cuda.synchronize()

    # Expect differences if unroll causes code generation issues.
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), (
        "Test failed to trigger depthwise kernel unroll issue with non constant kernel size.")

# Test 3: Pointwise kernel spatial tiling issue.
# Use spatial dimensions such that out_h * out_w is not a multiple of TILE_DIM (32)
def test_pointwise_spatial_tiling():
    batch = 2
    in_channels = 4
    out_channels = 5
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    # Use spatial size that yields an output whose total pixels is not a multiple of 32.
    height = 23
    width = 27

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    
    model_ref = DepthwiseSeparableConv(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding,
                                       dilation=dilation, bias=True).eval()
    depthwise_weight = model_ref.depthwise_weight.detach()
    pointwise_weight = model_ref.pointwise_weight.detach()
    depthwise_bias = model_ref.depthwise_bias.detach() if model_ref.depthwise_bias is not None else torch.tensor([])
    pointwise_bias = model_ref.pointwise_bias.detach() if model_ref.pointwise_bias is not None else torch.tensor([])

    ref_out = model_ref(x)
    
    custom_conv = build_kernel()
    custom_out = custom_conv.forward(x, depthwise_weight, pointwise_weight,
                                     depthwise_bias, pointwise_bias,
                                     stride, padding, dilation)
    torch.cuda.synchronize()

    # Expect differences if the shared memory spatial tiling fails.
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), (
        "Test failed to trigger pointwise kernel spatial tiling issue.")

# Test 4: Pointwise kernel in_channels tiling issue.
# Use an in_channels value that is not a multiple of TILE_DIM (32).
def test_pointwise_inchannels_tiling():
    batch = 1
    in_channels = 7  # not a multiple of TILE_DIM=32
    out_channels = 3
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    height = 64
    width = 64

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float32)
    
    model_ref = DepthwiseSeparableConv(in_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding,
                                       dilation=dilation, bias=False).eval()
    depthwise_weight = model_ref.depthwise_weight.detach()
    pointwise_weight = model_ref.pointwise_weight.detach()
    depthwise_bias = torch.tensor([], device="cuda")
    pointwise_bias = torch.tensor([], device="cuda")

    ref_out = model_ref(x)
    
    custom_conv = build_kernel()
    custom_out = custom_conv.forward(x, depthwise_weight, pointwise_weight,
                                     depthwise_bias, pointwise_bias,
                                     stride, padding, dilation)
    torch.cuda.synchronize()

    # Expect differences if the in_channels tiling in shared memory is mishandled.
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), (
        "Test failed to trigger pointwise kernel in_channels tiling issue.")
