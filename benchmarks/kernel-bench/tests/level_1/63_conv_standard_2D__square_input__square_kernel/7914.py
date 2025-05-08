
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

custom_conv = build_kernel()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_kernel_size_issue():
    # Issue 1: Kernel hard-coded to kernel size 3.
    # Create a convolution with kernel_size = 5.
    batch_size, in_channels, out_channels = 1, 3, 3
    kernel_size = 5
    stride, padding, dilation, groups = 1, 2, 1, 1
    height = width = 40  # ensure input is larger than MIN_SIZE_THRESHOLD
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    
    # Call the custom kernel -> it always assumes kernel size 3.
    out_ext = custom_conv.forward(x, weight, None, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, None, stride, padding, dilation, groups)
    
    # Expect the outputs to differ.
    assert not torch.allclose(out_ext, out_ref, atol=1e-4), \
        "Kernel size issue not triggered: outputs match unexpectedly for kernel size != 3."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dilation_issue():
    # Issue 2: Dilation parameter is ignored.
    batch_size, in_channels, out_channels = 1, 3, 3
    kernel_size = 3  # our kernel always assumes 3
    stride, padding, dilation, groups = 1, 1, 2, 1  # non-default dilation
    height = width = 40
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    
    out_ext = custom_conv.forward(x, weight, None, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, None, stride, padding, dilation, groups)
    
    # Because dilation is ignored in the custom kernel, the results should differ.
    assert not torch.allclose(out_ext, out_ref, atol=1e-4), \
        "Dilation issue not triggered: custom kernel output matches reference output despite dilation != 1."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_groups_issue():
    # Issue 3: Groups parameter is ignored.
    batch_size = 1
    in_channels = 4  # must be divisible by groups.
    out_channels = 4
    groups = 2           # grouped convolution: each group has 2 channels.
    kernel_size = 3      # our kernel only supports kernel size 3
    stride, padding, dilation = 1, 1, 1
    height = width = 40
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    
    out_ext = custom_conv.forward(x, weight, None, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, None, stride, padding, dilation, groups)
    
    # Since groups are not handled, the results will differ.
    assert not torch.allclose(out_ext, out_ref, atol=1e-4), \
        "Groups issue not triggered: custom kernel output matches reference output despite groups != 1."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_stride_issue():
    # Issue 4: Incorrect use of stride indexing in shared memory accumulation.
    batch_size, in_channels, out_channels = 1, 3, 3
    kernel_size = 3  # correct kernel size for the kernel
    stride, padding, dilation, groups = 2, 0, 1, 1  # non-unit stride
    height = width = 40  # ensure output dimensions are valid and exceed fallback thresholds
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    
    out_ext = custom_conv.forward(x, weight, None, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, None, stride, padding, dilation, groups)
    
    # The mismapped shared memory indices should produce differing outputs.
    assert not torch.allclose(out_ext, out_ref, atol=1e-4), \
        "Stride issue not triggered: outputs match despite non-unit stride."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dtype_issue():
    # Issue 5: Kernel only supports float32.
    batch_size, in_channels, out_channels = 1, 3, 3
    kernel_size = 3
    stride, padding, dilation, groups = 1, 1, 1, 1
    height = width = 40
    # Create input and weight in float64.
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float64)
    
    # Though PyTorch's conv2d works with float64, our kernel uses data_ptr<float>(), so behavior is undefined.
    out_ext = custom_conv.forward(x, weight, None, stride, padding, dilation, groups)
    out_ref = F.conv2d(x, weight, None, stride, padding, dilation, groups)
    
    # The outputs are expected to differ (or be wrong) because of the type mismatch.
    assert not torch.allclose(out_ext.to(torch.float64), out_ref, atol=1e-4), \
        "Dtype issue not triggered: outputs match even though input is float64."

# Note: We are not testing Issue 6 regarding the fallback thresholds because that design choice
# forces the kernel to delegate most realistic configurations to torch::conv2d.
