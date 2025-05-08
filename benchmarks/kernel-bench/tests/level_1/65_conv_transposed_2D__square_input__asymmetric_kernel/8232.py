
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Output_padding is ignored. We'll use a nonzero output_padding and compare against PyTorch's nn.ConvTranspose2d.
# We expect a discrepancy between the custom CUDA kernel and the reference implementation.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_output_padding_issue():
    cuda_module = build_kernel()
    torch.manual_seed(42)
    batch_size = 2
    in_channels = 4
    out_channels = 6
    in_height = 8
    in_width = 8
    kernel_size = (3, 3)
    stride = 2
    padding = 1
    output_padding = 1  # nonzero output_padding triggers the issue
    groups = 1
    dilation = 1

    # Create input, weight and bias tensors
    x = torch.randn(batch_size, in_channels, in_height, in_width, device='cuda', dtype=torch.float32)
    # For ConvTranspose2d, PyTorch weight is of shape (in_channels, out_channels/groups, kH, kW)
    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # PyTorch reference (using nn.ConvTranspose2d)
    ref_conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, output_padding=output_padding,
                                        groups=groups, bias=True, dilation=dilation).to('cuda')
    # Force the same weight and bias. PyTorch uses a different weight layout convention,
    # so we manually copy the weight (and flip kernel if needed) to match our kernel.
    with torch.no_grad():
        ref_conv.weight.copy_(weight)
        ref_conv.bias.copy_(bias)
    y_ref = ref_conv(x)

    # Custom CUDA kernel forward (our implementation ignores output_padding in its compute)
    y_custom = cuda_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)

    # We expect a significant error since our kernel ignores output_padding in the inner loop.
    diff = (y_custom - y_ref).abs().max().item()
    assert diff > 1e-4, f"Output_padding issue not triggered. Max diff: {diff}"

# Issue 2: Only scalar (symmetric) stride/padding/dilation are supported.
# Passing asymmetric parameters (e.g. padding as tuple) should raise an error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_asymmetric_params_issue():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 6
    in_height = 8
    in_width = 8
    kernel_size = (3, 3)
    # Asymmetric padding provided as tuple
    asymmetric_padding = (1, 2)
    stride = 2
    output_padding = 0
    groups = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, in_height, in_width, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # Our kernel expects int for padding. Passing a tuple might be caught in Python before the C++ code.
    with pytest.raises(TypeError):
        cuda_module.forward(x, weight, bias, stride, asymmetric_padding, output_padding, groups, dilation)

# Issue 3: Assumption of 128-bit alignment using __ldg() without checking alignment.
# Allocate misaligned input memory by slicing an otherwise contiguous tensor.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alignment_issue():
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 4
    out_channels = 8
    in_height = 16
    in_width = 16
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1
    dilation = 1

    # Create a contiguous tensor and then create a misaligned view by slicing one element off the first dimension.
    x_full = torch.randn(batch_size + 1, in_channels, in_height, in_width, device='cuda', dtype=torch.float32)
    # Use a slice that is unlikely to be 128-bit aligned.
    x = x_full.narrow(0, 1, batch_size).clone()  # clone to force new allocation

    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # Run the kernel. Depending on memory alignment, the __ldg usage may lead to wrong results.
    y_custom = cuda_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
    # Since we have no ground truth here, the test will at least check that results are produced.
    # In a real scenario, misalignment may cause silent errors. We check for NaNs as a hint.
    assert not torch.isnan(y_custom).any(), "Misaligned memory load resulted in NaN values."

# Issue 4: The kernel does not flip the spatial kernel (lack of kernel reversal).
# We'll use a non-symmetric kernel (where weight flipped produces a different result) and compare against the PyTorch ConvTranspose2d.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_reversal_issue():
    cuda_module = build_kernel()
    torch.manual_seed(0)
    batch_size = 1
    in_channels = 1
    out_channels = 1
    in_height = 5
    in_width = 5
    kernel_size = (3, 3)
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    dilation = 1

    # Create an input with constant values and a non-symmetric kernel.
    x = torch.ones(batch_size, in_channels, in_height, in_width, device='cuda', dtype=torch.float32)
    # Create a kernel where values differ in a non-symmetric pattern.
    weight = torch.tensor([[[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]]]], device='cuda', dtype=torch.float32)
    bias = torch.zeros(out_channels, device='cuda', dtype=torch.float32)

    # PyTorch reference using nn.ConvTranspose2d. Note:
    # To match the custom kernel we need to use the same weight layout.
    ref_conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, output_padding=output_padding,
                                        groups=groups, bias=False, dilation=dilation).to('cuda')
    with torch.no_grad():
        ref_conv.weight.copy_(weight)
    y_ref = ref_conv(x)

    y_custom = cuda_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
    
    # Because the custom kernel does not flip the kernel, its result will differ from the reference.
    diff = (y_custom - y_ref).abs().max().item()
    assert diff > 1e-4, f"Kernel reversal issue not triggered. Max diff: {diff}"

# Issue 5: Only float32 and float64 are supported.
# Using half precision (float16) input should cause a dispatch failure or a runtime error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dtype_support_issue():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 6
    in_height = 8
    in_width = 8
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, in_height, in_width, device='cuda', dtype=torch.float16)
    weight = torch.randn(in_channels, out_channels // groups, *kernel_size, device='cuda', dtype=torch.float16)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float16)
    
    with pytest.raises(RuntimeError):
        # This should raise an error because half precision is not dispatched.
        cuda_module.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
