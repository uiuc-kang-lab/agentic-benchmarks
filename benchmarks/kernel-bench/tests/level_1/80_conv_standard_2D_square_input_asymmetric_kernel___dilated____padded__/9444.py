
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="conv2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case for Issue 1: Unused shared memory.
# Although this unused shared memory does not lead to an incorrect output,
# one can test on a non–trivial input (with large spatial dimensions) and compare
# the result with PyTorch's native convolution. If the shared memory were properly used,
# one might expect a performance boost. Here we just check numerical correctness.
def test_unused_shared_memory():
    torch.manual_seed(0)
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)
    
    # Create inputs and weights
    x = torch.randn(batch_size, in_channels, 128, 128, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Run the custom kernel
    conv_module = build_kernel()
    out_kernel = conv_module.forward(x, weight, bias, stride, padding, dilation)
    
    # Run PyTorch's built-in conv2d for comparison
    out_ref = torch.nn.functional.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    
    # The kernel produces a correct result but the unused shared memory signals a missed optimization opportunity.
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), "Output of the custom kernel does not match reference."

# Test case for Issue 2: Lack of support for asymmetric stride.
# Since the kernel accepts an int for stride (applied to both dimensions),
# attempting to pass a tuple for stride (e.g. (1,2)) should lead to an error.
def test_asymmetric_stride():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 3
    out_channels = 5
    kernel_size = (3, 3)
    # Intentionally using a tuple for stride to simulate an asymmetric stride request.
    stride = (1, 2)
    padding = (1, 1)
    dilation = (1, 1)
    
    x = torch.randn(batch_size, in_channels, 32, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    conv_module = build_kernel()
    with pytest.raises(TypeError):
        # The kernel forward expects an int for stride, so passing a tuple should error out.
        conv_module.forward(x, weight, bias, stride, padding, dilation)

# Test case for Issue 3: Limited support for non-contiguous or non-float32 inputs.
# a) Non-contiguous input tensor should trigger a TORCH_CHECK failure.
def test_non_contiguous_input():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)
    
    # Create a contiguous tensor then make it non-contiguous by transposing a dimension
    x = torch.randn(batch_size, in_channels, 32, 32, device="cuda", dtype=torch.float32).transpose(1, 2)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    conv_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        conv_module.forward(x, weight, bias, stride, padding, dilation)

# b) Non-float32 input tensor should lead to an error when the kernel tries to use data_ptr<float>().
def test_non_float_input():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    dilation = (1, 1)
    
    # Create a double type tensor, which is not supported by the kernel.
    x = torch.randn(batch_size, in_channels, 32, 32, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)
    
    conv_module = build_kernel()
    with pytest.raises(RuntimeError):
        conv_module.forward(x, weight, bias, stride, padding, dilation)
