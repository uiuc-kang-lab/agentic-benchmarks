
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="depthwise_conv_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to mimic a depthwise conv2d using PyTorch's native Conv2d
def torch_depthwise_conv2d(x, weight, bias, stride, padding, dilation, groups):
    # Note: we assume a weight shape of (channels, 1, kernel_size_width)
    # For a proper comparison, we only vary the height dimension (vertical kernel) when using the custom kernel.
    # For a general kernel (with width > 1 or dilation != 1 horizontally) the native conv2d will differ.
    conv = torch.nn.Conv2d(
        in_channels=x.size(1),
        out_channels=x.size(1),
        kernel_size=(weight.shape[2], 1),  # the custom kernel uses kernel_h from dim2 and kernel width = 1.
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=x.size(1),
        bias=(bias is not None)
    ).to(x.device, x.dtype)
    # Overwrite the weights and bias to match our custom kernel settings
    conv.weight.data.copy_(weight.view(x.size(1), 1, weight.shape[2], 1))
    if bias is not None:
        conv.bias.data.copy_(bias)
    return conv(x)

def test_non_float32_input():
    """
    Issue 4: The kernel assumes float32 inputs.
    Passing a double precision (float64) tensor should trigger an error
    (or produce wrong results). We test that using a double tensor does not work.
    """
    cuda_module = build_kernel()
    batch, channels, in_h, in_w = 2, 3, 16, 16
    # Create double precision input and corresponding weight and bias tensors
    x = torch.randn(batch, channels, in_h, in_w, device="cuda", dtype=torch.float64)
    # Prepare weight: our custom kernel expects weight shape [channels, any?, kernel_h] flattened
    # Since the kernel reads weight as weight[c * kernel_h + kh], we create weight with shape (channels, 1, kernel_h)
    kernel_size = 3
    weight = torch.randn(channels, 1, kernel_size, device="cuda", dtype=torch.float64)
    bias = torch.randn(channels, device="cuda", dtype=torch.float64)

    with pytest.raises(RuntimeError):
        # Calling forward with double precision is expected to throw an error or misbehave.
        _ = cuda_module.forward(x, weight, bias, 1, 0, 1, channels)

def test_kernel_width_only_one_assumption():
    """
    Issue 2 & 3: The kernel is written for a kernel with width=1.
    If a weight with a horizontal kernel width greater than 1 is provided,
    the custom CUDA kernel will ignore the horizontal kernel dimension.
    Thus, its output will differ from that of PyTorch's native implementation.
    """
    cuda_module = build_kernel()
    batch, channels, in_h, in_w = 2, 3, 20, 20
    stride, padding, dilation = 1, 0, 1
    # Create an input tensor
    x = torch.randn(batch, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    # Here we create a weight with kernel size (3, 2) instead of (kernel_h, 1)
    # The custom kernel will only loop over the first (vertical) dimension.
    kernel_h, kernel_w = 3, 2
    # PyTorch Conv2d weight shape: (channels, 1, kernel_h, kernel_w).
    weight = torch.randn(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    
    # For the custom kernel, we flatten the weight to simulate it reading only kernel_h values.
    # We only pass the first column of the weight.
    weight_custom = weight[..., 0]  # shape (channels, 1, kernel_h)
    
    # Run custom kernel and native convolution.
    out_custom = cuda_module.forward(x, weight_custom, bias, stride, padding, dilation, channels)
    # Native convolution expects the full weight (and uses both kernel dimensions).
    conv = torch.nn.Conv2d(channels, channels, kernel_size=(kernel_h, kernel_w), stride=stride,
                           padding=padding, dilation=dilation, groups=channels, bias=True).to("cuda")
    conv.weight.data.copy_(weight)
    conv.bias.data.copy_(bias)
    out_torch = conv(x)
    
    # The outputs will differ because the custom kernel ignores the horizontal kernel dimension.
    # We check that they are not nearly equal.
    assert not torch.allclose(out_custom, out_torch, atol=1e-4), "Custom kernel unexpectedly matched PyTorch output!"

def test_incorrect_dilation_handling_horizontally():
    """
    Issue 3: The kernel does not incorporate dilation for the horizontal direction.
    When dilation != 1, the horizontal index computation is incorrect.
    We compare the custom kernel against PyTorch's native convolution,
    and they should differ in this case.
    """
    cuda_module = build_kernel()
    batch, channels, in_h, in_w = 2, 3, 20, 20
    stride, padding, dilation = 1, 1, 2
    kernel_size = 3  # vertical kernel size
    x = torch.randn(batch, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(channels, 1, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)
    
    # Custom kernel: only uses dilation for the vertical dimension.
    out_custom = cuda_module.forward(x, weight, bias, stride, padding, dilation, channels)
    
    # For PyTorch native conv2d, create a kernel with size (kernel_size, 1) and proper dilation.
    conv = torch.nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=stride,
                           padding=padding, dilation=dilation, groups=channels, bias=True).to("cuda")
    conv.weight.data.copy_(weight.view(channels, 1, kernel_size, 1))
    conv.bias.data.copy_(bias)
    out_torch = conv(x)
    
    # Because the custom kernel does not handle horizontal dilation (it assumes width=1 with no dilation)
    # the results will differ when dilation is applied.
    assert not torch.allclose(out_custom, out_torch, atol=1e-4), "Custom kernel output unexpectedly matches torch output!"

