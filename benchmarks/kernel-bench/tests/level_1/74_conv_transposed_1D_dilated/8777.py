
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="conv_transpose1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Get a reference output using PyTorch's built-in ConvTranspose1d
def reference_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    # Weight in PyTorch conv_transpose1d is of shape (in_channels, out_channels, kernel_size)
    # built-in function expects weight.transpose(0,1) because it uses output_channels as first dim.
    weight_pt = weight.permute(1, 0, 2).contiguous()
    conv = nn.ConvTranspose1d(
        in_channels=x.size(1),
        out_channels=weight_pt.size(0),
        kernel_size=weight_pt.size(2),
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias is not None,
    ).to(x.device)
    # Set the weight and bias manually to match our kernel
    conv.weight.data.copy_(weight_pt)
    if bias is not None:
        conv.bias.data.copy_(bias)
    else:
        conv.bias.data.zero_()
    return conv(x)

# Test 1: Trigger issue with non-float32 (double) input tensors
def test_input_dtype():
    cuda_module = build_kernel()
    device = "cuda"
    N = 2
    C_in = 3
    L_in = 16
    C_out = 4
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create double tensors instead of float32
    x = torch.randn(N, C_in, L_in, device=device, dtype=torch.float64)
    # PyTorch's ConvTranspose1d weight shape: (in_channels, out_channels, kernel_size)
    weight = torch.randn(C_in, C_out, kernel_size, device=device, dtype=torch.float64)
    bias = torch.randn(C_out, device=device, dtype=torch.float64)

    with pytest.raises(Exception):
        # This should raise an error or produce incorrect results because the kernel expects float32
        y = cuda_module.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()

# Test 2: Trigger issue with non-contiguous input tensor
def test_noncontiguous_input():
    cuda_module = build_kernel()
    device = "cuda"
    N = 2
    C_in = 3
    L_in = 20
    C_out = 5
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(N, C_in, L_in, device=device, dtype=torch.float32)
    # Make x non-contiguous by transposing last two dimensions then transposing back
    x = x.transpose(1, 2).transpose(1, 2)
    assert not x.is_contiguous(), "x is unexpectedly contiguous."

    weight = torch.randn(C_in, C_out, kernel_size, device=device, dtype=torch.float32)
    bias = torch.randn(C_out, device=device, dtype=torch.float32)

    # The kernel forces a contiguous call via .contiguous() on x inside the function,
    # but weight is also expected contiguous – we only check x here.
    y_kernel = cuda_module.forward(x, weight, bias, stride, padding, dilation)
    y_ref = reference_conv_transpose1d(x.contiguous(), weight, bias, stride, padding, dilation)
    torch.cuda.synchronize()

    # They may not match because our kernel assumes a contiguous layout.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-4), \
        "Kernel unexpectedly handled non-contiguous input."

# Test 3: Trigger issue with grouped convolution
def test_grouped_convolution():
    cuda_module = build_kernel()
    device = "cuda"
    # Simulate a grouped convolution by splitting channels.
    N = 2
    groups = 2
    C_in_total = 4  # 2 groups of 2 each
    C_out_total = 6  # Assume each group outputs 3 channels
    L_in = 10
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    # Create an input tensor
    x = torch.randn(N, C_in_total, L_in, device=device, dtype=torch.float32)

    # Instead of a single weight tensor for conv_transpose1d, group convolution weight would have shape:
    # (in_channels, out_channels_per_group, kernel_size) where in_channels = C_in_total / groups.
    # Our kernel expects weight of shape (C_in_total, C_out_total, kernel_size) (groups==1).
    in_channels_per_group = C_in_total // groups
    out_channels_per_group = C_out_total // groups
    # Create a weight tensor that concatenates groups incorrectly
    weight = torch.randn(C_in_total, C_out_total, kernel_size, device=device, dtype=torch.float32)
    bias = torch.randn(C_out_total, device=device, dtype=torch.float32)

    # Running the kernel extension: it does not support groups so the answer will be inconsistent.
    y_kernel = cuda_module.forward(x, weight, bias, stride, padding, dilation)
    # Compute a “reference” by performing grouped conv_transpose1d manually
    y_groups = []
    for g in range(groups):
        x_g = x[:, g*in_channels_per_group:(g+1)*in_channels_per_group, :]
        weight_g = weight[g*in_channels_per_group:(g+1)*in_channels_per_group,
                          g*out_channels_per_group:(g+1)*out_channels_per_group, :]
        bias_g = bias[g*out_channels_per_group:(g+1)*out_channels_per_group]
        y_g = reference_conv_transpose1d(x_g.contiguous(), weight_g, bias_g, stride, padding, dilation)
        y_groups.append(y_g)
    y_ref = torch.cat(y_groups, dim=1)
    torch.cuda.synchronize()

    # They are expected to differ since the kernel cannot handle grouped convolution.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-4), \
        "Kernel unexpectedly handled grouped convolution correctly."

# Test 4: Trigger issue with extremely large output where the absence of a grid-stride loop may cause problems.
def test_large_output_kernel_launch():
    cuda_module = build_kernel()
    device = "cuda"
    N = 1
    C_in = 2
    L_in = 1024
    C_out = 2
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(N, C_in, L_in, device=device, dtype=torch.float32)
    weight = torch.randn(C_in, C_out, kernel_size, device=device, dtype=torch.float32)
    bias = torch.randn(C_out, device=device, dtype=torch.float32)

    # Force a large L_out so that total_elements is very high.
    # With stride=1 and padding=1, L_out is same as L_in.
    # To simulate a large output, we bump L_in.
    L_in_large = 16384  # large enough to stress the launch configuration
    x_large = torch.randn(N, C_in, L_in_large, device=device, dtype=torch.float32)
    weight_large = torch.randn(C_in, C_out, kernel_size, device=device, dtype=torch.float32)
    bias_large = torch.randn(C_out, device=device, dtype=torch.float32)

    y_kernel = cuda_module.forward(x_large, weight_large, bias_large, stride, padding, dilation)
    y_ref = reference_conv_transpose1d(x_large.contiguous(), weight_large, bias_large, stride, padding, dilation)
    torch.cuda.synchronize()

    # The lack of a grid-stride loop may result in missing some outputs or incorrect calculation.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-4), \
        "Kernel unexpectedly produced correct results for large outputs."

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
