
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA extension kernel from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to run the CUDA kernel forward and compare with PyTorch's ConvTranspose2d
def run_conv_transpose2d(input, weight, bias, stride, padding, dilation):
    cuda_mod = build_kernel()
    # Call our custom kernel replacement
    output_cuda = cuda_mod.forward(input, weight, bias, stride, padding, dilation)
    # Create a reference using PyTorch's built-in module
    in_channels = input.size(1)
    out_channels = weight.size(1)
    kernel_size = weight.size(2)
    conv_transpose = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=(bias is not None)
    ).to(input.device)
    
    # Manually assign weight and bias
    # PyTorch expects weight shape (in_channels, out_channels, kH, kW)
    conv_transpose.weight.data.copy_(weight)
    if bias is not None:
        conv_transpose.bias.data.copy_(bias)
    else:
        # When no bias is provided, PyTorch internally adds zero bias.
        conv_transpose.bias = torch.nn.Parameter(torch.zeros(out_channels, device=input.device))
    
    output_ref = conv_transpose(input)
    return output_cuda, output_ref

# Test case 1: Pass non-float32 tensor (e.g., float16), which should trigger an issue.
def test_non_float32():
    batch_size = 4
    in_channels = 3
    out_channels = 5
    kernel_size = 3
    height_in = 16
    width_in = 16
    stride = 2
    padding = 1
    dilation = 1

    # Create input, weight and bias as float16 instead of the expected float32.
    x = torch.randn(batch_size, in_channels, height_in, width_in, device='cuda', dtype=torch.float16)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float16)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float16)

    cuda_mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting the kernel to fail or produce an error due to wrong data type.
        cuda_mod.forward(x, weight, bias, stride, padding, dilation)

# Test case 2: Pass non-square kernel weight to trigger indexing issues.
def test_non_square_kernel():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    # Create a non-square kernel, e.g., kernel heights=3 and widths=5.
    kH, kW = 3, 5
    height_in = 10
    width_in = 10
    stride = 2
    padding = 1
    dilation = 1

    # The CUDA kernel only reads weight.size(2) and assumes a square kernel.
    # Here, we deliberately create a non-square weight.
    x = torch.randn(batch_size, in_channels, height_in, width_in, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kH, kW, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # Run the custom kernel versus PyTorch's built-in ConvTranspose2d.
    output_cuda, output_ref = run_conv_transpose2d(x, weight, bias, stride, padding, dilation)

    # They should not match because the custom kernel misinterprets the non-square kernel.
    # We expect the maximum difference to be significant.
    max_diff = (output_cuda - output_ref).abs().max().item()
    assert max_diff > 1e-3, f"Expected incorrect output due to non-square kernel, but max diff is {max_diff}"

if __name__ == "__main__":
    pytest.main([__file__])
