
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile the CUDA kernel from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case for Issue 1: Non-square kernel
# Here we deliberately create a rectangular (non-square) kernel.
# The custom CUDA kernel always uses weight.size(2) for both height and width,
# so the output will be computed incorrectly compared to PyTorch's native ConvTranspose2d.
def test_non_square_kernel():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 3
    out_channels = 4
    # Create a rectangular kernel: kernel height != kernel width.
    kernel_height = 3
    kernel_width = 5  # non-square!
    stride = 2
    padding = 1
    dilation = 1
    height_in = 8
    width_in = 8

    # Create input tensor.
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda")
    # Create weight tensor with non-square kernel.
    # The custom kernel expects weight of shape [in_channels, out_channels, kernel_size, kernel_size],
    # so we force a rectangular kernel. This will break the intended indexing.
    weight = torch.randn(in_channels, out_channels, kernel_height, kernel_width, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    cuda_mod = build_kernel()
    # Call custom kernel
    output = cuda_mod.forward(input_tensor, weight, bias, stride, padding, dilation)

    # Now create a reference ConvTranspose2d with a rectangular kernel.
    # PyTorch's native implementation supports non-square kernels.
    ref_convtrans = torch.nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=(kernel_height, kernel_width),
                                             stride=stride, padding=padding, dilation=dilation, bias=True).to("cuda")
    # Force the same weight and bias.
    ref_convtrans.weight.data.copy_(weight)
    ref_convtrans.bias.data.copy_(bias)
    output_ref = ref_convtrans(input_tensor)

    # The outputs should differ as the custom kernel incorrectly assumes a square kernel.
    assert not torch.allclose(output, output_ref, atol=1e-4), \
           "Custom kernel output matches reference for non-square kernel, but it should not."

# Test case for Issue 2: Data type support
# The kernel only supports float32. Passing a double (float64) tensor should trigger an error.
def test_tensor_dtype():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    height_in = 8
    width_in = 8

    # Create double precision input, weight, bias.
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    cuda_mod = build_kernel()
    # Expect an error since the kernel uses float pointers.
    with pytest.raises(RuntimeError):
        _ = cuda_mod.forward(input_tensor, weight, bias, stride, padding, dilation)

# Test case for Issue 3: Grouped convolution
# The CUDA kernel’s indexing assumes weight shape [in_channels, out_channels, k, k]
# but for grouped convolution PyTorch uses weight shape [in_channels, out_channels // groups, k, k].
# We simulate a grouped convolution situation and show that the custom kernel output
# does not match the reference implementation.
def test_grouped_convolution():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 4
    out_channels = 4
    groups = 2
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    height_in = 8
    width_in = 8

    # For a grouped convolution, PyTorch expects weight with shape [in_channels, out_channels//groups, k, k]
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)

    cuda_mod = build_kernel()
    # When using the custom kernel, we pass in the weight without adjusting for groups.
    output = cuda_mod.forward(input_tensor, weight, bias, stride, padding, dilation)

    # Reference: use PyTorch native ConvTranspose2d with groups.
    ref_convtrans = torch.nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride, padding=padding, dilation=dilation,
                                             bias=True, groups=groups).to("cuda")
    # For the reference, we need to adjust the weight:
    # PyTorch's ConvTranspose2d expects weight shape [in_channels, out_channels//groups, k, k].
    ref_convtrans.weight.data.copy_(weight)
    ref_convtrans.bias.data.copy_(bias)
    output_ref = ref_convtrans(input_tensor)

    # The custom kernel does not support groups, so its output will differ.
    assert not torch.allclose(output, output_ref, atol=1e-4), \
           "Custom kernel output matches reference for grouped convolution, but it should not."

