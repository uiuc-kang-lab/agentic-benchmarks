
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Utility function to create a default valid input for the kernel forward function.
def create_valid_tensors(dtype=torch.float32, contiguous=True):
    # Use a fixed configuration.
    batch_size = 2
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    in_height = 16
    in_width = 32
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    
    x = torch.randn(batch_size, in_channels, in_height, in_width, dtype=dtype, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=dtype, device="cuda")
    bias = torch.randn(out_channels, dtype=dtype, device="cuda")
    
    if not contiguous:
        # Make the tensor non-contiguous by transposing two dimensions and then slicing.
        x = x.transpose(2, 3)
        weight = weight.permute(0,2,1,3)
        bias = bias.unsqueeze(0)
    
    return x, weight, bias, stride, padding, dilation, groups

def test_groups_not_supported():
    x, weight, bias, stride, padding, dilation, _ = create_valid_tensors()
    groups = 2  # set groups != 1
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError, match="Only groups==1 is supported"):
        # Calling the forward function with groups != 1 should throw an error.
        kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)

def test_non_contiguous_input():
    x, weight, bias, stride, padding, dilation, groups = create_valid_tensors(contiguous=True)
    # Create a non-contiguous input tensor by advanced indexing.
    non_contig_x = x[:, :, ::2, ::2]  # slicing usually returns a non-contiguous tensor
    # But adjust the size of x to match weight convolution output expectations.
    # Compute output sizes:
    in_height, in_width = non_contig_x.size(2), non_contig_x.size(3)
    kernel_size = weight.size(2)
    out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width  = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    if out_height <=0 or out_width<=0:
        pytest.skip("Skip test_non_contiguous_input due to invalid output dimensions from slicing")
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        kernel_module.forward(non_contig_x, weight, bias, stride, padding, dilation, groups)

def test_non_contiguous_weight():
    x, weight, bias, stride, padding, dilation, groups = create_valid_tensors(contiguous=True)
    # Make weight non-contiguous by transposition.
    non_contig_weight = weight.permute(1, 0, 2, 3)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        kernel_module = build_kernel()
        kernel_module.forward(x, non_contig_weight, bias, stride, padding, dilation, groups)

def test_incorrect_dtype():
    # Create tensors with double type which is not supported.
    x, weight, bias, stride, padding, dilation, groups = create_valid_tensors(dtype=torch.double, contiguous=True)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        # Although the error message might not be exactly for dtype,
        # the kernel expects float data and using double will trigger an unexpected error.
        kernel_module.forward(x, weight, bias, stride, padding, dilation, groups)
