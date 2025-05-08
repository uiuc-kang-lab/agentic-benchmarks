
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build/load the CUDA kernel module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Prepare a standard valid input for a transposed convolution.
def get_valid_inputs(dtype=torch.float32, contiguous=True):
    batch_size = 16
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    height = 128
    width = 128
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    # Create input tensor and weight tensor in correct shapes.
    x = torch.randn(batch_size, in_channels, height, width, dtype=dtype, device='cuda')
    # Weight shape: (in_channels, out_channels, kernel_size, kernel_size)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, dtype=dtype, device='cuda')
    bias = None  # For simplicity, bias is omitted (or can be added similarly)

    if not contiguous:
        # Force non-contiguity by transposing and then back.
        x = x.transpose(2, 3)
        weight = weight.transpose(2, 3)

    return x, weight, bias, stride, padding, output_padding, groups

# Test 1: groups != 1 should trigger a runtime error.
def test_groups_not_one():
    module = build_kernel()
    x, weight, bias, stride, padding, output_padding, groups = get_valid_inputs()
    groups = 2  # Change groups to a value that is not supported.
    with pytest.raises(RuntimeError, match="Only groups==1 is supported"):
        # The forward function is wrapped by the module.forward
        module.forward(x, weight, bias, stride, padding, output_padding, groups)

# Test 2: output_padding != 0 should trigger a runtime error.
def test_nonzero_output_padding():
    module = build_kernel()
    x, weight, bias, stride, padding, output_padding, groups = get_valid_inputs()
    output_padding = 1  # non-zero output_padding not supported.
    with pytest.raises(RuntimeError, match="Only output_padding==0 is supported"):
        module.forward(x, weight, bias, stride, padding, output_padding, groups)

# Test 3: Using a tensor with dtype other than float32 should trigger an error.
def test_non_float32_dtype():
    module = build_kernel()
    x, weight, bias, stride, padding, output_padding, groups = get_valid_inputs(dtype=torch.float64)
    # The kernel expects float. torch::Tensor.data_ptr<float>() conversion will be invalid.
    with pytest.raises(RuntimeError):
        module.forward(x, weight, bias, stride, padding, output_padding, groups)

# Test 4: Using non-contiguous tensors should trigger an error.
def test_non_contiguous_input():
    module = build_kernel()
    x, weight, bias, stride, padding, output_padding, groups = get_valid_inputs(contiguous=False)
    # Even if the .is_contiguous() check is on the CPU side, launching the kernel with non-contiguous
    # memory should trigger a TORCH_CHECK failure.
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        module.forward(x, weight, bias, stride, padding, output_padding, groups)
        
if __name__ == "__main__":
    pytest.main([__file__])
