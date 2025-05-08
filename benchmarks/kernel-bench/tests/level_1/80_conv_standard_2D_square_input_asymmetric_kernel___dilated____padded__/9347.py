
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case for Issue 1: non-symmetric stride
def test_non_symmetric_stride():
    # Build the module
    cuda_module = build_kernel()

    # Prepare dummy inputs: using float32 so that dtype issue is avoided.
    batch_size, in_channels, height, width = 1, 3, 32, 32
    out_channels = 8
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # Create weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
    kernel_h, kernel_w = 3, 5  # asymmetric kernel shape
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    # Create bias tensor if needed
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Non-symmetric stride passed as a tuple instead of an int.
    # The kernel.forward expects an int stride. In a general conv2d interface one might want to pass 
    # a tuple (stride_h, stride_w) which should produce an error.
    with pytest.raises(TypeError):
        # This attempt to pass a tuple for stride should cause an error because the C++ function signature is int.
        cuda_module.forward(x, weight, bias, (2, 3), (1, 2), (1, 1))

# Test case for Issue 2: unsupported data type (only float32 is supported)
def test_unsupported_dtype():
    cuda_module = build_kernel()

    # Prepare inputs in float64
    batch_size, in_channels, height, width = 1, 3, 32, 32
    out_channels = 8
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float64)
    kernel_h, kernel_w = 3, 5
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    # Since the kernel always uses float (float32) data_ptrs, calling forward() with double tensors
    # will lead to misinterpreting the underlying memory if reinterpreted as float.
    # We test and expect that the output is significantly different from a reference PyTorch convolution.
    stride = 1
    padding = (1, 2)
    dilation = (1, 1)
    output = cuda_module.forward(x, weight, bias, stride, padding, dilation)

    # Create a reference convolution using PyTorch’s conv2d (after casting x and weight to float32)
    # so that we can detect a significant difference.
    x_ref = x.float()
    weight_ref = weight.float()
    bias_ref = bias.float()
    ref_conv = torch.nn.functional.conv2d(x_ref, weight_ref, bias_ref,
                                           stride=stride, padding=padding, dilation=dilation)
    output_cast = output  # output is produced by the kernel (interpreted as float32 values)
    
    # The output from the kernel should be far from the reference because the wrong precision was used.
    # We use a high tolerance to detect the error.
    diff = (output_cast - ref_conv).abs().max().item()
    assert diff > 1e-2, f"Expected significant numerical difference due to dtype misinterpretation, got diff: {diff}"
