
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper to create random tensors on CUDA
def create_tensors(B, C_in, H, W, C_out, K, dtype=torch.float32, contiguous=True):
    input_tensor = torch.randn(B, C_in, H, W, dtype=dtype, device="cuda")
    # weight shape assumed to be [C_in, C_out, K, K] in the kernel
    weight_tensor = torch.randn(C_in, C_out, K, K, dtype=dtype, device="cuda")
    bias_tensor = torch.randn(C_out, dtype=dtype, device="cuda")
    if not contiguous:
        input_tensor = input_tensor.transpose(2, 3)  # This forces non-contiguous layout.
        input_tensor = input_tensor.transpose(2, 3)  # Return shape but non-contiguous in memory.
    return input_tensor, weight_tensor, bias_tensor

# Test 1: groups != 1
def test_groups_not_supported():
    cuda_module = build_kernel()
    B, C_in, H, W, C_out, K = 2, 4, 8, 8, 6, 3
    input_tensor, weight_tensor, bias_tensor = create_tensors(B, C_in, H, W, C_out, K)
    stride = 1
    padding = 0
    output_padding = 0
    groups = 2  # not supported
    with pytest.raises(RuntimeError, match="Only groups==1 is supported in the optimized kernel"):
        cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)

# Test 2: output_padding != 0
def test_output_padding_not_supported():
    cuda_module = build_kernel()
    B, C_in, H, W, C_out, K = 2, 4, 8, 8, 6, 3
    input_tensor, weight_tensor, bias_tensor = create_tensors(B, C_in, H, W, C_out, K)
    stride = 1
    padding = 0
    output_padding = 1  # not supported
    groups = 1
    with pytest.raises(RuntimeError, match="Only output_padding==0 is supported in the optimized kernel"):
        cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)

# Test 3: Non float32 input (e.g. float64)
def test_non_float32_dtype():
    cuda_module = build_kernel()
    B, C_in, H, W, C_out, K = 2, 4, 8, 8, 6, 3
    # Create tensors with dtype float64 - kernel expects float32 so this is not handled.
    input_tensor, weight_tensor, bias_tensor = create_tensors(B, C_in, H, W, C_out, K, dtype=torch.float64)
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    # The kernel does not check the dtype and will call data_ptr<float>() on a float64 tensor.
    # This can lead to undefined behavior. Here, we run the forward function and then compare against
    # PyTorch's native ConvTranspose2d. They should not match within tolerance.
    output_custom = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)
    
    # Prepare reference using native nn.ConvTranspose2d.
    # Note: We must cast weights to float64 for the built-in layer.
    conv = torch.nn.ConvTranspose2d(C_in, C_out, K, stride=stride, padding=padding, output_padding=output_padding, bias=True)
    conv.weight.data = weight_tensor.clone().to(torch.float64)
    conv.bias.data = bias_tensor.clone().to(torch.float64)
    input_ref = input_tensor.clone()
    output_ref = conv(input_ref)
    
    # They should not be close because our kernel misinterprets double data.
    assert not torch.allclose(output_custom.to(torch.float64), output_ref, atol=1e-4), \
           "Kernel incorrectly produced matching results for float64 input."

# Test 4: Non-square (rectangular) kernel dimensions
def test_rectangular_kernel():
    cuda_module = build_kernel()
    B, C_in, H, W = 2, 4, 8, 8
    C_out = 6
    K_h, K_w = 3, 4  # rectangular kernel (non-square)
    # Create input and bias normally.
    input_tensor = torch.randn(B, C_in, H, W, dtype=torch.float32, device="cuda")
    bias_tensor = torch.randn(C_out, dtype=torch.float32, device="cuda")
    # Create a weight with rectangular shape.
    weight_tensor = torch.randn(C_in, C_out, K_h, K_w, dtype=torch.float32, device="cuda")
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    # Our kernel takes K = weight.size(2) and expects square weight [K, K]. With a rectangular kernel,
    # only the first dimension (K_h) is used and K_w is ignored. The computed output will be different
    # from PyTorch's native ConvTranspose2d, so we expect a mismatch.
    output_custom = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)
    
    # Define a native conv-transpose layer using the rectangular kernel.
    conv = torch.nn.ConvTranspose2d(C_in, C_out, (K_h, K_w), stride=stride, padding=padding, output_padding=output_padding, bias=True)
    conv.weight.data = weight_tensor.clone()
    conv.bias.data = bias_tensor.clone()
    output_ref = conv(input_tensor)
    
    # They should not match.
    assert not torch.allclose(output_custom, output_ref, atol=1e-4), \
           "Kernel incorrectly produced matching results for a rectangular kernel."

# Test 5: Non-contiguous input tensor
def test_non_contiguous_input():
    cuda_module = build_kernel()
    B, C_in, H, W, C_out, K = 2, 4, 8, 8, 6, 3
    # Create non-contiguous input tensor.
    input_tensor, weight_tensor, bias_tensor = create_tensors(B, C_in, H, W, C_out, K, contiguous=False)
    
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding, groups)
