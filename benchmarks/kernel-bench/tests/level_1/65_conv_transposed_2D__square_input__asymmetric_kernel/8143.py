
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Test case 1: Asymmetric spatial parameters (tuple stride, padding, output_padding, dilation).
# The extension binding expects scalar ints, so passing a tuple should trigger an error.
def test_asymmetric_parameters():
    cuda_module = build_kernel()
    
    # Prepare dummy input tensors.
    # Using CUDA tensors.
    x = torch.randn(2, 4, 8, 8, device="cuda", dtype=torch.float32)
    # weight should have shape [in_channels, out_channels // groups, kH, kW]
    # Here, let in_channels=4, groups=1 so out_channels must be defined accordingly.
    weight = torch.randn(4, 8, 3, 5, device="cuda", dtype=torch.float32)
    bias = torch.randn(8, device="cuda", dtype=torch.float32)
    
    # Instead of scalar, we pass tuple for stride -> should trigger an error in argument binding.
    with pytest.raises(TypeError):
        # Trying to pass a tuple in place of stride.
        cuda_module.forward(x, weight, bias, (2, 2), 1, 0, 1, 1)

# Test case 2: Non-contiguous input tensor.
# If a non-contiguous input is passed, the kernel’s index computations may be incorrect.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    
    # Create a contiguous tensor then make it non-contiguous by transposition.
    x = torch.randn(2, 4, 8, 8, device="cuda", dtype=torch.float32).transpose(1, 2)
    # Ensure it is non-contiguous.
    assert not x.is_contiguous()
    
    # Create weight tensor matching [in_channels, out_channels//groups, kH, kW].
    # After transpose, the number of channels is not positioned as expected by the kernel.
    # We deliberately set parameters that do work for contiguous tensors.
    # For testing, we assume in_channels=4, groups=1.
    weight = torch.randn(4, 8, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(8, device="cuda", dtype=torch.float32)
    
    # Call the kernel and compare with PyTorch native conv_transpose2d.
    # Expect the outputs to differ because the kernel assumed contiguous layout.
    output_kernel = cuda_module.forward(x.contiguous(), weight, bias, 2, 1, 0, 1, 1)
    output_native = torch.nn.functional.conv_transpose2d(x, weight, bias, stride=2, padding=1, output_padding=0)
    
    # The outputs should not be close.
    assert not torch.allclose(output_kernel, output_native, atol=1e-4), \
        "Kernel produced correct result on non-contiguous input, but it should fail due to wrong indexing."

# Test case 3: Input tensor in half-precision.
# Since only float32 and float64 are dispatched, passing a half precision tensor should trigger an error.
def test_half_precision_input():
    cuda_module = build_kernel()
    
    x = torch.randn(2, 4, 8, 8, device="cuda", dtype=torch.float16)
    weight = torch.randn(4, 8, 3, 3, device="cuda", dtype=torch.float16)
    bias = torch.randn(8, device="cuda", dtype=torch.float16)
    
    with pytest.raises(RuntimeError):
        # This should raise an error because AT_DISPATCH_FLOATING_TYPES does not cover float16.
        cuda_module.forward(x, weight, bias, 2, 1, 0, 1, 1)

# Test case 4: in_channels not divisible by groups.
# If in_channels is not divisible by groups, computed indices for the weight will be invalid.
def test_invalid_groups():
    cuda_module = build_kernel()
    
    # in_channels=5 and groups=2 is invalid since 5 is not divisible by 2.
    x = torch.randn(2, 5, 8, 8, device="cuda", dtype=torch.float32)
    # weight shape: [in_channels, out_channels//groups, kH, kW].
    # We set out_channels arbitrarily; here groups=2.
    weight = torch.randn(5, 6, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(12, device="cuda", dtype=torch.float32)
    
    # We expect an error or incorrect execution due to improper memory indexing.
    # Here we use pytest.raises to catch a potential runtime error.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, weight, bias, 2, 1, 0, 2, 1)
