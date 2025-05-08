
import torch
import pytest
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# A helper function to compile and load the CUDA kernel extension.
def build_kernel():
    cuda_module = load(
        name="cuda_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2: Incorrect unrolling and missing kernel flipping.
# This test compares the custom CUDA kernel output with PyTorch's built-in conv_transpose2d.
# Because the kernel does not flip the weights, the outputs should differ significantly.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incorrect_kernel_flipping_and_loop_unroll():
    # (Using a kernel whose dimensions do not match the unroll factor to trigger issue 1)
    batch_size = 4
    in_channels = 3
    out_channels = 2
    kernel_size = (3, 5)  # Note: kH is 3 but kW is 5, so the unroll pragma "3" is not valid for kW.
    stride = (2, 2)
    padding = (1, 2)
    
    # Create a random input tensor.
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    
    # Create weights and (optional) bias.
    # PyTorch conv_transpose2d expects weight shape: [in_channels, out_channels, kH, kW]
    weight = torch.randn(in_channels, out_channels, *kernel_size, device="cuda", dtype=torch.float32)
    bias = None  # No bias for this test.
    
    # Build module using PyTorch’s built-in conv_transpose2d.
    ref_output = F.conv_transpose2d(x, weight, bias, stride=stride, padding=padding)
    
    # Load our custom CUDA kernel.
    cuda_module = build_kernel()
    custom_output = cuda_module.forward(x, weight, bias, list(stride), list(padding))
    
    # The outputs are expected to differ due to missing kernel flipping and incorrect unrolling.
    # We deliberately assert that the maximum absolute difference is large.
    diff = (custom_output - ref_output).abs().max().item()
    assert diff > 1e-2, f"Custom kernel output appears to match the reference, diff = {diff}. Expected significant difference due to issues."

# Issue 3a: The kernel assumes float32 input.
# This test verifies that passing a tensor with a different dtype (e.g., float64) triggers an error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_input_tensor_type():
    batch_size = 4
    in_channels = 3
    out_channels = 2
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels, *kernel_size, device="cuda", dtype=torch.float64)
    
    cuda_module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The kernel expects float32 so using float64 should result in an error.
        cuda_module.forward(x, weight, None, list(stride), list(padding))

# Issue 3b: The kernel does not check for non-contiguous inputs.
# We test that providing a non-contiguous tensor leads to an incorrect result (or at least a noticeable deviation)
# when compared to the same input made contiguous.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_input():
    batch_size = 2
    in_channels = 3
    out_channels = 2
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    
    # Create a contiguous input then make it non-contiguous via transpose.
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    x_non_contig = x.permute(0, 2, 3, 1)  # Now shape is (N, H, W, C) and non-contiguous.
    x_non_contig = x_non_contig.contiguous().transpose(1, 3)  # Force a permutation that is non-standard.
    # Confirm non-contiguity.
    assert not x_non_contig.is_contiguous(), "Test setup error: tensor is unexpectedly contiguous."
    
    weight = torch.randn(in_channels, out_channels, *kernel_size, device="cuda", dtype=torch.float32)
    
    cuda_module = build_kernel()
    # Force the kernel to use non-contiguous input by passing x_non_contig.
    # The built-in conv_transpose2d requires contiguous inputs, so we compare against a contiguous copy.
    output_non_contig = cuda_module.forward(x_non_contig, weight, None, list(stride), list(padding))
    output_contig   = cuda_module.forward(x_non_contig.contiguous(), weight, None, list(stride), list(padding))
    
    # Since the kernel does a raw data_ptr<float>() access, non-contiguous inputs will likely produce incorrect results.
    diff = (output_non_contig - output_contig).abs().max().item()
    assert diff > 1e-4, f"Non-contiguous input did not affect output as expected, diff = {diff}."

