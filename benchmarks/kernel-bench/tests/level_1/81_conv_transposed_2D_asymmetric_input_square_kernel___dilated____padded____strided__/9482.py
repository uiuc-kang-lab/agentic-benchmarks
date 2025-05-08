
import pytest
import torch
from torch.utils.cpp_extension import load

# Build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to compute the expected output dimensions
def compute_out_dims(in_size, kernel_size, stride, padding, dilation):
    # Note: This formula does not include output_padding.
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

# 1. Test to trigger issue with non-float32 inputs
def test_non_float32_input():
    cuda_module = build_kernel()
    
    # Create input, weight and bias in float64
    batch_size = 2
    in_channels = 3
    out_channels = 4
    in_height, in_width = 8, 10
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    
    input_tensor = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float64)
    # weight shape for ConvTranspose2d in our kernel is (in_channels, out_channels, kernel_size, kernel_size)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)
    
    # Since our kernel only processes float32, this call is expected to produce wrong results or crash.
    with pytest.raises(RuntimeError):
        # We expect an error because the kernel internally uses __ldg on float pointers.
        cuda_module.forward(input_tensor, weight, bias, stride, padding, dilation)

# 2. Test to trigger the issue with non-contiguous input tensors
def test_non_contiguous_input():
    cuda_module = build_kernel()
    
    batch_size = 2
    in_channels = 4
    out_channels = 5
    in_height, in_width = 10, 12
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    
    # Create contiguous tensor and then make it non-contiguous by transposing spatial dimensions.
    input_contig = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    input_noncontig = input_contig.permute(0, 1, 3, 2)  # now non-contiguous
    # For the weight, we enforce contiguous memory.
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Our kernel assumes contiguous memory; thus, using a non-contiguous tensor should lead to incorrect results.
    output = cuda_module.forward(input_noncontig, weight, bias, stride, padding, dilation)
    
    # Compute expected output dimensions using the same formula.
    out_height = compute_out_dims(in_height, kernel_size, stride, padding, dilation)
    out_width  = compute_out_dims(in_width, kernel_size, stride, padding, dilation)
    
    # Check that the produced output has the expected size.
    assert output.shape == (batch_size, out_channels, out_height, out_width), \
        f"Expected output shape {(batch_size, out_channels, out_height, out_width)}, got {output.shape}"
    
    # Since non-contiguous memory may cause misaligned __ldg loads, compare against a contiguous version.
    input_contig2 = input_noncontig.contiguous()
    output_correct = cuda_module.forward(input_contig2, weight, bias, stride, padding, dilation)
    # The outputs should be close if memory accesses were safe; here we expect them to differ.
    with pytest.raises(AssertionError):
        assert torch.allclose(output, output_correct, atol=1e-4), "Kernel unexpectedly succeeded with non-contiguous input."

# 3. Test to trigger the issue with output dimension computation (missing output_padding support)
def test_output_dimension_mismatch():
    cuda_module = build_kernel()
    
    batch_size = 1
    in_channels = 3
    out_channels = 3
    in_height, in_width = 5, 5
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    output_padding = 1  # This parameter is not supported in our kernel formula.
    
    input_tensor = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.zeros(out_channels, device="cuda", dtype=torch.float32)
    
    # Create a PyTorch ConvTranspose2d with output_padding set.
    conv_transpose = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        dilation=dilation, output_padding=output_padding, bias=True
    ).to("cuda")
    # Manually set the conv_transpose weights and bias to match our test data.
    with torch.no_grad():
        conv_transpose.weight.copy_(weight)
        conv_transpose.bias.copy_(bias)
    
    output_ref = conv_transpose(input_tensor)
    output_cuda = cuda_module.forward(input_tensor, weight, bias, stride, padding, dilation)
    
    # The output dimension computed by our kernel ignores output_padding.
    # Thus, the output shape differs from the reference.
    assert output_cuda.shape != output_ref.shape, \
        f"Kernel output shape {output_cuda.shape} unexpectedly matches reference shape {output_ref.shape} despite missing output_padding support."

# 4. Test to trigger potential issues with fixed TILE_SIZE accumulation when in_channels is not a multiple of 4.
def test_in_channels_non_multiple_of_tile():
    cuda_module = build_kernel()
    
    batch_size = 2
    in_channels = 5  # Not a multiple of TILE_SIZE (4)
    out_channels = 3
    in_height, in_width = 7, 8
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    input_tensor = torch.randn(batch_size, in_channels, in_height, in_width, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    output = cuda_module.forward(input_tensor, weight, bias, stride, padding, dilation)
    # Compute expected dimensions.
    out_height = compute_out_dims(in_height, kernel_size, stride, padding, dilation)
    out_width = compute_out_dims(in_width, kernel_size, stride, padding, dilation)
    assert output.shape == (batch_size, out_channels, out_height, out_width), \
        f"Expected output shape {(batch_size, out_channels, out_height, out_width)}, got {output.shape}"
    
    # Compare with PyTorch's native conv_transpose2d (which is correct) to highlight differences.
    conv_transpose = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation, bias=True).to("cuda")
    with torch.no_grad():
        conv_transpose.weight.copy_(weight)
        conv_transpose.bias.copy_(bias)
    output_ref = conv_transpose(input_tensor)
    
    # The fixed TILE_SIZE handling might accumulate slight differences when in_channels is not divisible by 4.
    # We expect a difference beyond an acceptable tolerance.
    with pytest.raises(AssertionError):
        assert torch.allclose(output, output_ref, atol=1e-5), "Kernel output unexpectedly close to reference output with non-multiple in_channels."

