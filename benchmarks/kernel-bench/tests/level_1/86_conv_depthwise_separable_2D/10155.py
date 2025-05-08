
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function implementing a standard PyTorch depthwise separable convolution for reference.
def torch_depthwise_separable_conv(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation):
    # For depthwise: note that PyTorch's conv2d with groups=in_channels works correctly.
    depthwise = torch.nn.functional.conv2d(
        x, depthwise_weight, bias=depthwise_bias, stride=stride, padding=padding, dilation=dilation, groups=x.shape[1]
    )
    pointwise = torch.nn.functional.conv2d(
        depthwise, pointwise_weight, bias=pointwise_bias, stride=1, padding=0
    )
    return pointwise

# Issue 1: Non-contiguous inputs – The kernel assumes contiguous inputs.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch = 2
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    height, width = 32, 32
    stride, padding, dilation = 1, 1, 1

    # Create input tensor and force it non-contiguous by transposing spatial dimensions 
    x = torch.randn(batch, in_channels, height, width, device="cuda")
    x_non_contig = x.permute(0, 2, 3, 1).contiguous().transpose(1, 3)  # make non-contiguous
    # Create weights with correct shapes.
    # For PyTorch depthwise conv, weight expected shape: (in_channels, 1, kernel_size, kernel_size)
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda")
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device="cuda")
    depthwise_bias = torch.randn(in_channels, device="cuda")
    pointwise_bias = torch.randn(out_channels, device="cuda")

    # Call the custom CUDA kernel; it will use the data_ptr() of non-contiguous tensor.
    output = cuda_module.forward(x_non_contig, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)
    ref_out = torch_depthwise_separable_conv(x_non_contig.contiguous(), depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)
    torch.cuda.synchronize()

    # This test should fail or produce a significant difference because the kernel assumed contiguous memory.
    assert not torch.allclose(output, ref_out, atol=1e-4), "Kernel unexpectedly handled non-contiguous inputs correctly."

# Issue 2: Use of #pragma unroll on runtime-determined loop bounds.
def test_runtime_kernel_size_unroll():
    cuda_module = build_kernel()
    batch = 2
    in_channels = 3
    out_channels = 8
    # Use a kernel size that is not common (e.g. 5) to force runtime determination.
    kernel_size = 5
    height, width = 32, 32
    stride, padding, dilation = 1, 2, 1  # padding chosen to keep output size reasonable

    x = torch.randn(batch, in_channels, height, width, device="cuda")
    # Note: depthwise weight shape for groups conv2d is (in_channels, 1, k, k)
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda")
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device="cuda")
    depthwise_bias = torch.randn(in_channels, device="cuda")
    pointwise_bias = torch.randn(out_channels, device="cuda")

    # Run the custom kernel
    output = cuda_module.forward(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)
    ref_out = torch_depthwise_separable_conv(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)
    torch.cuda.synchronize()

    # If unrolling was improperly applied due to the runtime kernel size, the results might differ.
    assert not torch.allclose(output, ref_out, atol=1e-4), "Kernel appears to handle runtime kernel size unrolling correctly, but an issue was expected."

# Issue 3: Lack of support for half-precision inputs.
def test_half_precision_input():
    cuda_module = build_kernel()
    batch = 2
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    height, width = 32, 32
    stride, padding, dilation = 1, 1, 1

    x = torch.randn(batch, in_channels, height, width, device="cuda", dtype=torch.float16)
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float16)
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device="cuda", dtype=torch.float16)
    depthwise_bias = torch.randn(in_channels, device="cuda", dtype=torch.float16)
    pointwise_bias = torch.randn(out_channels, device="cuda", dtype=torch.float16)

    # Expect the kernel dispatch to fail due to unsupported dtype.
    with pytest.raises(RuntimeError):
        output = cuda_module.forward(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 4: Inefficient or incorrect thread mapping when output size is not a multiple of (BLOCK_SIZE/WARP_SIZE)
def test_pointwise_thread_mapping_corner():
    cuda_module = build_kernel()
    # Choose dimensions so that total number of pixels is not a multiple of (BLOCK_SIZE/WARP_SIZE)
    # BLOCK_SIZE=256, WARP_SIZE=32 => pixels per block = 8.
    batch = 1
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    height, width = 7, 7  # 49 pixels, which is not a multiple of 8.
    stride, padding, dilation = 1, 1, 1

    x = torch.randn(batch, in_channels, height, width, device="cuda")
    depthwise_weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda")
    pointwise_weight = torch.randn(out_channels, in_channels, 1, 1, device="cuda")
    depthwise_bias = torch.randn(in_channels, device="cuda")
    pointwise_bias = torch.randn(out_channels, device="cuda")

    output = cuda_module.forward(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)
    ref_out = torch_depthwise_separable_conv(x, depthwise_weight, pointwise_weight, depthwise_bias, pointwise_bias, stride, padding, dilation)
    torch.cuda.synchronize()

    # The expected behavior is that the kernel would cover all pixels.
    # Here we check if the result is significantly different, which may indicate errors in thread mapping.
    assert not torch.allclose(output, ref_out, atol=1e-4), "Kernel unexpectedly handled pointwise mapping corner case correctly."

if __name__ == "__main__":
    pytest.main([__file__])
