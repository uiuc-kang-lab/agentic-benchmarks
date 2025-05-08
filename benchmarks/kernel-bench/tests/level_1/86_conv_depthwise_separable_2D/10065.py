
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_cuda_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_missing_synchronization_issue():
    """
    Issue 1: Without proper device synchronization, errors from asynchronous kernel launches
    (like an out‐of‐bounds memory access caused by a shape mismatch) might not be caught immediately.
    Here we intentionally supply an incorrectly shaped depthwise weight tensor.
    """
    kernel_module = build_kernel()
    batch = 1
    in_channels = 3
    h = 8
    w = 8
    x = torch.randn(batch, in_channels, h, w, device='cuda', dtype=torch.float32)
    k_size = 3
    # Intentionally supply a depthwise weight with a wrong channel count (in_channels+1 instead of in_channels)
    depthwise_weight = torch.randn(in_channels + 1, 1, k_size, k_size, device='cuda', dtype=torch.float32)
    # Correct pointwise weight shape is assumed: [out_channels, in_channels]
    pointwise_weight = torch.randn(64, in_channels, device='cuda', dtype=torch.float32)
    depthwise_bias = torch.randn(in_channels, device='cuda', dtype=torch.float32)
    pointwise_bias = torch.randn(64, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # If the kernel does not force a cudaDeviceSynchronize and catch errors, the out-of-bound access
        # will eventually trigger a runtime error when synchronizing.
        out = kernel_module.forward(x, depthwise_weight, pointwise_weight,
                                    depthwise_bias, pointwise_bias, 1, 0, 1)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input_issue():
    """
    Issue 2: The kernel assumes contiguous memory. By passing a non-contiguous tensor (via a transpose)
    the manual indexing will be incorrect and the kernel output will differ from that of a correct implementation.
    """
    kernel_module = build_kernel()
    batch = 1
    in_channels = 3
    h = 16
    w = 16
    # Create a contiguous input and then make it non-contiguous by transposing spatial dimensions.
    x = torch.randn(batch, in_channels, h, w, device='cuda', dtype=torch.float32)
    x_non_contig = x.transpose(2, 3)  # now non-contiguous
    k_size = 3
    depthwise_weight = torch.randn(in_channels, 1, k_size, k_size, device='cuda', dtype=torch.float32)
    pointwise_weight = torch.randn(64, in_channels, device='cuda', dtype=torch.float32)
    depthwise_bias = torch.randn(in_channels, device='cuda', dtype=torch.float32)
    pointwise_bias = torch.randn(64, device='cuda', dtype=torch.float32)
    # Run the custom kernel
    out = kernel_module.forward(x_non_contig, depthwise_weight, pointwise_weight,
                                depthwise_bias, pointwise_bias, 1, 1, 1)
    
    # Build a reference model that uses PyTorch’s built-in layers.
    # Note: PyTorch layers automatically handle non-contiguous inputs.
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, in_channels, k_size, padding=1, groups=in_channels, bias=True),
        torch.nn.Conv2d(in_channels, 64, kernel_size=1, bias=True)
    ).to('cuda')
    with torch.no_grad():
        # Copy the weights to the reference model (adjusting the dimensions appropriately).
        model[0].weight.copy_(depthwise_weight.squeeze(1))
        model[0].bias.copy_(depthwise_bias)
        model[1].weight.copy_(pointwise_weight)
        model[1].bias.copy_(pointwise_bias)
    out_ref = model(x_non_contig)
    # Since the custom kernel does not account for non-contiguous strides,
    # its output is expected to not match the reference.
    assert not torch.allclose(out, out_ref, atol=1e-5), \
        "Kernel unexpectedly handled non-contiguous input correctly."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_half_precision_issue():
    """
    Issue 3: The kernel uses AT_DISPATCH_FLOATING_TYPES which excludes half precision.
    Passing float16 tensors should trigger a runtime error.
    """
    kernel_module = build_kernel()
    batch = 1
    in_channels = 3
    h = 16
    w = 16
    k_size = 3
    x = torch.randn(batch, in_channels, h, w, device='cuda', dtype=torch.float16)
    depthwise_weight = torch.randn(in_channels, 1, k_size, k_size, device='cuda', dtype=torch.float16)
    pointwise_weight = torch.randn(64, in_channels, device='cuda', dtype=torch.float16)
    depthwise_bias = torch.randn(in_channels, device='cuda', dtype=torch.float16)
    pointwise_bias = torch.randn(64, device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError):
        out = kernel_module.forward(x, depthwise_weight, pointwise_weight,
                                    depthwise_bias, pointwise_bias, 1, 1, 1)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_weight_shape_issue():
    """
    Issue 4: The kernel assumes that weight tensors have a fixed shape.
    By deliberately providing a pointwise weight tensor with an incorrect second dimension,
    we trigger an out-of-bound memory access.
    """
    kernel_module = build_kernel()
    batch = 1
    in_channels = 3
    h = 16
    w = 16
    k_size = 3
    x = torch.randn(batch, in_channels, h, w, device='cuda', dtype=torch.float32)
    depthwise_weight = torch.randn(in_channels, 1, k_size, k_size, device='cuda', dtype=torch.float32)
    # Provide pointwise weight with an incorrect input channel dimension (e.g., in_channels+1)
    pointwise_weight = torch.randn(64, in_channels + 1, device='cuda', dtype=torch.float32)
    depthwise_bias = torch.randn(in_channels, device='cuda', dtype=torch.float32)
    pointwise_bias = torch.randn(64, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = kernel_module.forward(x, depthwise_weight, pointwise_weight,
                                    depthwise_bias, pointwise_bias, 1, 1, 1)
        torch.cuda.synchronize()
