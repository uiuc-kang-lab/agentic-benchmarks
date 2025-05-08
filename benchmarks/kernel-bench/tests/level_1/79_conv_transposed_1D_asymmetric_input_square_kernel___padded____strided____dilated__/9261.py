
import pytest
import torch
from torch.utils.cpp_extension import load

# This helper builds the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that passing double (float64) tensors leads to incorrect results.
# The kernel never checks the dtype and will interpret the data as float,
# so we expect the output from the custom CUDA kernel to differ significantly from PyTorch's correct result.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_dtype():
    # Create tensors with dtype double
    batch_size = 2
    in_channels = 3
    out_channels = 4
    input_length = 10
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create input and weight in double precision.
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.double, device='cuda')
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.double, device='cuda')
    bias = torch.randn(out_channels, dtype=torch.double, device='cuda')
    
    # Build kernel module and run custom CUDA kernel which blindly uses float pointer casts.
    mod = build_kernel()
    with pytest.raises(Exception) as excinfo:
        # Calling forward; the kernel uses data_ptr<float>() even though the data is double.
        out = mod.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()
        # For a correct conv-transpose, PyTorch's nn.ConvTranspose1d should produce a different result.
        conv_transpose = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=True
        ).to('cuda', torch.double)
        # Manually set conv_transpose weight and bias to match our input.
        conv_transpose.weight.data = weight
        conv_transpose.bias.data = bias
        out_ref = conv_transpose(x)
        # Expect the result from the custom kernel to be far off
        assert not torch.allclose(out, out_ref, atol=1e-5), "Kernel erroneously produced correct output for double dtype"
        
    # If no exception is raised, we manually check the discrepancy.
    try:
        out = mod.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()
        conv_transpose = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=True
        ).to('cuda', torch.double)
        conv_transpose.weight.data = weight
        conv_transpose.bias.data = bias
        out_ref = conv_transpose(x)
        # The results should differ significantly because the kernel misinterprets double as float.
        diff = (out.to(torch.double) - out_ref).abs().max().item()
        assert diff > 1e-3, f"Kernel output appears accurate (diff={diff}); expected inaccuracy due to dtype issue"
    except Exception:
        pass

# Issue 2: The kernel does not support output_padding.
# We simulate this by comparing the output shapes.
# In many generalized transposed convolution implementations, an extra output_padding parameter is allowed.
# Since our kernel does not handle it, if a caller intends to use output_padding,
# the output shape will be different than expected.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_missing_output_padding_support():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    input_length = 8
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    output_padding = 1  # Extra output padding desired

    # Create input and weight in float32.
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device='cuda')
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float32, device='cuda')
    bias = torch.randn(out_channels, dtype=torch.float32, device='cuda')

    # Build kernel module and run custom CUDA kernel.
    mod = build_kernel()
    # Our kernel does not support output_padding, so we must ignore it.
    out_custom = mod.forward(x, weight, bias, stride, padding, dilation)
    torch.cuda.synchronize()

    # Use PyTorch's ConvTranspose1d which supports output_padding.
    conv_transpose = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=True,
        output_padding=output_padding
    ).to('cuda', torch.float32)
    conv_transpose.weight.data = weight
    conv_transpose.bias.data = bias
    out_ref = conv_transpose(x)
    torch.cuda.synchronize()

    # The shapes should differ because custom kernel ignores output_padding.
    assert out_custom.shape != out_ref.shape, \
        f"Output shape matches reference despite output_padding; custom shape: {out_custom.shape}, ref shape: {out_ref.shape}"

# Issue 3: No error checking after kernel launch.
# Although it is hard to trigger asynchronous CUDA errors deliberately,
# we can at least check that the function properly throws an error when passed a CPU tensor.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_on_wrong_device():
    batch_size = 2
    in_channels = 3
    out_channels = 4
    input_length = 10
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create input tensor on CPU
    x_cpu = torch.randn(batch_size, in_channels, input_length, device='cpu', dtype=torch.float32)
    weight_cpu = torch.randn(in_channels, out_channels, kernel_size, device='cpu', dtype=torch.float32)
    bias_cpu = torch.randn(out_channels, device='cpu', dtype=torch.float32)

    mod = build_kernel()
    with pytest.raises(AssertionError):
        # The TORCH_CHECK in forward_cuda should trigger an error since x is not on CUDA.
        mod.forward(x_cpu, weight_cpu, bias_cpu, stride, padding, dilation)
