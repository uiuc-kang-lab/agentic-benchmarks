
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Ensure that the kernel.cu file is in the current directory
_KERNEL_FILE = os.path.join(os.path.dirname(__file__), "kernel.cu")

def build_kernel():
    cuda_module = load(
        name="test_cuda_conv1d",
        sources=[_KERNEL_FILE],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to simulate a 1D convolution using torch.nn.functional.conv1d
def ref_conv1d(x, weight, bias, stride, dilation):
    return torch.nn.functional.conv1d(x, weight, bias, stride=stride, dilation=dilation)

# Issue 1: Limited grid configuration (max_blocks capped).
# Test: Use an input where the total number of output elements is huge.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_tensor_max_blocks():
    # Large tensor parameters: use a large batch and long spatial dimension
    B = 128
    in_channels = 3
    out_channels = 64
    in_size = 4096  # long input signal
    kernel_size = 3
    stride = 1
    dilation = 1

    # This setting will produce a huge number of output elements: 128*64*4094
    x = torch.randn(B, in_channels, in_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = None  # test without bias

    cuda_module = build_kernel()
    # run the custom CUDA kernel
    out_cuda = cuda_module.forward(x, weight, bias, stride, dilation)
    # run reference conv1d from PyTorch
    out_ref = ref_conv1d(x, weight, bias, stride, dilation)
    torch.cuda.synchronize()
    assert torch.allclose(out_cuda, out_ref, atol=1e-4), "Output mismatch in large-tensor test (grid config issue)"

# Issue 2: Use of runtime-dependent unrolling.
# Test: Use a larger kernel_size to force the loop bounds to be less amenable to compile-time unrolling.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_kernel_size_unroll():
    B = 4
    in_channels = 3
    out_channels = 8
    in_size = 64
    kernel_size = 17  # large kernel size that does not easily unroll
    stride = 1
    dilation = 1

    x = torch.randn(B, in_channels, in_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    cuda_module = build_kernel()
    out_cuda = cuda_module.forward(x, weight, bias, stride, dilation)
    out_ref = ref_conv1d(x, weight, bias, stride, dilation)
    torch.cuda.synchronize()
    assert torch.allclose(out_cuda, out_ref, atol=1e-4), "Output mismatch with large kernel size (unroll issue)"

# Issue 3: Hard-coded support for float32 only.
# Test: Provide double precision inputs and check that the kernel raises an error.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dtype_support():
    B = 2
    in_channels = 3
    out_channels = 4
    in_size = 32
    kernel_size = 3
    stride = 1
    dilation = 1

    # Create double precision tensors.
    x = torch.randn(B, in_channels, in_size, device="cuda", dtype=torch.double)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.double)
    bias = None

    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the TORCH_CHECK in forward() to trigger for wrong dtype.
        cuda_module.forward(x, weight, bias, stride, dilation)

# Issue 4: Contiguity assumptions.
# Test: Pass non-contiguous input and weight tensors to trigger TORCH_CHECK failures.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_inputs():
    B = 2
    in_channels = 3
    out_channels = 4
    in_size = 32
    kernel_size = 3
    stride = 1
    dilation = 1

    x = torch.randn(B, in_channels, in_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    # Make x and weight non-contiguous by transposing a dimension and then restoring shape.
    x_noncontig = x.transpose(1, 2).contiguous().transpose(1, 2)
    weight_noncontig = weight.transpose(1, 2).contiguous().transpose(1, 2)

    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        cuda_module.forward(x_noncontig, weight, bias, stride, dilation)
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, weight_noncontig, bias, stride, dilation)
