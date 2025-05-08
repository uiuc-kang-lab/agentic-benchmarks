
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Kernel only supports float32.
def test_dtype_issue():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    length = 100
    stride = 1
    dilation = 1

    # Create input tensors with float64 data type.
    x = torch.randn(batch_size, in_channels, length, dtype=torch.float64, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.float64, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda")
    
    # Since the kernel code forcibly casts data_ptr<float>(), using float64 inputs will yield wrong results.
    # We detect this by comparing with PyTorch's native Conv1d (which casts properly).
    conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=True).to(device="cuda", dtype=torch.float64)
    # Force the conv weight and bias to be the same so that any difference comes solely from the kernel.
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    
    # Call the optimized cuda kernel function.
    # Note: The kernel does not check tensor dtype, so it will interpret the double data as float.
    out_cuda = cuda_module.forward(x, weight, bias, stride, dilation)
    out_ref = conv(x)
    
    # Expect that the results do not match due to wrong interpretation of the data (precision bug).
    assert not torch.allclose(out_cuda, out_ref, atol=1e-5), "Kernel incorrectly handled non-float32 tensors!"

# Issue 2: Kernel requires contiguous tensors.
def test_noncontiguous_input():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    length = 100
    stride = 1
    dilation = 1
    
    # Create a contiguous tensor, then make it noncontiguous.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32).transpose(1, 2)
    # weight and bias remain contiguous.
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError, match="x must be contiguous"):
        cuda_module.forward(x, weight, bias, stride, dilation)
    
# Issue 3: Shared memory allocation size may exceed hardware limits.
def test_excessive_shared_memory():
    cuda_module = build_kernel()
    # Choose parameters that force a very large shared memory allocation.
    # For example, a very high number of input channels and/or kernel size.
    batch_size = 1
    in_channels = 2048   # Very large number of channels.
    out_channels = 4
    kernel_size = 64     # Large kernel.
    length = 1024
    stride = 1
    dilation = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # The kernel launch should fail if the shared memory request exceeds the device limit.
    with pytest.raises(RuntimeError, match="CUDA kernel launch error"):
        cuda_module.forward(x, weight, bias, stride, dilation)

# Issue 4: Kernel does not support grouped convolution or alternative weight layouts.
def test_weight_channel_mismatch():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    length = 100
    stride = 1
    dilation = 1
    
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    # Construct weight with a mismatched in_channel dimension (simulate grouped conv weight layout).
    # E.g., weight expected shape for standard conv is (out_channels, in_channels, kernel_size)
    # Here we deliberately set the second dimension to a different value.
    weight = torch.randn(out_channels, in_channels // 2, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError, match="Input channels mismatch"):
        cuda_module.forward(x, weight, bias, stride, dilation)

# Issue 5: Use of "#pragma unroll" assumes compile-time constant kernel_size.
# This issue may not trigger a runtime error, but we provide a test to pass a weight with runtime kernel_size.
def test_runtime_kernel_size():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 3
    out_channels = 8
    # Kernel size determined at runtime, e.g., from a variable.
    kernel_size = int(torch.randint(2, 7, (1,)).item())
    length = 100
    stride = 1
    dilation = 1
    
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # This test is not expected to raise an error, but it shows that the assumption of compile-time kernel_size
    # (due to #pragma unroll) may not hold. We compare against PyTorch's Conv1d result.
    conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=True).to("cuda")
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    out_cuda = cuda_module.forward(x, weight, bias, stride, dilation)
    out_ref = conv(x)
    # The outputs might not be equal if unrolling assumptions lead to miscompiled code.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-5), "Kernel unexpectedly handled runtime kernel_size as if it were constant!"

