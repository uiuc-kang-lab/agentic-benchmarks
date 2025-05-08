
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="conv_transpose_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_float_type():
    """
    Test for Issue #1: The kernel only supports float32.
    Passing double precision tensors should trigger an error.
    """
    mod = build_kernel()
    batch_size = 1
    in_channels = 3
    out_channels = 3
    input_length = 5
    kernel_size = 3
    stride, padding, dilation = 1, 0, 1

    # Create double (float64) tensors.
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.double, device="cuda")
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.double, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.double, device="cuda")
    
    with pytest.raises(RuntimeError):
        # This should fail because the kernel internally calls data_ptr<float>().
        mod.forward(x, weight, bias, stride, padding, dilation)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_shared_memory_exceed():
    """
    Test for Issue #2: The kernel assumes that the entire weight tensor fits in shared memory.
    By creating a very large weight tensor, we force the shared memory requirement to exceed the limit.
    """
    mod = build_kernel()
    batch_size = 1
    # Use large dimensions to force high shared memory usage.
    in_channels = 1024  
    out_channels = 1024
    kernel_size = 7
    input_length = 10
    stride, padding, dilation = 1, 0, 1

    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    
    with pytest.raises(RuntimeError):
        # Likely to trigger an error due to excessive shared memory allocation.
        mod.forward(x, weight, bias, stride, padding, dilation)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_grouped_convolution():
    """
    Test for Issue #3: The kernel does not support grouped convolutions.
    We simulate a grouped convolution by providing a weight tensor with mismatched dimensions.
    """
    mod = build_kernel()
    batch_size = 1
    in_channels = 4         # suppose we intend groups=2
    out_channels = 8
    input_length = 10
    kernel_size = 3
    stride, padding, dilation = 1, 0, 1

    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
    # For a grouped convolution with groups=2, the weight tensor should be of shape 
    # (in_channels, out_channels/groups, kernel_size), i.e., (4, 4, kernel_size).
    # But our kernel expects the weight's first dimension to equal in_channels.
    # Here, we intentionally use an incorrect shape to simulate grouped conv usage.
    weight = torch.randn(in_channels // 2, out_channels, kernel_size, dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    
    with pytest.raises(RuntimeError):
        mod.forward(x, weight, bias, stride, padding, dilation)
