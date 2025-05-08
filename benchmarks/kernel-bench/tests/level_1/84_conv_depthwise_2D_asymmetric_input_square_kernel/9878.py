
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="depthwise_conv_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Lack of CUDA error checking after kernel launch.
# We simulate this by providing an intentionally mismatched weight tensor shape that leads to illegal memory accesses.
# With proper error checking, the cuda kernel launch would report an error during synchronization.
def test_kernel_launch_error_due_to_weight_shape():
    # Use contiguous float tensor inputs
    batch_size = 4
    in_channels = 3
    input_h = 16
    input_w = 16
    kernel_size = 3
    stride = 1
    padding = 0

    # Create input tensor (float32, contiguous)
    input_tensor = torch.randn(batch_size, in_channels, input_h, input_w, device="cuda", dtype=torch.float32)
    
    # Intentionally create a mismatched weight tensor shape.
    # Expected weight shape for depthwise conv: (in_channels, channels_per_group, kernel_size, kernel_size)
    # Here, we provide a weight with a wrong kernel size (kernel_size+1) so that the kernel accesses out‐of‐bounds indices.
    # This should trigger an error in the CUDA kernel when an out-of-bound access occurs.
    wrong_kernel_size = kernel_size + 1
    weight_tensor = torch.randn(in_channels, 1, wrong_kernel_size, wrong_kernel_size, device="cuda", dtype=torch.float32)
    
    # Provide bias as None for simplicity
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        output = cuda_module.forward(input_tensor, weight_tensor, None, stride, padding)
        # Force synchronization to catch launch or memory access errors.
        torch.cuda.synchronize()

# Issue 2: The kernel assumes all input tensors are contiguous and of type float32.
# Provide a non-contiguous weight tensor and a double-precision input tensor to trigger the error checks.
def test_non_contiguous_and_wrong_dtype():
    batch_size = 4
    in_channels = 3
    input_h = 16
    input_w = 16
    kernel_size = 3
    stride = 1
    padding = 0

    # Create input tensor with wrong data type (float64) and contiguous.
    input_tensor = torch.randn(batch_size, in_channels, input_h, input_w, device="cuda", dtype=torch.float64)
    
    # Create a weight tensor with correct shape but then make it non-contiguous.
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32).transpose(1, 2)
    
    cuda_module = build_kernel()
    # Check that the forward function raises an error for non-float32 dtype and/or non-contiguous tensor.
    with pytest.raises(RuntimeError):
        output = cuda_module.forward(input_tensor, weight_tensor, None, stride, padding)
        torch.cuda.synchronize()
