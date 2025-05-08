
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2 Test:
# Test correctness: compare our kernel's output with PyTorch's native max_pool2d.
# With a controlled input, if the negative-infinity initialization
# or improper max() implementation is causing trouble, the results will differ.
def test_max_pool2d_correctness():
    # Use a moderate input size.
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    input_tensor = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float32)
    # Get output via our CUDA kernel (wrapped via our custom extension)
    module = build_kernel()
    output_custom = module.forward(input_tensor, kernel_size, stride, padding, dilation)
    
    # Compute PyTorch's reference result
    output_ref = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    # They should be exactly equal within tolerance
    torch.cuda.synchronize()
    assert torch.allclose(output_custom, output_ref, atol=1e-4), (
        f"Custom kernel output does not match PyTorch reference output!"
        f" Max diff = {(output_custom - output_ref).abs().max()}"
    )

# Issue 3 Test:
# Create an input tensor with such a large batch_size*channels that gridDim.z > 65535.
# Expect a CUDA kernel launch error.
def test_exceed_grid_dimension():
    # Choose batch_size and channels such that batch_size*channels > 65535
    batch_size = 300
    channels = 256   # 300*256 = 76800 > 65535
    height = 32
    width = 32
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    input_tensor = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the grid.z dimension exceeds the CUDA limit.
        _ = module.forward(input_tensor, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 4 Test:
# Use a half-precision tensor; the AT_DISPATCH_FLOATING_TYPES macro does not support half.
# Expect a RuntimeError (or similar exception) due to unsupported type.
def test_unsupported_tensor_dtype():
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    input_tensor = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.half)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should throw an error because half-precision is not dispatched.
        _ = module.forward(input_tensor, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()

# Additional test for integer types (if somebody mistakenly passes integer data)
def test_integer_tensor_dtype():
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    input_tensor = torch.randint(0, 10, (batch_size, channels, height, width), device='cuda', dtype=torch.int32)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(input_tensor, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()

