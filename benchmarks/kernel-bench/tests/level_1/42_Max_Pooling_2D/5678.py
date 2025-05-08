
import pytest
import torch
from torch.utils.cpp_extension import load

# Build and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="maxpool_cuda_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference max pooling using PyTorch's own nn.MaxPool2d.
def ref_maxpool2d(input, kernel_size, stride, padding, dilation):
    maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation)
    return maxpool(input)

# Test case 1: Dynamic (unsupported) kernel size.
# Description: Using a kernel_size (e.g. 5) different from 2 or 3 should trigger a problem,
# since the templated loops will not iterate correctly.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dynamic_kernel_size():
    cuda_mod = build_kernel()
    batch_size = 2
    channels = 3
    input_height = 16
    input_width = 16
    # Use a kernel_size that is not 2 or 3 to trigger the default branch.
    kernel_size = 5  
    stride = 1
    padding = 1
    dilation = 1

    # Create input tensor
    input_tensor = torch.randn(batch_size, channels, input_height, input_width, device='cuda', dtype=torch.float32)
    # Compute reference result using PyTorch MaxPool2d
    ref_output = ref_maxpool2d(input_tensor, kernel_size, stride, padding, dilation)

    # Run the CUDA kernel
    output = cuda_mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # Since the kernel implementation is not dynamic, the result is likely wrong.
    # We test that the CUDA kernel output DOES NOT match the reference.
    assert not torch.allclose(output, ref_output, atol=1e-5), \
        "Test failed: Dynamic kernel_size (unsupported) did not trigger an error."

# Test case 2: Double precision input.
# Description: The kernel uses fmaxf throughout. When provided with a double precision tensor,
# using fmaxf instead of fmax should cause an error or produce incorrect output.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_double_precision():
    cuda_mod = build_kernel()
    batch_size = 2
    channels = 3
    input_height = 16
    input_width = 16
    kernel_size = 2
    stride = 1
    padding = 1
    dilation = 1

    input_tensor = torch.randn(batch_size, channels, input_height, input_width, device='cuda', dtype=torch.double)
    ref_output = ref_maxpool2d(input_tensor, kernel_size, stride, padding, dilation)

    # We expect the use of fmaxf with double might lead to incorrect behavior.
    output = cuda_mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # Check if output is significantly different from reference.
    max_diff = (output - ref_output).abs().max()
    assert max_diff > 1e-3, \
        f"Test failed: Using double precision did not trigger an error. Max difference: {max_diff}"

