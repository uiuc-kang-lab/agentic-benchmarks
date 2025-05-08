
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Load the CUDA extension from the kernel.cu file.
    # Adjust extra_cuda_cflags if needed.
    return load(
        name="max_pool2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fallback_kernel_size():
    # This test triggers Issue 1: using a kernel size (e.g., 4) that is not 2 or 3.
    # Because the fallback instantiation uses KERNEL_SIZE = -1, the pooling loops never run,
    # and the output is left as -infinity.
    kernel_size = 4   # Unsupported kernel size for our kernel’s template specializations.
    stride = 2
    padding = 1
    dilation = 1

    batch_size = 2
    channels = 3
    height = 16
    width = 16

    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    
    # Use PyTorch's MaxPool2d as the reference.
    ref_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    expected_output = ref_pool(input_tensor)
    
    module = build_kernel()
    # Call the custom CUDA kernel.
    output = module.forward(input_tensor, kernel_size, stride, padding, dilation)
    
    # In the fallback path the loops do not execute and max_val remains -infinity.
    # Thus, we expect all output elements to be -infinity.
    if not torch.isinf(output).all():
        raise AssertionError("Fallback kernel for kernel_size != 2 and != 3 did not produce -infinity output as expected.")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_double_precision():
    # This test triggers Issue 2: using double precision input with a kernel that applies fmaxf.
    # fmaxf is only for float and will produce incorrect results for double.
    kernel_size = 2    # Supported by the specializations.
    stride = 2
    padding = 1
    dilation = 3

    batch_size = 2
    channels = 3
    height = 16
    width = 16

    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.double)
    
    # Get the expected output using PyTorch's native implementation.
    ref_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    expected_output = ref_pool(input_tensor)
    
    module = build_kernel()
    # Call the custom CUDA kernel.
    output = module.forward(input_tensor, kernel_size, stride, padding, dilation)
    
    # Because using fmaxf on double precision is not correct, the output will differ from the expected result.
    # We check that they are not almost equal.
    if torch.allclose(output, expected_output, atol=1e-5):
        raise AssertionError("Kernel output in double precision unexpectedly matches the expected output; fmaxf may have been used erroneously.")
