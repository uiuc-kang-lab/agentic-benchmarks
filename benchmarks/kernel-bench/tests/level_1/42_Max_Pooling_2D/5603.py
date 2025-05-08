
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_maxpool",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_standard_kernel_size():
    """
    Test for issue #1.
    When kernel_size != 2 or 3 (here, 4 is used), the kernel instantiation uses a template parameter of -1.
    This causes the inner loops to be skipped – the max remains -infinity rather than reflecting the true max value.
    """
    batch_size, channels, height, width = 1, 1, 8, 8
    kernel_size = 4  # Non 2 or 3 size triggers the issue.
    stride = 2
    padding = 1
    dilation = 1

    # Create input tensor with all positive values so a proper max would be > -inf.
    input_tensor = torch.ones(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    
    # Expected result using PyTorch native max pooling (which would normally give a tensor of 1's)
    pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    expected = pool(input_tensor)

    custom_module = build_kernel()
    output = custom_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    
    # Because of the bug the kernel never updates max_val, so it remains -infinity everywhere.
    # We check if output equals -infinity.
    assert torch.all(output == float("-inf")), (
        "Custom kernel did not produce -infinity output for kernel_size !=2 and !=3. "
        "This indicates that the loops did not execute as expected."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_double_input_with_kernel_size_2():
    """
    Test for issue #2.
    When using kernel_size == 2 with double precision input,
    the kernel incorrectly uses fmaxf (a float intrinsic) rather than an appropriate double version,
    leading to incorrect (or imprecise) results.
    """
    batch_size, channels, height, width = 1, 1, 8, 8
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    
    # Create a double-precision input tensor with distinct values.
    # Using a range ensures the max operation is non-trivial.
    input_tensor = torch.linspace(0, 100, steps=batch_size*channels*height*width,
                                  device="cuda", dtype=torch.float64).reshape(batch_size, channels, height, width)
    
    # Expected result computed by PyTorch's built-in functionality for double tensors.
    expected = torch.nn.functional.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride, 
                                                padding=padding, dilation=dilation)
    
    custom_module = build_kernel()
    output = custom_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    
    # If fmaxf is used, the computation may happen in lower precision or incorrectly.
    # We check that the error is larger than a tight tolerance.
    max_diff = (output - expected).abs().max().item()
    assert max_diff > 1e-5, (
        "Custom kernel produced results too close to expected for double inputs with kernel_size 2. "
        "This suggests that the issue with using fmaxf may not be triggering as expected."
    )
