
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # The extension will compile kernel.cu into a module named "cuda_maxpool"
    cuda_module = load(
        name="cuda_maxpool",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_issue_double_precision():
    # Issue 1: Using double leads to the invocation of fmaxf
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    # Create an input tensor of type double
    input_tensor = torch.randn(1, 1, 16, 16, dtype=torch.double, device='cuda')
    mod = build_kernel()
    # Expect that the kernel mishandles double (either by error or wrong result)
    output = mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    # Compute the expected output using PyTorch's max_pool2d
    expected = torch.nn.functional.max_pool2d(input_tensor, kernel_size,
                                               stride=stride, padding=padding,
                                               dilation=dilation)
    # We expect a deviation because fmaxf cannot handle double correctly.
    assert not torch.allclose(output, expected, atol=1e-6), (
        "Double type should trigger an issue with using fmaxf instead of fmax!"
    )

def test_issue_kernel_size_fallback():
    # Issue 2: Using a kernel_size other than 2 or 3 triggers the fallback branch
    # which was instantiated with KERNEL_SIZE = -1, causing the loops not to execute.
    kernel_size = 4  # This will choose the fallback branch
    stride = 2
    padding = 0
    dilation = 1
    input_tensor = torch.randn(1, 1, 16, 16, dtype=torch.float32, device='cuda')
    mod = build_kernel()
    output = mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    expected = torch.nn.functional.max_pool2d(input_tensor, kernel_size,
                                               stride=stride, padding=padding,
                                               dilation=dilation)
    # Because the loops never run properly, output should remain at the initial value, i.e. -infinity.
    # We check that every output element is still -infinity.
    assert torch.all(output == float("-inf")), (
        "Kernel fallback for kernel_size!=2/3 should yield -inf values because the loop did not run."
    )

def test_issue_unsigned_index():
    # Issue 3: Bad handling of indices due to conversion to unsigned.
    # Excessive padding leads to negative starting indices, which when converted to unsigned
    # become huge and bypass the intended boundary checks.
    kernel_size = 2
    stride = 1
    padding = 2  # This padding will cause ih_base and iw_base to be negative.
    dilation = 1
    input_tensor = torch.randn(1, 1, 8, 8, dtype=torch.float32, device='cuda')
    mod = build_kernel()
    output = mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    expected = torch.nn.functional.max_pool2d(input_tensor, kernel_size,
                                               stride=stride, padding=padding,
                                               dilation=dilation)
    # Due to unsigned conversion issues, the computed output will be incorrect.
    assert not torch.allclose(output, expected, atol=1e-5), (
        "Kernel did not exhibit the unsigned index error when excessive padding is applied!"
    )
