
import torch
import pytest
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

def test_half_precision_not_supported():
    # Issue 1: The kernel does not support half precision.
    cuda_module = build_kernel()
    # Create a half precision input tensor.
    x = torch.randn(1, 1, 4, 4, device="cuda", dtype=torch.float16)
    # Expect that launching the kernel with half will raise a RuntimeError.
    with pytest.raises(RuntimeError):
        # The following call should fail because AT_DISPATCH_FLOATING_TYPES excludes half.
        _ = cuda_module.forward(x, 2, 2, 1, 1)

def test_missing_cuda_launch_error_check():
    # Issue 2: No error checking after kernel launch.
    # We attempt to trigger a kernel error by passing an invalid parameter.
    cuda_module = build_kernel()
    # Use an invalid dilation (negative value) which will likely cause invalid memory accesses.
    x = torch.randn(1, 1, 8, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x, 2, 2, 1, -1)
        # Force synchronization to trigger any asynchronous CUDA errors.
        torch.cuda.synchronize()

def test_non_square_parameter_limitation():
    # Issue 3: The kernel is hard-coded for square pooling parameters.
    # We simulate a scenario where a user might want rectangular pooling.
    # Although both PyTorch and the custom kernel accept a single integer for kernel_size and stride,
    # a user wishing to perform rectangular pooling (e.g. different stride in height and width)
    # will get incorrect results from the custom kernel.
    cuda_module = build_kernel()
    # Create an input tensor.
    x = torch.randn(1, 1, 10, 15, device="cuda", dtype=torch.float32)
    # Our custom kernel only accepts a single stride value, so we choose stride=2.
    out_custom = cuda_module.forward(x, 3, 2, 1, 1)
    # PyTorch’s max_pool2d supports rectangular stride via a tuple.
    # Here we intentionally create a discrepancy: the user intended stride=(2,1) but the kernel can’t handle it.
    out_pytorch = torch.nn.functional.max_pool2d(x, kernel_size=(3, 3), stride=(2, 1), padding=1)
    # The outputs should differ because the custom kernel cannot support non-square stride.
    assert not torch.allclose(out_custom, out_pytorch), \
        "Custom kernel unexpectedly matched PyTorch output for non-square pooling parameters."

def test_ambiguous_max_function():
    # Issue 4: Using the ambiguous 'max' function inside the CUDA kernel.
    # We test a simple input where we know the maximum value.
    cuda_module = build_kernel()
    # Create an input tensor with a clearly defined maximum in one pooling window.
    x = torch.tensor([[[[1, 2, 10, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]]]], device="cuda", dtype=torch.float32)
    # Use pooling parameters such that the pooling window covers all input elements.
    out_custom = cuda_module.forward(x, 4, 1, 0, 1)  # kernel_size=4, stride=1, no padding, dilation=1
    expected = torch.tensor([[[[16]]]], device="cuda", dtype=torch.float32)
    # If using the wrong max function, the computed maximum may be incorrect.
    assert torch.allclose(out_custom, expected), \
        f"Custom kernel output {out_custom} does not match expected {expected}."

