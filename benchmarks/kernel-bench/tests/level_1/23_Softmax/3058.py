
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the extension module from kernel.cu.
    # Note: The kernel.cu file must be in the current directory.
    cuda_module = load(
        name="softmax_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test for Issue 1 and Issue 2:
# We try to simulate a “non-standard” launch by sending an input type/shape that would be
# unexpected if the kernel were later adapted to use other block sizes. While we cannot force a
# different block size from Python in the current extension (because it is hard-coded),
# we can at least trigger the related input checks.
def test_wrong_dtype():
    my_module = build_kernel()
    # Create a double tensor instead of float32.
    x = torch.randn(16, 16384, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor\\.|Input tensor must be float32."):
        # The binding in forward() expects float32 so this should raise an error.
        my_module.forward(x)

def test_wrong_dimensions():
    my_module = build_kernel()
    # Create a 3D tensor; the kernel forward() expects a 2D tensor.
    x = torch.randn(4, 16, 16384, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be 2D."):
        my_module.forward(x)

# Test for Issue 3:
# Because the kernel launch does not call cudaDeviceSynchronize, some asynchronous errors might
# be undetected. We can work around this by forcing an error condition.
# Here we deliberately pass a tensor on the wrong device to force an error in the kernel launch.
def test_async_error_detection():
    my_module = build_kernel()
    # Create a CPU tensor instead of a CUDA tensor. The check in forward() should catch it.
    x = torch.randn(16, 16384, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor."):
        my_module.forward(x)

# Test for Issue 4:
# Pass an input tensor with -INFINITY values that will cause exponentials to be 0.0 and hence a sum of 0.
# This should produce a NaN output due to division by 0.
def test_division_by_zero():
    my_module = build_kernel()
    # Create an input where all values are -INFINITY
    x = torch.full((16, 16384), float("-inf"), device="cuda", dtype=torch.float32)
    y = my_module.forward(x)
    # The output should be NaN because sum of exp(x-max) is 0, and division by zero occurs.
    # Check that at least one output element is NaN.
    assert torch.isnan(y).any(), "Expected NaN values due to division by zero but none were found."

# Test for Issue 5:
# Provide an empty tensor to check that the kernel is robust (or fails explicitly) when an empty input is given.
def test_empty_tensor():
    my_module = build_kernel()
    # Create an empty tensor with shape (0, 16384)
    x = torch.empty((0, 16384), device="cuda", dtype=torch.float32)
    # Depending on how one wants to handle empty tensors, this might either be allowed to pass through
    # (with an empty output) or raise an error. Here we check that the output is empty.
    y = my_module.forward(x)
    assert y.numel() == 0, "Expected an empty output tensor for an empty input."

