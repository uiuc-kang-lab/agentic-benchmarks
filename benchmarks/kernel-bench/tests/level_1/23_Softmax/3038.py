
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="softmax_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Issue 1: Using "max" instead of fmaxf may not be caught by standard tests when inputs are "normal"
# because in many cases the result might appear correct.
# To stress the reduction and catch any potential miscomparison in edge cases,
# we use extreme values that force the reduction to operate in unusual ranges.
def test_extreme_values(kernel_module):
    # Create a tensor with very large negative numbers in one row,
    # except one element which is much higher; the softmax should pick that element.
    batch_size = 1
    num_features = 1024
    x = torch.full((batch_size, num_features), -1e20, device='cuda', dtype=torch.float32)
    x[0, 500] = 1.0  # the maximum value
    y = kernel_module.forward(x)
    # The output at index 500 should be nearly 1, and the others nearly 0.
    assert torch.argmax(y, dim=1).item() == 500, (
        f"Softmax maximum index expected to be 500, got {torch.argmax(y, dim=1).item()}"
    )

# Issue 2: Missing kernel launch error checking.
# Although the kernel wrapper does not check errors,
# we simulate a misuse scenario by passing an input that should trigger a TORCH_CHECK failure.
def test_wrong_device(kernel_module):
    # Pass a CPU tensor instead of a CUDA tensor to trigger the TORCH_CHECK.
    x_cpu = torch.randn(16, 1024, device='cpu', dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        _ = kernel_module.forward(x_cpu)

# Issue 3: Kernel supports only float32. Passing a tensor of another type (e.g. float64) should trigger an error.
def test_wrong_dtype(kernel_module):
    x_double = torch.randn(16, 1024, device='cuda', dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Input tensor must be float32"):
        _ = kernel_module.forward(x_double)
