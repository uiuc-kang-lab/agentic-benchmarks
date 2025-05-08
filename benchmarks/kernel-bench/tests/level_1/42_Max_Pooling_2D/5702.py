
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

# Test case 1: Triggering the unsupported kernel_size issue.
# When kernel_size is not 2 or 3 (e.g., 4), the kernel will use a fallback instantiation,
# leading to an invalid loop unrolling (using a negative loop bound) and therefore incorrect results.
def test_unsupported_kernel_size():
    batch_size = 4
    channels = 3
    height = 16
    width = 16
    kernel_size = 4  # not supported in the provided kernel code (only 2 and 3 are explicitly handled)
    stride = 2
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    output = my_module.forward(x, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # Since the fallback kernel instantiation uses KERNEL_SIZE = -1,
    # the kernel loops won't execute and the output will remain at the initialized (garbage) state.
    # We thus expect that the output does not match the correct output computed using PyTorch's max_pool2d.
    correct_output = torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    # Check that the outputs are not close, thus triggering the issue.
    assert not torch.allclose(output, correct_output, atol=1e-5), "Kernel incorrectly handled an unsupported kernel_size."

# Test case 2: Triggering the use of fmaxf with a double precision tensor.
# The AT_DISPATCH_FLOATING_TYPES macro dispatches on float and double types. With a double tensor,
# using fmaxf (which is for float) may produce wrong behavior or compile errors.
def test_double_input():
    batch_size = 2
    channels = 2
    height = 8
    width = 8
    kernel_size = 2  # supported kernel size branch instantiated with KERNEL_SIZE = 2
    stride = 2
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float64)
    my_module = build_kernel()
    # The following call should trigger misbehavior due to the use of fmaxf with double precision.
    output = my_module.forward(x, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    correct_output = torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    # The output calculated by the kernel is expected to be different as the kernel incorrectly uses fmaxf.
    assert not torch.allclose(output, correct_output, atol=1e-5), "Kernel improperly handled double precision input."

if __name__ == "__main__":
    pytest.main([__file__])
