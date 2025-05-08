
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 test:
# When using std::numeric_limits in device code, some devices (or types) may be mishandled.
# We trigger a possible error by using double-precision input.
# If the kernel does not correctly initialize max_val for double, the result will differ from PyTorch.
def test_incorrect_infinity_initialization_double():
    cuda_module = build_kernel()
    # Use a double tensor
    batch_size = 2
    channels = 3
    height = 10
    width = 10
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1

    # Create a tensor with values that are all negative so that the starting value (-infinity) must be replaced.
    x = -torch.abs(torch.randn(batch_size, channels, height, width, dtype=torch.float64, device="cuda"))
    # Compute expected result using PyTorch’s own max_pool2d.
    expected = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    out = cuda_module.forward(x, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # If the device-code infinity handling is not working properly then the result will be far off.
    assert torch.allclose(out, expected, atol=1e-5), \
        f"Issue 1 triggered: double precision max pooling output incorrect.\nExpected: {expected}\nGot: {out}"

# Issue 2 test:
# The use of an unqualified max() call in device code may lead to surprises.
# Here we design a simple test where the correct value is known.
def test_unqualified_max_function():
    cuda_module = build_kernel()
    batch_size = 1
    channels = 1
    height = 4
    width = 4
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1

    # Create an input tensor with a distinct pattern.
    # In every 2x2 pooling window the maximum is the bottom-right element.
    # For example, use a tensor that increases from left-to-right, top-to-bottom.
    x = torch.tensor([[[[ 1,  2,  3,  4],
                        [ 5,  6,  7,  8],
                        [ 9, 10, 11, 12],
                        [13, 14, 15, 16]]]], dtype=torch.float32, device="cuda")
    # Expected output for a 2x2 window, stride 1:
    expected = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    out = cuda_module.forward(x, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # If the max() operation is miscompiled then the computed maximum values can be wrong.
    assert torch.allclose(out, expected), \
        f"Issue 2 triggered: unqualified max() resulted in incorrect values.\nExpected: {expected}\nGot: {out}"

# Issue 3 test:
# A very large output is needed such that the computed number of blocks becomes capped via the min() call.
# If the unqualified min() causes problems then the kernel may not cover all output elements.
def test_min_usage_in_block_calculation():
    cuda_module = build_kernel()
    batch_size = 1
    channels = 1
    # Set up input dimensions so that the output number of elements is huge.
    height = 1024
    width = 1024
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    expected = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out = cuda_module.forward(x, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # Even though a cap on the number of blocks is intended, the kernel should process all output elements correctly.
    assert torch.allclose(out, expected, atol=1e-4), \
        f"Issue 3 triggered: output mismatch when processing a large tensor (block count calculation problem).\nExpected: {expected.mean()}\nGot: {out.mean()}"

# Issue 4 test:
# The kernel assumes that the input (and the output buffer) are contiguous.
# Passing a non-contiguous input (for example, via a transpose) will cause incorrect index computations.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size = 2
    channels = 3
    height = 16
    width = 16
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    # Create a contiguous tensor then transpose one of the spatial dims to get a non-contiguous tensor.
    x_contig = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    x_noncontig = x_contig.transpose(2, 3)  # now non-contiguous

    # We compare the kernel result using non-contiguous input to the expected result computed on a contiguous copy.
    expected = F.max_pool2d(x_noncontig.contiguous(), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out = cuda_module.forward(x_noncontig, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # Since the kernel does not check for contiguity, its output will be wrong in this case.
    assert not torch.allclose(out, expected, atol=1e-5), \
        "Issue 4 triggered: non-contiguous input was handled correctly whereas it should have produced an error (or wrong result) due to lack of contiguity check."

if __name__ == '__main__':
    pytest.main([__file__])
