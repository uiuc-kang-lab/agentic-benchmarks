
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Create a contiguous tensor and then make it non-contiguous by transposing
    batch_size, channels, height, width = 2, 3, 8, 8
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda")
    non_contiguous_input = input_tensor.transpose(1, 2)  # Now shape becomes (batch, height, channels, width)
    # For pooling, we need a tensor of shape (batch, channels, height, width).
    # We force a non-contiguous tensor by transposing twice.
    non_contiguous_input = non_contiguous_input.transpose(1, 2)

    # Define the pooling parameters (with non-trivial dilation and padding)
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1

    # Expected output using built-in max_pool2d (which works with non-contiguous tensors)
    expected = torch.nn.functional.max_pool2d(non_contiguous_input, kernel_size=kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
    module = build_kernel()
    # Our kernel implementation assumes contiguous mem, so we expect it to return a different result when input is non-contiguous.
    output = module.forward(non_contiguous_input, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # This assert is expected to fail if the kernel did not handle non-contiguous memory correctly.
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Test failed to detect non-contiguous memory issue: kernel output matched expected output despite non-contiguous input."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_half_precision_input():
    batch_size, channels, height, width = 2, 3, 8, 8
    half_input = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.half)
    # Use parameters that are valid.
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1

    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Since AT_DISPATCH_FLOATING_TYPES does not cover half,
        # invoking the kernel with a half type tensor should lead to an error.
        output = module.forward(half_input, kernel_size, stride, padding, dilation)
        # Use synchronize to force error propagation.
        torch.cuda.synchronize()
