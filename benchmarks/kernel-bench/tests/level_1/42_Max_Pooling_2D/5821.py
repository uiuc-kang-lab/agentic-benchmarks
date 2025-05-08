
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="max_pool2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 is detected at compile time; to trigger a runtime error similar to it,
# we assume that if the unqualified usage of max were problematic, then compiling for an unsupported
# type (e.g. integer) would trigger an error.

def test_integer_tensor_input():
    # Issue 2: Kernel limited to floating types.
    # Create an integer tensor and expect the dispatch to fail.
    cuda_module = build_kernel()
    x_int = torch.randint(0, 10, (2, 3, 8, 8), device='cuda', dtype=torch.int32)
    with pytest.raises(RuntimeError):
        # Should throw because integer is not handled by AT_DISPATCH_FLOATING_TYPES
        cuda_module.forward(x_int, 2, 2, 0, 1)

def test_half_tensor_input():
    # Issue 2: Kernel limited to floating types.
    # Create a half tensor. AT_DISPATCH_FLOATING_TYPES does not dispatch for half.
    cuda_module = build_kernel()
    x_half = torch.randn(2, 3, 8, 8, device='cuda', dtype=torch.half)
    with pytest.raises(RuntimeError):
        cuda_module.forward(x_half, 2, 2, 0, 1)

def test_noncontiguous_input():
    # Issue 3: Kernel assumes contiguous input.
    # Create a contiguous tensor, then make it noncontiguous by transposing
    cuda_module = build_kernel()
    x = torch.randn(2, 3, 8, 8, device='cuda', dtype=torch.float32)
    x_noncontiguous = x.transpose(1, 2)  # This will make the tensor noncontiguous
    # To clearly show the issue we compare against PyTorch's nn.MaxPool2d
    pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
    # If the custom kernel does pointer arithmetic without handling strides,
    # the output would differ from the reference.
    out_ref = pool(x_noncontiguous)
    out_cuda = cuda_module.forward(x_noncontiguous.contiguous(), 2, 2, 0, 1)
    # The test intentionally fails if noncontiguous input is silently accepted,
    # so here we trigger the behavior:
    if not x_noncontiguous.is_contiguous():
        with pytest.raises(AssertionError):
            assert torch.allclose(out_cuda, out_ref, atol=1e-5)

def test_empty_pooling_window():
    # Issue 4: When the pooling window has no valid elements, the output remains -infinity.
    # This test chooses parameters so that for a given output location, no valid input index is covered.
    cuda_module = build_kernel()
    # Create an input tensor with one spatial element.
    x = torch.tensor([[[[5.0]]]], device='cuda', dtype=torch.float32)
    # Set parameters such that the pooling window is shifted completely outside.
    # For example: kernel_size=3, stride=1, padding=0, dilation=2 makes the intended receptive field span indices [0,4],
    # but the input only has index 0. Hence for many output locations, the precomputed valid range will be empty.
    out_cuda = cuda_module.forward(x, 3, 1, 0, 2)
    # Since no valid input element is covered for the pooling window, the output should be -infinity.
    assert torch.isinf(out_cuda).all() and (out_cuda < 0).all(), "Expected output to be -infinity for empty pooling window."

# Note: Issue 5 (lack of kernel launch error checking) is more of a debug-time/cuda runtime issue and
# is not directly invoked via a pytest test case. However, if a kernel launch error occurs due to bad launch parameters,
# the following test may trigger a CUDA error.

def test_kernel_launch_error():
    # Issue 5: Kernel launch error checking absent.
    # We pass deliberately invalid kernel parameters (e.g. negative output dimensions).
    cuda_module = build_kernel()
    x = torch.randn(2, 3, 8, 8, device='cuda', dtype=torch.float32)
    # Negative padding (invalid value) may cause miscomputed dimensions and trigger a CUDA error.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 2, 2, -1, 1)
