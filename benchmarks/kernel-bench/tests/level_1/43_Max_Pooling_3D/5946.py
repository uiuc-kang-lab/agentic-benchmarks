
import math
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    # Load the CUDA kernel from kernel.cu
    cuda_module = load(
        name="custom_maxpool3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Non-uniform parameters are not supported.
def test_non_uniform_parameters():
    cuda_module = build_kernel()
    # The custom kernel interface only supports int parameters.
    # Attempt to pass a tuple (mimicking non-uniform parameters) should trigger an error.
    input_tensor = torch.randn(2, 3, 10, 10, 10, device="cuda", dtype=torch.float32)
    # We simulate the ill‐configuration by trying to call forward with a tuple.
    with pytest.raises(TypeError):
        # This call is intended to mimic an attempt at non uniform parameters.
        # (the extension expects ints; passing a tuple should raise a TypeError)
        cuda_module.forward(input_tensor, (3, 3, 3), 2, 1, 1, False, False)

# Issue 2: Use with non-floating point types.
def test_non_floating_point_input():
    cuda_module = build_kernel()
    # Create an integer tensor on CUDA.
    int_input = torch.randint(0, 100, (2, 3, 8, 8, 8), device="cuda", dtype=torch.int32)
    # The kernel is dispatched only for floating types so an int tensor should trigger an error.
    with pytest.raises(RuntimeError):
        cuda_module.forward(int_input, 3, 2, 1, 1, False, False)

# Issue 3: Lack of explicit CUDA error checking.
def test_missing_cuda_error_sync():
    cuda_module = build_kernel()
    # Create an input tensor with an anomalous shape that might cause an invalid memory access.
    # For instance, a too-small tensor that (due to ceil_mode) requests an output element whose pooling
    # window lies completely outside the real input.
    # This may not throw an error immediately, but we force a synchronization to check for asynchronous errors.
    input_tensor = torch.randn(1, 1, 2, 2, 2, device="cuda", dtype=torch.float32)
    # Using ceil_mode True to compute an output size that is larger than a valid pooling.
    output = cuda_module.forward(input_tensor, 3, 1, 0, 1, False, True)
    # Force synchronization to trigger any latent error.
    with pytest.raises(RuntimeError):
        torch.cuda.synchronize()

# Issue 4: Using #pragma unroll with a non constant kernel_size.
def test_unroll_with_dynamic_kernel_size():
    cuda_module = build_kernel()
    # Use a kernel_size that is larger than typical so that the runtime value might
    # not be optimized by the unroll pragma as expected.
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1
    # Create a random input tensor.
    input_tensor = torch.randn(2, 3, 15, 15, 15, device="cuda", dtype=torch.float32)
    # Compute the output with the custom kernel.
    custom_out = cuda_module.forward(input_tensor, kernel_size, stride, padding, dilation, False, False)
    # Compute the output with PyTorch's native max_pool3d.
    torch_out = F.max_pool3d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=False)
    # They should be similar. If not, it might indicate that unrolling (or lack thereof) is affecting correctness.
    assert torch.allclose(custom_out, torch_out, atol=1e-4), f"Custom output differs from reference. Max diff: {(custom_out - torch_out).abs().max().item()}"

# Issue 5: Improper handling of ceil_mode may produce -infinity where padding is insufficient.
def test_ceil_mode_border_handling():
    cuda_module = build_kernel()
    # Create an input tensor whose dimensions plus padding barely allow a single pooling window with ceil_mode=True.
    input_tensor = torch.randn(1, 1, 4, 4, 4, device="cuda", dtype=torch.float32)
    # Choose parameters so that the computed output size with ceil_mode leads to a pooling window partially out-of-bound.
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    custom_out = cuda_module.forward(input_tensor, kernel_size, stride, padding, dilation, False, True)
    torch_out = F.max_pool3d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=True)
    # In the problematic case, the kernel would leave -infinity in locations where there were no valid elements.
    # We expect a mismatch with PyTorch's native result.
    # Test will fail if the custom kernel returns -infinity or otherwise deviates significantly.
    assert not torch.allclose(custom_out, torch_out, atol=1e-4), "Custom kernel output unexpectedly matches PyTorch output despite known ceil_mode issue."

