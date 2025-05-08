
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="max_pool_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to compute MaxPool2d using PyTorch's own implementation.
def maxpool_torch(input, kernel_size, stride, padding, dilation):
    pooling = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation=dilation)
    return pooling(input)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fixed_unroll_issue():
    # Issue 1: Test the kernel with a kernel_size that is not 2 or 3.
    # Using kernel_size=5 will force the "else" branch with "#pragma unroll 4".
    device = "cuda"
    batch_size, channels, height, width = 2, 3, 16, 16
    input = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)
    kernel_size = 5  # Non-standard size triggers the unroll fixed factor
    stride = 1
    padding = 2
    dilation = 1
    mod = build_kernel()
    output_kernel = mod.forward(input, kernel_size, stride, padding, dilation)
    output_torch = maxpool_torch(input, kernel_size, stride, padding, dilation)
    # This assertion is expected to fail if the unroll assumption in the kernel causes errors.
    assert torch.allclose(output_kernel, output_torch, atol=1e-5), (
        "Mismatch in max pooling results for kernel_size != 2 or 3 (possible fixed unroll issue)"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unqualified_max_function():
    # Issue 2: The kernel’s usage of an unqualified max might yield subtle errors.
    # Here we compare results for a kernel_size value that uses the unqualified max branch.
    # Even though the arithmetic may seem correct, this test can expose if the wrong max()
    # is chosen by the compiler or if there is any precision/behavior mismatch.
    device = "cuda"
    batch_size, channels, height, width = 2, 3, 16, 16
    input = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)
    kernel_size = 3  # This branch uses unrolled loops with unqualified max()
    stride = 1
    padding = 1
    dilation = 1
    mod = build_kernel()
    output_kernel = mod.forward(input, kernel_size, stride, padding, dilation)
    output_torch = maxpool_torch(input, kernel_size, stride, padding, dilation)
    assert torch.allclose(output_kernel, output_torch, atol=1e-5), (
        "The CUDA kernel output does not match Torch's output; check the unqualified max() usage."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_type_initialization():
    # Issue 3: Feeding the kernel a non-floating point (e.g., int32) tensor that
    # does not support -infinity initialization should cause an error or incorrect behavior.
    device = "cuda"
    batch_size, channels, height, width = 2, 2, 8, 8
    # Create an integer type tensor.
    input = torch.randint(low=0, high=10, size=(batch_size, channels, height, width), device=device, dtype=torch.int32)
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting the kernel to fail (or produce a runtime error)
        _ = mod.forward(input, kernel_size, stride, padding, dilation)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_missing_error_checking_behavior():
    # Issue 4: Because the kernel does not check for CUDA errors, passing an invalid parameter
    # (such as a negative padding) might not immediately trigger an exception but will produce
    # an output that deviates from the correct behavior.
    device = "cuda"
    batch_size, channels, height, width = 2, 3, 16, 16
    input = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)
    kernel_size = 2
    stride = 2
    padding = -1   # Invalid negative padding
    dilation = 1
    mod = build_kernel()
    output_kernel = mod.forward(input, kernel_size, stride, padding, dilation)
    output_torch = maxpool_torch(input, kernel_size, stride, padding, dilation)
    # Since the padding is invalid, the kernel output is not expected to match the torch version.
    assert not torch.allclose(output_kernel, output_torch, atol=1e-5), (
        "Expected mismatch due to invalid padding; the kernel did not check and handle CUDA errors."
    )
