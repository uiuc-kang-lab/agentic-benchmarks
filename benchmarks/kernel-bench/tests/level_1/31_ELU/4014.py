
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="elu_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def ref_elu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Use torch's built-in ELU for reference
    return F.elu(x, alpha=alpha)

def test_non_float32_dtype(kernel_module):
    # Issue 1: The kernel assumes float32 dtype.
    # Create a double tensor and pass it to the kernel.
    N = 1024
    x = torch.randn(N, device="cuda", dtype=torch.double)
    # The kernel does not check for dtype, so it will reinterpret the data pointer.
    # The result will be numerically incorrect compared to torch.nn.functional.elu.
    alpha = 1.0
    out = kernel_module.forward(x, alpha)
    expected = ref_elu(x.to(torch.float32), alpha)
    # Convert kernel output to float32 for fair comparison if necessary.
    out_float = out.to(torch.float32)
    # Assert that the output is not close to the expected result.
    # If they were accidentally close then the issue would be hidden.
    assert not torch.allclose(out_float, expected, atol=1e-5), (
        "Kernel did not trigger an error when using non-float32 tensor. "
        "The output is suspiciously close to the expected one."
    )

def test_misaligned_memory(kernel_module):
    # Issue 2: The kernel uses vectorized loads (float4) and assumes memory alignment.
    # We purposely create a misaligned tensor by slicing off one element.
    # This slicing is likely to produce a tensor whose underlying pointer is not 16-byte aligned.
    N = 1024 + 1  # ensure at least one element will be sliced out for misalignment.
    x_full = torch.randn(N, device="cuda", dtype=torch.float32)
    # Create a misaligned view by skipping the first element.
    x = x_full[1:]  # this view may not be 16-byte aligned.
    alpha = 1.0
    out = kernel_module.forward(x, alpha)
    expected = ref_elu(x, alpha)
    # It is expected that if misalignment affects the vectorized load,
    # then the result from the kernel will differ significantly.
    max_diff = (out - expected).abs().max().item()
    assert max_diff > 1e-3, (
        f"Kernel output for a misaligned tensor is too close to expected results (max diff {max_diff}). "
        "This may indicate that the kernel is not correctly handling memory alignment issues."
    )

def test_kernel_error_checking(kernel_module):
    # Issue 3: The kernel does not perform any error checking after launching.
    # While we cannot directly force a launch error from Python, we can simulate a scenario
    # that may trigger asynchronous errors by passing an empty tensor.
    x = torch.empty(0, device="cuda", dtype=torch.float32)
    alpha = 1.0
    # Launch the kernel on an empty tensor which should complete gracefully.
    out = kernel_module.forward(x, alpha)
    # Synchronize to force any asynchronous error to appear.
    torch.cuda.synchronize()
    # Verify that the output is also an empty tensor.
    assert out.numel() == 0, (
        "Kernel did not return an empty tensor when given an empty input. "
        "This may indicate issues with error handling in the kernel launch."
    )
