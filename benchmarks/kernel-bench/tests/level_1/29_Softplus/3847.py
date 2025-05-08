
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure that the path is correct relative to this test file.
    cuda_module = load(
        name="softplus_kernel_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2: Shared memory allocation is specified even though it is never used.
# This is subtle since the kernel will still compute correct softplus.
# We simulate a scenario where the waste might be exposed via performance or unexpected behavior.
def test_unused_shared_memory():
    # Note: The kernel does not use shared memory, but we can at least ensure that the result is correct.
    # However, the test does not invoke an error, it only documents the mis-design.
    cuda_module = build_kernel()
    x = torch.randn(1024, device='cuda', dtype=torch.float32)
    out = cuda_module.forward(x)
    expected = F.softplus(x)
    # The result should be correct even though shared memory is allocated but not used.
    assert torch.allclose(out, expected, atol=1e-5), "Output does not match softplus reference. (Shared memory misuse issue)"

# Issue 3: Use of fixed double-precision constants in compute_softplus.
def test_threshold_constants_conversion():
    cuda_module = build_kernel()
    # Create a tensor containing values around the threshold boundaries.
    # Values >20 should return x, values < -20 should return exp(x),
    # and values in between should return log1p(exp(x)).
    values = torch.tensor([-30.0, -20.5, -20.0, 0.0, 20.0, 20.5, 30.0], device='cuda', dtype=torch.float32)
    out = cuda_module.forward(values)
    expected = F.softplus(values)
    # Even if the kernel works correctly, the use of double literals might affect performance/precision.
    # Here we only test for numerical closeness.
    assert torch.allclose(out, expected, atol=1e-5), "Threshold handling in compute_softplus may be affected by literal constant conversion."

# Issue 4: No error checking after kernel launch.
def test_kernel_launch_error_checking():
    cuda_module = build_kernel()
    # We simulate an error scenario by passing an input tensor with an invalid shape
    # for which the kernel’s 1D iteration might not correctly cover the multi-dimensional layout
    # if the tensor is not contiguous – this can result in incorrect behavior that goes undetected.
    x = torch.randn(16, 16384, device='cuda', dtype=torch.float32)
    # Make x non-contiguous deliberately.
    x_noncontiguous = x.t()
    out = cuda_module.forward(x_noncontiguous)
    expected = F.softplus(x_noncontiguous)
    # With non-contiguous input the arithmetic based on numel() is unsafe.
    # We expect the outputs to differ.
    if torch.allclose(out, expected, atol=1e-5):
        pytest.skip("Non-contiguous input unexpectedly produced correct result; kernel may be inadvertently handling it.")
    else:
        # The error is that the kernel does not check for non-contiguity so wrong results are produced.
        assert not torch.allclose(out, expected, atol=1e-5), "Kernel did not raise an error on non-contiguous input (error checking issue)."

# Issue 5: Kernel assumes contiguous input.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    x = torch.randn(1024, 128, device='cuda', dtype=torch.float32)
    # Make a non-contiguous tensor by transposing
    x_noncontiguous = x.transpose(0, 1)
    out = cuda_module.forward(x_noncontiguous)
    expected = F.softplus(x_noncontiguous)
    # Since the kernel simply iterates over numel() without stride conversion,
    # the result will be different.
    assert not torch.allclose(out, expected, atol=1e-5), "Kernel incorrectly assumed contiguous input."

# Issue 6: AT_DISPATCH_FLOATING_TYPES prevents usage with half precision.
def test_half_precision_input():
    cuda_module = build_kernel()
    x_half = torch.randn(1024, device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # Expect the dispatch to fail since half precision is not included in AT_DISPATCH_FLOATING_TYPES.
        _ = cuda_module.forward(x_half)
