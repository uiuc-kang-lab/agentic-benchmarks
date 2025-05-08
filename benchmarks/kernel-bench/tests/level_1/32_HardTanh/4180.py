
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    cuda_module = load(
        name="hardtanh_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger device-side min/max issue
# If the wrong min/max is used, the kernel might either not compile/device crash
# Since the error would occur during kernel compilation or launch, we attempt to run it.
# For this test we use a regular contiguous float tensor.
def test_min_max_issue():
    kernel_module = build_kernel()
    # Use a moderate size tensor (single-stream mode)
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    out = kernel_module.forward(x, -1.0, 1.0)
    # Compare with PyTorch’s hardtanh
    expected = F.hardtanh(x, min_val=-1.0, max_val=1.0)
    # In a correct implementation these should match.
    # If the device min/max functions are not properly called, the output may be off.
    # We use an equality check and report a failure if they match unexpectedly.
    if torch.allclose(out, expected, atol=1e-5):
        pytest.fail("Kernel min/max operations seemed correct; issue not triggered.")
    else:
        # If the outputs differ, we assume the improper device min/max is causing the discrepancy.
        assert True

# Test 2: Trigger unsupported floating point type (e.g. half precision)
def test_half_precision_not_supported():
    kernel_module = build_kernel()
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # We expect that dispatching on float16 would not find a valid case,
        # causing a runtime error.
        kernel_module.forward(x, -1.0, 1.0)

# Test 3: Lack of proper error checking in CUDA API calls
# One common error is passing a non-CUDA tensor. The kernel is supposed to throw an exception.
def test_non_cuda_input_error():
    kernel_module = build_kernel()
    x = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(Exception):
        kernel_module.forward(x, -1.0, 1.0)

# Test 4: Input tensor noncontiguity issues.
# The kernel uses data_ptr() and assumes contiguous memory.
def test_non_contiguous_tensor():
    kernel_module = build_kernel()
    # Create a 2D tensor then transpose it to make it noncontiguous.
    x = torch.randn(128, 128, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # noncontiguous view
    out = kernel_module.forward(x_noncontig, -1.0, 1.0)
    # The expected result using PyTorch’s hardtanh on the noncontiguous tensor.
    expected = F.hardtanh(x_noncontig, min_val=-1.0, max_val=1.0)
    # If the kernel incorrectly interprets noncontiguous memory, the result will differ.
    if torch.allclose(out, expected, atol=1e-5):
        pytest.fail("Kernel produced correct output with a noncontiguous tensor; issue not triggered.")
    else:
        # The difference indicates the kernel’s assumption of contiguity is problematic.
        assert True
