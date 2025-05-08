
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Test that non-float32 tensors are not handled correctly.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create a double precision input tensor.
    x = torch.randn(16, 64, 256, 256, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # Attempting to run forward on non-float32 should cause an error (or unintended behavior).
        _ = my_module.forward(x)

# Issue 2: Test that an empty input tensor (numel == 0) is not properly handled.
def test_empty_tensor():
    my_module = build_kernel()
    x = torch.empty(0, device="cuda", dtype=torch.float32)
    # Depending on the runtime behavior, the kernel launch may fail or produce an error.
    # Here we expect the operation to raise an error.
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x)

# Issue 3: Test that the lack of proper device synchronization/error checking manifests as incorrect results.
def test_missing_synchronization():
    my_module = build_kernel()
    # Create a normal tensor.
    x = torch.randn(16, 64, 256, 256, device="cuda", dtype=torch.float32)
    # Run the forward kernel multiple times to increase the chance to expose race conditions.
    # If the kernel is not synchronized properly, the normalization factor (Frobenius norm) might be computed incorrectly.
    out = my_module.forward(x)
    torch.cuda.synchronize()
    
    # Compute reference result using PyTorch
    norm = torch.norm(x, p='fro')
    expected = x / norm
    # We expect the output to match the reference within a tolerance.
    # If there is a race condition, the result may diverge.
    assert torch.allclose(out, expected, atol=1e-5), "Output does not match expected normalization. Possible synchronization issue."

# Issue 4: Test that division by zero in the normalization kernel is not properly handled.
def test_division_by_zero():
    my_module = build_kernel()
    # Create an input tensor that is all zeros, which causes the norm to be zero.
    x = torch.zeros(16, 64, 256, 256, device="cuda", dtype=torch.float32)
    out = my_module.forward(x)
    torch.cuda.synchronize()
    
    # Division by zero should produce NaNs or infinities.
    # Verify that the output contains either NaN or Inf values.
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    assert has_nan or has_inf, "Division by zero did not produce NaN or Inf in the output, even though the input is all zeros."
