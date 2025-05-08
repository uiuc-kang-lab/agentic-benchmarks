
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Compile and load the CUDA extension from kernel.cu.
# Ensure the current working directory contains kernel.cu.
def build_kernel():
    cuda_module = load(
        name="l1_norm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger unsupported data type (only float32 is supported)
def test_non_float32_input():
    my_module = build_kernel()
    # Create a double (float64) tensor on CUDA
    x = torch.randn(16, 16384, dtype=torch.double, device='cuda')
    with pytest.raises(RuntimeError) as excinfo:
        _ = my_module.forward(x)
    assert "Input tensor must be on CUDA" not in str(excinfo.value), "Unexpected CUDA check failure."
    # Expect the kernel to malfunction / raise an error due to type mismatch.
    # If the kernel does not check type, the CUDA code will produce undefined behavior.
    # Therefore, we expect an error (or a mis-computation) when using non-float32 types.

# Test case 2: Trigger invalid input dimensions (kernel expects 2D tensor)
def test_invalid_dimension_input():
    my_module = build_kernel()
    # Create a 3D tensor on CUDA
    x = torch.randn(4, 16, 16384, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError) as excinfo:
        _ = my_module.forward(x)
    assert "Expected 2D tensor" in str(excinfo.value), "Kernel should enforce 2D tensor input."

# Test case 3: Kernel launch error/synchronization issue
def test_kernel_launch_error():
    my_module = build_kernel()
    # This input is valid in shape and dtype.
    # To try to force a kernel launch error, we provide a tensor with zero columns,
    # which may lead to a reduction over an empty range.
    x = torch.randn(16, 0, device="cuda", dtype=torch.float32)
    # Even when D == 0, the kernel sets total = 1e-12 and iterates 0 times over cols.
    # It is debatable if this is a kernel launch error.
    # Here, we check that the operation completes without synchronization issues.
    # In a production scenario, lack of post-kernel error checking might hide errors.
    out = my_module.forward(x)
    # Synchronize to catch any asynchronous CUDA errors.
    torch.cuda.synchronize()
    # The output should be an empty tensor with the same shape.
    assert out.shape == x.shape, "Output shape must match input shape."
