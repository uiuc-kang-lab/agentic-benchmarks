
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# A helper function to build the CUDA extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_gelu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Multiple kernel launches lead to redundant computation.
# Test that the output is still numerically correct even though multiple kernels run.
def test_multiple_kernel_launches():
    mod = build_kernel()
    # Create an aligned input tensor (size a multiple of 4 for float; alignment is likely)
    N = 16384 + 16  # add a few extra elements
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    # Call the extension forward function
    y = mod.forward(x)
    # Reference GELU computation using PyTorch’s functional GELU
    y_ref = F.gelu(x)
    # Check that the result is correct despite repeated kernel launches.
    assert torch.allclose(y, y_ref, atol=1e-5), "Output incorrect when multiple kernel launches occur."

# Issue 2: The tiled scalar kernel is defined but never used.
# Test to ensure that gelu_kernel_tiled_scalar is not exported.
def test_tiled_kernel_not_exported():
    mod = build_kernel()
    # Try to access a symbol for the tiled kernel from the module.
    # The symbol should not be found since it is not used or exported.
    assert not hasattr(mod, "gelu_kernel_tiled_scalar"), "Tiled kernel should not be exported."

# Issue 3: Lack of explicit synchronization may hide asynchronous errors.
# Test that using a non-CUDA tensor as input throws an error.
def test_non_cuda_input():
    mod = build_kernel()
    x_cpu = torch.randn(100, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        _ = mod.forward(x_cpu)

# Additional test to force a remainder (non-multiple of the vector factor) branch.
def test_remainder_handling():
    mod = build_kernel()
    # For float, vec_factor == 4.
    # Create a tensor whose number of elements is not divisible by 4.
    N = 1023  # 1023 % 4 = 3 (remainder != 0)
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = mod.forward(x)
    y_ref = F.gelu(x)
    assert torch.allclose(y, y_ref, atol=1e-5), "Output incorrect for tensors with remainder elements."

if __name__ == "__main__":
    pytest.main([__file__])
