
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Ensure that the kernel file "kernel.cu" exists in the current directory.
# The load call will compile and load the CUDA extension from kernel.cu.
def build_kernel():
    # Set verbose=True to see compilation messages if needed.
    module = load(
        name="batchnorm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# ------------------------------
# Issue 1: Kernel only supports float32.
# This test passes input tensors of a different type (e.g. float64)
# to trigger the issue.
def test_float32_only():
    cuda_mod = build_kernel()

    # Create input tensors in float64 instead of float32.
    x = torch.randn(16, 64, 256, 256, dtype=torch.float64, device="cuda")
    # weight, bias, running_mean, running_var expected to be float32.
    weight = torch.randn(64, dtype=torch.float64, device="cuda")
    bias = torch.randn(64, dtype=torch.float64, device="cuda")
    running_mean = torch.zeros(64, dtype=torch.float64, device="cuda")
    running_var = torch.ones(64, dtype=torch.float64, device="cuda")
    
    with pytest.raises(Exception) as excinfo:
        # This should trigger either an invalid memory access or type mismatch.
        cuda_mod.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
    assert "CUDA" in str(excinfo.value) or "float" in str(excinfo.value)

# ------------------------------
# Issue 2: Kernel only supports 4D inputs.
# This test supplies a non-4D input tensor (e.g. 3D) to trigger an index error.
def test_non_four_d_input():
    cuda_mod = build_kernel()

    # Create a 3D tensor to simulate an unsupported input shape.
    x = torch.randn(16, 256, 256, dtype=torch.float32, device="cuda")
    weight = torch.randn(256, dtype=torch.float32, device="cuda")
    bias = torch.randn(256, dtype=torch.float32, device="cuda")
    running_mean = torch.zeros(256, dtype=torch.float32, device="cuda")
    running_var = torch.ones(256, dtype=torch.float32, device="cuda")
    
    with pytest.raises(IndexError):
        cuda_mod.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)

# ------------------------------
# Issue 3: Kernel requires contiguous inputs.
# This test provides a non-contiguous tensor to trigger the built-in check.
def test_non_contiguous_input():
    cuda_mod = build_kernel()

    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(16, 64, 256, 256, dtype=torch.float32, device="cuda").transpose(1, 2)
    weight = torch.randn(64, dtype=torch.float32, device="cuda")
    bias = torch.randn(64, dtype=torch.float32, device="cuda")
    running_mean = torch.zeros(64, dtype=torch.float32, device="cuda")
    running_var = torch.ones(64, dtype=torch.float32, device="cuda")
    
    with pytest.raises(Exception) as excinfo:
        cuda_mod.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
    assert "contiguous" in str(excinfo.value)

# ------------------------------
# Issue 4: Hard-coded kernel launch configuration.
# This test uses an unusual input shape with a very small number of elements per channel
# to highlight potential inefficiencies or correctness issues due to the fixed launch config.
def test_unusual_input_shape():
    cuda_mod = build_kernel()

    # Create an input tensor with very small spatial dimensions.
    # For example, H and W are 1, which might not provide enough work per block 
    # when the kernel expects many rows to be reduced.
    x = torch.randn(32, 128, 1, 1, dtype=torch.float32, device="cuda")
    weight = torch.randn(128, dtype=torch.float32, device="cuda")
    bias = torch.randn(128, dtype=torch.float32, device="cuda")
    running_mean = torch.zeros(128, dtype=torch.float32, device="cuda")
    running_var = torch.ones(128, dtype=torch.float32, device="cuda")
    
    # In this case, the kernel launch grid is <<<C,256>>>, so each block will have many threads
    # while there is little work in each channel. The test checks for out-of-bound or NaNs.
    output = cuda_mod.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
    torch.cuda.synchronize()
    # Verify output shape matches input shape.
    assert output.shape == x.shape, "Output shape mismatch for unusual input shape."
    # Additionally, check that the output does not contain NaNs (which might occur if reduction is incorrect).
    assert not torch.isnan(output).any(), "Output contains NaNs for unusual input shape."
