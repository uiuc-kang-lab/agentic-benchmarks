
import os
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA kernel module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel does not check for input tensor type.
def test_input_tensor_wrong_dtype():
    # Create a tensor with dtype float64 instead of float32.
    x = torch.randn(1024, device='cuda', dtype=torch.float64)
    mod = build_kernel()
    # Run the CUDA kernel. Since the kernel uses data_ptr<float>(), the result 
    # will be computed from an incorrectly interpreted memory layout.
    result = mod.forward(x, 0.01)
    # Compute the expected result using PyTorch's native leaky_relu (which properly
    # handles float64) and compare. They should be different.
    result_ref = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    # We expect the outputs to differ.
    assert not torch.allclose(result.double(), result_ref), "Kernel incorrectly processed a non-float32 tensor."

# Issue 2: The kernel lacks error checking after launch.
def test_non_contiguous_tensor():
    # Create a non-contiguous tensor.
    x = torch.randn(1024, device='cuda', dtype=torch.float32).view(32, 32).t()  # .t() makes it non-contiguous.
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the CHECK_CONTIGUOUS macro mandates contiguous input.
        mod.forward(x, 0.01)

# Issue 3: The kernel comments promise __ldg usage, but it is not actually used.
def test_ldg_usage_in_kernel():
    # Read the kernel source code file to verify if __ldg intrinsic is used.
    kernel_file = "kernel.cu"
    assert os.path.exists(kernel_file), f"{kernel_file} does not exist in the current directory."
    with open(kernel_file, "r") as f:
        kernel_src = f.read()
    # Check for the presence of the __ldg( pattern.
    assert "__ldg(" in kernel_src, "Kernel does not use __ldg intrinsic as promised in its comments."

