
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Ensure that we compile in a temporary directory so that multiple builds don't conflict.
BUILD_DIR = "./build_temp"
if not os.path.exists(BUILD_DIR):
    os.makedirs(BUILD_DIR)

def build_kernel():
    cuda_module = load(
        name="custom_relu",
        sources=["kernel.cu"],
        extra_include_paths=[],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=BUILD_DIR,
        verbose=True,
    )
    return cuda_module

# Test case 1:
# Trigger issue with tensor of type double.
# The kernel always casts data pointers to float*.
def test_invalid_dtype():
    my_module = build_kernel()
    # Create a double precision tensor on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # Compute reference using torch.relu
    ref = torch.relu(x)
    # Call our custom kernel
    y = my_module.forward(x)
    # Since the kernel treats the data as float (4 bytes) while the tensor is double (8 bytes),
    # the output will be incorrect.
    # We expect that the output does not match the correct reference.
    assert not torch.allclose(y, ref, atol=1e-5), "Kernel unexpectedly produced correct results for double tensors!"

# Test case 2:
# Trigger issue with non-contiguous input tensor.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor on CUDA.
    x_full = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    # Produce a non-contiguous slice.
    x = x_full.t()  # transpose makes the tensor non-contiguous
    ref = torch.relu(x)
    y = my_module.forward(x)
    # The kernel uses data_ptr assuming contiguous memory so the output may be corrupted.
    assert not torch.allclose(y, ref, atol=1e-5), "Kernel unexpectedly produced correct results on non-contiguous input!"

# Test case 3:
# Trigger issue with a CPU tensor.
def test_cpu_tensor_input():
    my_module = build_kernel()
    # Create a CPU tensor.
    x = torch.randn(1024, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expect a runtime error because the kernel is launched on GPU but input is on CPU.
        _ = my_module.forward(x)
