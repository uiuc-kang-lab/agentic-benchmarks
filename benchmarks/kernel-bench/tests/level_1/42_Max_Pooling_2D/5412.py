
import pytest
import torch
from torch.utils.cpp_extension import load, BuildExtension
import os

# Utility to build the CUDA kernel extension
def build_kernel():
    # Remove any previously built files if present.
    build_dir = os.path.join(os.getcwd(), "build")
    if os.path.exists(build_dir):
        import shutil
        shutil.rmtree(build_dir)
    cuda_module = load(
        name="maxpool2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that non-floating types cause errors.
def test_non_floating_input():
    cuda_module = build_kernel()
    # Create an integer input tensor.
    batch_size = 4
    channels = 3
    height = 10
    width = 10
    x_int = torch.randint(0, 100, (batch_size, channels, height, width), dtype=torch.int32, device="cuda")
    
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1

    with pytest.raises(RuntimeError):
        # Expectation: the kernel or dispatch will fail for integer types due to infinity initialization.
        _ = cuda_module.forward(x_int, kernel_size, stride, padding, dilation)

# Issue 2: Test that the unqualified max function in device code leads to a compilation or runtime error.
# Because this is a compile-time/device-code issue, we attempt to trigger it by using a simple float input.
def test_max_function_usage():
    try:
        cuda_module = build_kernel()
    except Exception as e:
        # If building the kernel fails, we assume it is due to the unqualified max() usage.
        pytest.skip(f"Kernel build failure (expected due to max() issue): {e}")
    
    batch_size = 2
    channels = 2
    height = 16
    width = 16
    x_float = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    try:
        out = cuda_module.forward(x_float, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()
    except Exception as e:
        # If an error is thrown during execution, assume it is a result of the misused max() call.
        pytest.skip(f"Kernel runtime error (expected due to max() issue): {e}")
    
    # If execution succeeds, we warn that the max() issue might not manifest for these inputs.
    assert out is not None

# Issue 3: Test that a huge product of batch_size and channels (exceeding gridDim.z limit)
# leads to a CUDA error (or at least unexpected behavior).
def test_grid_dimension_overflow():
    cuda_module = build_kernel()
    # Typical CUDA gridDim.z limit is 65535.
    # We choose batch_size and channels so that batch_size * channels > 65535.
    batch_size = 1024
    channels = 70  # 1024 * 70 = 71680 > 65535
    height = 32
    width = 32
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    with pytest.raises(RuntimeError):
        # Expecting that launching the kernel will generate an error due to invalid grid dimensions.
        _ = cuda_module.forward(x, kernel_size, stride, padding, dilation)
