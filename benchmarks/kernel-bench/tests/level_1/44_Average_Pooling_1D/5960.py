
import pytest
import torch
from torch.utils.cpp_extension import load
import threading
import time
import cuda  # dummy import to ensure CUDA is available

def build_kernel():
    cuda_module = load(
        name="pooling_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that using unsupported data types (e.g. float64) triggers an error.
def test_dtype_support():
    cuda_module = build_kernel()
    # create a double tensor which is not supported by our kernel expecting float32
    x = torch.randn(16, 32, 128, device="cuda", dtype=torch.double)
    kernel_size = 4
    stride = 2
    padding = 1
    with pytest.raises(RuntimeError):
        # Expect that our kernel (or host code) will error out when trying to get a double pointer casted to float*
        cuda_module.forward(x, kernel_size, stride, padding)

# Issue 2: Test that kernel launch errors are not caught.
# We force a grid configuration that exceeds typical hardware limits.
def test_grid_dimension_limit():
    cuda_module = build_kernel()
    # Most devices support grid dimensions up to 65535 for y and z.
    # We use an in_channels value that is likely to exceed this limit.
    huge_in_channels = 70000
    batch_size = 1
    input_length = 128
    x = torch.randn(batch_size, huge_in_channels, input_length, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 1
    padding = 1
    # Launching the kernel with huge_in_channels might lead to an invalid grid configuration 
    # and result in a CUDA error on kernel launch.
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, kernel_size, stride, padding)
        # Force synchronization to ensure the error is caught.
        torch.cuda.synchronize()

# Issue 3: Test concurrent kernel launches (using different parameters in different streams) 
# might interfere with one another because of constant memory usage.
def test_concurrent_kernels():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    input_length = 32

    # Create an input tensor.
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)

    # Define two different parameter sets.
    params1 = (3, 1, 1)  # (kernel_size, stride, padding)
    params2 = (5, 2, 2)

    result = {}

    def launch_kernel(name, params, stream):
        with torch.cuda.stream(stream):
            out = cuda_module.forward(x, params[0], params[1], params[2])
            # Force stream synchronization to get the result.
            stream.synchronize()
            result[name] = out

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # Launch two threads concurrently.
    t1 = threading.Thread(target=launch_kernel, args=("out1", params1, stream1))
    t2 = threading.Thread(target=launch_kernel, args=("out2", params2, stream2))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Compute expected outputs using PyTorch avg_pool1d.
    avg_pool1 = torch.nn.AvgPool1d(kernel_size=params1[0], stride=params1[1], padding=params1[2])
    avg_pool2 = torch.nn.AvgPool1d(kernel_size=params2[0], stride=params2[1], padding=params2[2])
    expected1 = avg_pool1(x)
    expected2 = avg_pool2(x)

    # Due to the constant memory conflict issue, one or both results may be incorrect.
    # We check and trigger an assertion failure if the result does match expectation.
    err1 = (result["out1"] - expected1).abs().max().item()
    err2 = (result["out2"] - expected2).abs().max().item()
    assert err1 < 1e-5 and err2 < 1e-5, (
        f"Concurrent kernel launches produced incorrect results due to constant memory interference. "
        f"Error for kernel1: {err1}, kernel2: {err2}"
    )

# Issue 4: Although not directly causing an exception in our kernel, this test attempts to simulate
# a scenario with large batch sizes that might force an invalid grid.z dimension.
def test_large_batch_size():
    cuda_module = build_kernel()
    in_channels = 4
    # Use a batch size that may exceed the maximum grid.z dimension.
    # The maximum grid z-dimension is device-dependent; here we use an artificially large number.
    huge_batch_size = 70000
    input_length = 64
    x = torch.randn(huge_batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 1
    padding = 1
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, kernel_size, stride, padding)
        torch.cuda.synchronize()
