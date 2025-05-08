
import torch
import pytest
from torch.utils.cpp_extension import load
import time

def build_kernel():
    cuda_module = load(
        name="hardtanh_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test kernel input type compatibility (non-float32 tensor)
def test_non_float_input():
    my_module = build_kernel()
    # Create a double tensor (float64) even though the kernel only supports float
    x = torch.randn(16, 16384, dtype=torch.float64, device="cuda")
    min_val = -1.0
    max_val = 1.0
    with pytest.raises(RuntimeError):
        # It is expected that the kernel will either crash or produce an error,
        # because the kernel uses x.data_ptr<float>() which is an invalid cast for double.
        out = my_module.forward(x, min_val, max_val)
        # Force synchronization to trigger any potential CUDA error.
        torch.cuda.synchronize()

# Issue 2: Test concurrent kernel calls that update constant memory concurrently.
def test_constant_memory_race_condition():
    my_module = build_kernel()
    # Create a float32 tensor input.
    x = torch.randn(16, 16384, dtype=torch.float32, device="cuda")
    
    # Using two different sets of min/max values.
    min_val1, max_val1 = -1.0, 1.0
    min_val2, max_val2 = 0.5, 2.0

    # Create two different CUDA streams.
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # Place the operations in different streams.
    # Note: We deliberately do not synchronize before launching the second call
    # to increase the chance that the constant memory values will be overwritten.
    torch.cuda.synchronize()  # ensure a clean start

    with torch.cuda.stream(stream1):
        out1 = my_module.forward(x, min_val1, max_val1)

    # Minimal delay to increase concurrency likelihood.
    time.sleep(0.001)

    with torch.cuda.stream(stream2):
        out2 = my_module.forward(x, min_val2, max_val2)
    
    # Wait for all streams to complete.
    torch.cuda.synchronize()
    
    # Compute expected results for both cases.
    expected1 = torch.clamp(x, min=min_val1, max=max_val1)
    expected2 = torch.clamp(x, min=min_val2, max=max_val2)
    
    # Because of the race condition in constant memory, one or both outputs may be incorrect.
    # We check if at least one output does not match its expected value.
    err1 = (out1 - expected1).abs().max().item()
    err2 = (out2 - expected2).abs().max().item()
    
    # If both are close to the expected output, the race condition did not trigger (this might rarely happen),
    # but the test is designed to trigger the potential race issue with concurrent launches.
    assert not (torch.allclose(out1, expected1, atol=1e-5) and torch.allclose(out2, expected2, atol=1e-5)), (
        "Both outputs match expectations. This may indicate that the race condition in constant memory was not triggered. "
        "However, in concurrent scenarios with different parameter values, the kernel is not reliable."
    )

# Issue 3: Test that lack of error checking on cudaMemcpyToSymbol might cause silent failures.
# (We simulate an error by passing an input tensor that is not allocated on the device.)
def test_cudaMemcpyToSymbol_error():
    my_module = build_kernel()
    # Create a valid tensor on CUDA.
    x = torch.randn(16, 16384, dtype=torch.float32, device="cuda")
    min_val = -1.0
    max_val = 1.0
    # Force an error by passing a tensor that is not on the CUDA device.
    x_cpu = x.cpu()
    with pytest.raises(RuntimeError):
        # This should raise an error because forward() checks for CUDA tensor.
        out = my_module.forward(x_cpu, min_val, max_val)
        torch.cuda.synchronize()
