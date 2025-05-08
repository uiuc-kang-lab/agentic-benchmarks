
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import time

def build_kernel():
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Double precision mismatch due to constant memory being float.
def test_double_precision_mismatch():
    cuda_module = build_kernel()
    # Use high-precision min/max values that are not exactly representable as float
    min_val = -0.123456789012345
    max_val = 0.123456789012345

    # Create a double tensor
    x = torch.randn(1024, device="cuda", dtype=torch.double)
    # Expected output using PyTorch’s native hardtanh (which will be double-precise)
    expected = x.clone().clamp(min=min_val, max=max_val)
    
    # Call our kernel – note that forward expects float values for min/max,
    # but our tensor is double. The constant values are stored as float so precision may be lost.
    out = cuda_module.forward(x, min_val, max_val)
    torch.cuda.synchronize()

    # The output may differ from expected if the double precision min/max are not respected.
    # We expect a significant difference if the issue is present.
    assert not torch.allclose(out, expected, atol=1e-12), (
        "Kernel output unexpectedly matches expected double precision result. "
        "This indicates no precision error, but the kernel uses float constants."
    )

# Issue 2: Missing error handling for cudaMemcpyToSymbol.
# We simulate a scenario by calling the kernel repeatedly while forcing a situation
# where an error might be expected. Since we cannot force cudaMemcpyToSymbol to fail
# easily, we rely on passing an invalid type for min_val/max_val.
def test_cudaMemcpyToSymbol_no_error_handling():
    cuda_module = build_kernel()
    # Pass values as integers instead of floats (this will be implicitly converted,
    # but it tests that no error is raised even if the inputs are not float).
    min_val = -1
    max_val = 1

    x = torch.randn(1024, device="cuda")
    try:
        out = cuda_module.forward(x, min_val, max_val)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel call failed due to cudaMemcpyToSymbol error handling issue: {e}")
    # Even though no error is raised, the lack of error checking in the kernel means
    # that we cannot catch internal cudaMemcpyToSymbol failures.
    # (This test case is a placeholder to signal the missing error checking.)

# Issue 3: Missing kernel launch error checking.
# We simulate a launch error by attempting to launch the kernel on a tensor with an extremely
# large number of elements. The kernel launch parameters may be invalid, but in many cases this
# will result in a silent failure. We then check for cuda errors.
def test_kernel_launch_error_checking():
    cuda_module = build_kernel()
    # Create a tensor with extremely large element count.
    # Note: This tensor may be too large for available GPU memory on some machines;
    # in that case, we simulate by creating a tensor that forces an enormous grid dimension.
    # We use a try/except to catch any cuda runtime error.
    try:
        # Creating a tensor with a huge number of elements.
        # It is expected in a proper implementation to check grid launch errors,
        # but here we want to see that no explicit error is caught despite the faulty kernel.
        num_elements = 2**31  # a number that may cause problems in grid config
        x = torch.randn(num_elements, device="cuda")
        out = cuda_module.forward(x, -1.0, 1.0)
        # Force synchronization to catch asynchronous kernel launch errors.
        torch.cuda.synchronize()
    except RuntimeError as e:
        # If an error is raised, it indicates that the lack of kernel launch error checking
        # in the C++ code would have caught this error, so we mark the test as failed.
        pytest.fail(f"Kernel launch error detected (lack of error checking): {e}")
    # If no error is raised, the test passes, but note that the missing check remains an issue.

# Issue 4: Race condition due to global constant memory usage.
# We create two CUDA streams and launch kernel calls with different min and max values concurrently.
def test_race_condition_in_constant_memory():
    cuda_module = build_kernel()
    # Prepare two input tensors
    x1 = torch.linspace(-2.0, 2.0, steps=1024, device="cuda")
    x2 = torch.linspace(-2.0, 2.0, steps=1024, device="cuda")
    # Set different clamp values in two streams
    min_val1, max_val1 = -0.5, 0.5
    min_val2, max_val2 = -0.2, 0.2

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    out1 = torch.empty_like(x1)
    out2 = torch.empty_like(x2)

    # Launch kernel on stream1
    with torch.cuda.stream(stream1):
        out1 = cuda_module.forward(x1, min_val1, max_val1)
    # Launch kernel on stream2 immediately in parallel
    with torch.cuda.stream(stream2):
        out2 = cuda_module.forward(x2, min_val2, max_val2)

    # Synchronize streams
    stream1.synchronize()
    stream2.synchronize()

    # Compute expected outputs using PyTorch’s clamp (applied to CPU copies for precision)
    expected1 = x1.clamp(min=min_val1, max=max_val1)
    expected2 = x2.clamp(min=min_val2, max=max_val2)
    # Because the constant memory is global, one stream's cudaMemcpyToSymbol may
    # overwrite the other’s values. Thus, at least one of these outputs may be incorrect.
    err1 = (out1 - expected1).abs().max().item()
    err2 = (out2 - expected2).abs().max().item()
    # We assert that at least one of them deviates significantly; if both are correct,
    # then the race condition did not occur.
    assert (err1 > 1e-3 or err2 > 1e-3), (
        "Concurrent kernel launches did not lead to interference, but global constant "
        "memory usage should induce a race condition in more complex scenarios."
    )
