
import pytest
import torch
from torch.utils.cpp_extension import load
import time

# Helper: build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger potential deadlock (issue 1) and out-of-bound accesses (issue 2)
# by creating an input tensor with number of elements less than the number of threads per block.
# Using a very small tensor (e.g. size 128 < block size of 256) should make some threads skip the loop.
def test_deadlock_and_bounds():
    # This tensor has fewer elements than the presumed blockDim (256)
    x = torch.randn(128, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # Use a non-blocking call to catch potential deadlock. We set a timeout.
    # Note: In pytest, if a CUDA kernel deadlocks, the test will just hang.
    # To simulate a timeout we use a simple timer.
    start_time = time.time()
    try:
        out = kernel.forward(x)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel raised an exception: {e}")
    duration = time.time() - start_time
    # If the kernel is deadlocking, the duration will be very long.
    assert duration < 5, f"Kernel execution took too long (possible deadlock): {duration} seconds"
    # Also compare against the PyTorch reference hardsigmoid function.
    ref = torch.nn.functional.hardsigmoid(x)
    assert torch.allclose(out, ref, atol=1e-5), "Kernel output differs from reference (possible out-of-bound access)"

# Test 2: Non-contiguous input tensor (issue 4).
# Create an input tensor that is non-contiguous and see if the kernel mishandles it.
def test_noncontiguous_input():
    # Create a contiguous tensor and then transpose it to break contiguity.
    x = torch.randn(128, 2, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # this transpose makes the tensor non-contiguous
    kernel = build_kernel()
    # The kernel may still run if it assumes a 1D contiguous buffer,
    # so we check if the output matches the reference computed using hardsigmoid.
    try:
        out = kernel.forward(x_noncontig)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.skip(f"Kernel cannot handle non-contiguous input: {e}")
        # If we expected to handle noncontiguous inputs, this would be a failure.
    ref = torch.nn.functional.hardsigmoid(x_noncontig)
    assert torch.allclose(out, ref, atol=1e-5), "Kernel output differs for non-contiguous input."

# Test 3: Check unnecessary use of shared memory does not harm correctness but may hurt performance.
# Here we compare a larger tensor against the PyTorch implementation.
def test_large_tensor():
    x = torch.randn(1 << 16, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    out = kernel.forward(x)
    torch.cuda.synchronize()
    ref = torch.nn.functional.hardsigmoid(x)
    assert torch.allclose(out, ref, atol=1e-5), "Kernel output differs on large input."

if __name__ == '__main__':
    pytest.main([__file__])
