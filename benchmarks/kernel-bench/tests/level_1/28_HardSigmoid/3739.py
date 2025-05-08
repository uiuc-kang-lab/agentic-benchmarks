
import pytest
import torch
from torch.utils.cpp_extension import load
import time

def build_kernel():
    cuda_module = load(
        name="hardsigmoid_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Force divergence in __syncthreads()
# Create an input tensor whose total number of elements is smaller than the block size (threads per block),
# so that in each block some threads do not process any element. This situation forces divergence: some threads
# enter the loop while others skip it, leading to an imbalanced __syncthreads() call.
@pytest.mark.timeout(10)
def test_divergent_syncthreads():
    # The kernel is launched with 256 threads per block.
    # Create an input tensor with numel < 256. For example, 128 elements.
    x = torch.randn(128, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # The call below is expected to hang or eventually trigger a CUDA error due to deadlock.
    start = time.time()
    try:
        y = kernel.forward(x)
        # Force synchronization to capture deadlocks
        torch.cuda.synchronize()
    except Exception as e:
        # If an error is raised, that indicates the kernel mis-synchronization issue.
        duration = time.time() - start
        pytest.skip(f"Kernel raised an exception (likely due to divergence): {e}, duration: {duration:.2f} sec")
    duration = time.time() - start
    # If it returns successfully quickly then the test should fail because we expect a deadlock or error.
    pytest.fail(f"Kernel executed in {duration:.2f} seconds when a deadlock or synchronization issue was expected.")

# Test 2: Trigger the boundary check issue in the grid-stride loop.
# Create an input tensor with a number of elements that is not a multiple of the thread block size.
# This should force some threads to operate on elements that are near the end of the tensor.
@pytest.mark.timeout(10)
def test_boundary_check():
    # Create a tensor with number of elements a little above a multiple of the block size.
    # Given kernel launches with 256 threads per block, choose 256 * 4 + 13 elements.
    num_elements = 256 * 4 + 13
    x = torch.randn(num_elements, device="cuda", dtype=torch.float32)
    expected = torch.nn.functional.hardsigmoid(x)
    kernel = build_kernel()
    try:
        y = kernel.forward(x)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel raised an exception unexpectedly: {e}")
    # The mis-handling of out-of-bound threads may lead to corrupted output.
    if not torch.allclose(y, expected, atol=1e-5):
        pytest.fail("Kernel output does not match expected HardSigmoid results (possible out-of-bound memory access)")

# Test 3: A correctness test with a larger tensor where multiple grid-stride iterations occur.
# This test is intended to show that even if the kernel computes the right mathematics in ideal conditions,
# its extra __syncthreads() and shared memory usage might be unnecessary.
@pytest.mark.timeout(10)
def test_large_input_correctness():
    # Use a large tensor so that the grid-stride loop runs multiple iterations.
    num_elements = 256 * 1000 + 73  # deliberately non-multiple of block size
    x = torch.randn(num_elements, device="cuda", dtype=torch.float32)
    expected = torch.nn.functional.hardsigmoid(x)
    kernel = build_kernel()
    try:
        y = kernel.forward(x)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel raised an exception unexpectedly: {e}")

    assert torch.allclose(y, expected, atol=1e-5), \
        f"Kernel output does not match expected results; max difference: {(y-expected).abs().max().item()}"

