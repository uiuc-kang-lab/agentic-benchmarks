
import torch
import pytest
from torch.utils.cpp_extension import load
import time

# Build the CUDA extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="masked_cumsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True
    )
    return cuda_module

# Issue 1 (Inefficient thread utilization): A correctness test which 
# forces the kernel to process a very long vector per row. Although the output 
# is correct, this test highlights that only 1 thread per row computes the cumulative sum.
def test_inefficient_thread_utilization():
    # Use a long cumulative dimension so that the lost parallelism becomes evident
    batch_size = 128
    length = 1000000  # very large L
    # Use a simple pattern for which the cumulative sum is easy to predict.
    x = torch.ones(batch_size, length, device="cuda")
    # Use a mask that is True for every element
    mask = torch.ones(batch_size, length, dtype=torch.bool, device="cuda")
    
    mod = build_kernel()
    t0 = time.time()
    out = mod.forward(x, mask)
    torch.cuda.synchronize()
    t1 = time.time()
    # expected cumulative sum is [1, 2, 3, ..., L] for each row
    expected = torch.cumsum(x, dim=1)
    # Check correctness even if performance is poor.
    assert torch.allclose(out, expected), "Incorrect cumulative sum computed"
    
    # We also warn if the kernel is unusually slow relative to torch.cumsum (pure-Python)
    t2 = time.time()
    expected = torch.cumsum(x, dim=1)
    t3 = time.time()
    kernel_time = t1 - t0
    torch_cumsum_time = t3 - t2
    # If the kernel time is more than, say, 10x the torch version,
    # warn that the implementation seems inefficient.
    assert kernel_time < torch_cumsum_time * 10, \
      f"Kernel appears inefficient: kernel_time={kernel_time} vs torch_cumsum_time={torch_cumsum_time}"

# Issue 2 (Over-synchronization): This test uses a non-trivial mask pattern
# such that the unnecessary synchronizations inside the loop will be exercised.
def test_over_synchronization():
    batch_size = 32
    length = 1024
    # Create an x tensor with a ramp so that errors would appear if synchronization were broken.
    x = torch.arange(0, batch_size * length, device="cuda", dtype=torch.float32).view(batch_size, length)
    # Create a mask that is True for even indices only.
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, ::2] = True

    mod = build_kernel()
    out = mod.forward(x, mask)
    
    # Expected output, computed on CPU
    x_masked = x * mask
    expected = torch.cumsum(x_masked, dim=1)
    assert torch.allclose(out, expected, atol=1e-5), "Output mismatch with over-synchronization test"

# Issue 3 (Unused optimization parameter ELEMENTS_PER_THREAD):
# Since ELEMENTS_PER_THREAD is defined but not used, make sure that the kernel 
# does not assume any grouping of elements per thread. This test uses an input 
# shape that would expose grouping if it were incorrectly implemented.
def test_unused_elements_per_thread():
    batch_size = 10
    length = 513  # A number that is not a multiple of any typical grouping factor
    x = torch.randn(batch_size, length, device="cuda")
    # A random boolean mask.
    mask = torch.randint(0, 2, (batch_size, length), device="cuda").bool()
    
    mod = build_kernel()
    out = mod.forward(x, mask)
    # Compare with torch.cumsum applied on masked input.
    expected = torch.cumsum(x * mask, dim=1)
    assert torch.allclose(out, expected, atol=1e-5), "Kernel failed on input that exposes unused grouping"

# Issue 4 (Limited scalability with grid configuration):
# If the number of rows (N) exceeds the typical grid dimension, the kernel’s one-block-per-row strategy 
# may break.  This test forces a high number of rows.
def test_grid_configuration_scalability():
    # Try to simulate a scenario with many rows.
    # Note: The actual limit is device dependent; we choose a number high enough to stress the grid config.
    device_props = torch.cuda.get_device_properties("cuda")
    max_blocks = device_props.maxGridSize[0]
    # We choose a number that is 20% of the maximum grid width, or at least 1000 rows.
    N = max(1000, max_blocks // 5)
    length = 128
    x = torch.randn(N, length, device="cuda")
    mask = torch.randint(0, 2, (N, length), device="cuda").bool()
    
    mod = build_kernel()
    # This call should work provided N does not exceed the device limits.
    out = mod.forward(x, mask)
    expected = torch.cumsum(x * mask, dim=1)
    assert torch.allclose(out, expected, atol=1e-5), "Kernel failed in high grid count scenario"

# Additional tests to trigger input checks from the host code

def test_non_contiguous_input():
    # The kernel requires contiguous inputs. Provide a non-contiguous tensor.
    x = torch.randn(64, 512, device="cuda")
    x_noncontig = x.t()  # This transpose is non-contiguous
    mask = torch.randint(0, 2, (512, 64), device="cuda").bool()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        mod = build_kernel()
        mod.forward(x_noncontig, mask)

def test_mismatched_shapes():
    # x and mask must have the same shape.
    x = torch.randn(32, 256, device="cuda")
    mask = torch.randint(0, 2, (32, 255), device="cuda").bool()
    with pytest.raises(RuntimeError, match="x and mask must have the same shape"):
        mod = build_kernel()
        mod.forward(x, mask)

def test_invalid_dimension():
    # Test providing an invalid dimension (e.g., dim that is out of range)
    x = torch.randn(16, 128, device="cuda")
    mask = torch.randint(0, 2, (16, 128), device="cuda").bool()
    # Here we deliberately pass an invalid dimension (e.g., 5 for a 2D tensor)
    with pytest.raises(RuntimeError, match="Invalid dimension"):
        mod = build_kernel()
        # Wrap the call so that the invalid dim is passed into the host function.
        mod.forward(x, mask, 5)
