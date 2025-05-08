
import torch
import pytest
from torch.utils.cpp_extension import load
import time

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Trigger shared memory usage issue by comparing runtime on a small tensor.
# Even though the kernel functionally computes SELU correctly, the extra shared memory load–sync pattern
# can become a performance liability on certain inputs.
def test_shared_memory_overhead():
    my_module = build_kernel()
    # Use a relatively small tensor size to emphasize per–thread overhead
    x = torch.randn(64, device='cuda', dtype=torch.float32)
    # Warmup call
    out = my_module.forward(x)
    torch.cuda.synchronize()
    # Time multiple launches of the kernel
    n_iter = 1000
    start = time.perf_counter()
    for _ in range(n_iter):
        out = my_module.forward(x)
    torch.cuda.synchronize()  # ensure completion
    elapsed = time.perf_counter() - start

    # Compare with PyTorch's native SELU (which is optimized)
    start_ref = time.perf_counter()
    for _ in range(n_iter):
        out_ref = torch.selu(x)
    torch.cuda.synchronize()
    elapsed_ref = time.perf_counter() - start_ref

    # We expect the custom kernel to be slower due to redundant shared memory usage.
    # This test will fail if the kernel is optimized to avoid shared memory overhead.
    assert elapsed > elapsed_ref, f"Custom kernel appears too fast; shared memory overhead expected. Elapsed: {elapsed}, ref elapsed: {elapsed_ref}"

# Test 2: Trigger issue with non–contiguous input. The kernel assumes contiguous access.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then a non–contiguous view (e.g. via transposition)
    x = torch.randn(128, 64, device='cuda', dtype=torch.float32)
    x_t = x.t()  # transposed tensor; non–contiguous in memory
    # The kernel uses numel() and flat indexing, so if input is non–contiguous the result may be wrong.
    out_kernel = my_module.forward(x_t)
    out_torch = torch.selu(x_t)
    # Check that the results match (they may not if kernel assumed contiguity)
    assert torch.allclose(out_kernel, out_torch, atol=1e-5), "Kernel did not compute SELU correctly on non–contiguous input."

# Test 3: Trigger issue with the grid–stride loop pattern in a non–trivial shape.
def test_nontrivial_shape():
    my_module = build_kernel()
    # Create a 3D tensor which is likely to be non–contiguous after a permutation of dimensions.
    x = torch.randn(8, 16, 32, device='cuda', dtype=torch.float32)
    x_perm = x.permute(2, 0, 1)  # change the memory layout (non–contiguous)
    out_kernel = my_module.forward(x_perm)
    out_torch = torch.selu(x_perm)
    # The flat indexing in the kernel might not properly account for the memory strides.
    assert torch.allclose(out_kernel, out_torch, atol=1e-5), "Kernel did not compute SELU correctly on a non–trivial (permuted) tensor shape."
