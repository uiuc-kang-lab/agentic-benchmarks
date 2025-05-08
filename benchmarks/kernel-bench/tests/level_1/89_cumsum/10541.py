
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

# Test Issue 1: Kernel only supports float32.
def test_non_float_dtype():
    kernel = build_kernel()
    # Create a double tensor even though the kernel expects float32.
    x = torch.randn(128, 4000, device="cuda", dtype=torch.double)
    # The kernel will treat the raw pointer as float*, resulting in an incorrect result.
    out_kernel = kernel.forward(x, 1)
    out_ref = torch.cumsum(x, dim=1)
    # They are expected not to match.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), (
        "Kernel incorrectly processed a non-float32 tensor without error."
    )

# Test Issue 2: Misleading name 'warp_optimized' despite lacking warp-level optimizations.
def test_warp_optimization_misnomer():
    # This test is designed to flag the naming issue.
    pytest.fail(
        "Issue 2: The kernel function 'cumsum_warp_optimized' is misnamed since it does not implement "
        "any actual warp-level optimizations."
    )

# Test Issue 3: No error checking after kernel launch.
# We trigger a condition that causes an error during kernel execution.
def test_kernel_launch_error_check():
    kernel = build_kernel()
    # Create a 0-dimensional tensor; torch.cumsum (and hence our kernel) requires at least 1 dimension.
    x = torch.tensor(1.0, device="cuda")
    with pytest.raises(RuntimeError):
        kernel.forward(x, 0)

# Test Issue 4: Kernel assumes contiguous memory.
def test_non_contiguous_input():
    kernel = build_kernel()
    x = torch.randn(128, 4000, device="cuda", dtype=torch.float32)
    # Make the tensor non-contiguous by transposing it.
    x_nc = x.t()  # Transpose makes it non-contiguous
    with pytest.raises(RuntimeError):
        kernel.forward(x_nc, 1)

# Test Issue 5: Sequential accumulation degrades performance for large strides.
def test_sequential_scan_performance():
    kernel = build_kernel()
    # Create a tensor with a large cumulative dimension (large 'stride')
    # For instance, shape (128, 100000) where the cumulative scan is along dim=1.
    x = torch.randn(128, 100000, device="cuda", dtype=torch.float32)
    
    # Measure kernel execution time.
    torch.cuda.synchronize()
    start = time.time()
    out_kernel = kernel.forward(x, 1)
    torch.cuda.synchronize()
    elapsed_kernel = time.time() - start
    
    # Measure PyTorch's internal cumsum (highly optimized and probably parallelized)
    torch.cuda.synchronize()
    start = time.time()
    out_ref = torch.cumsum(x, dim=1)
    torch.cuda.synchronize()
    elapsed_ref = time.time() - start

    # Check correctness first.
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel cumulative sum does not match torch.cumsum."
    
    # If the kernel is more than, say, 10x slower than torch.cumsum,
    # we consider that a sign of suboptimal performance due to sequential accumulation.
    if elapsed_kernel > (elapsed_ref * 10):
        pytest.fail(
            f"Issue 5: Kernel performance is suboptimal. Kernel time: {elapsed_kernel:.6f}s, "
            f"torch.cumsum time: {elapsed_ref:.6f}s."
        )
