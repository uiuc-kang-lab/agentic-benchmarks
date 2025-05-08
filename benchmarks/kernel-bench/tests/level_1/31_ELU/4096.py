
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper to compile and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32.
def test_non_float32_input():
    my_module = build_kernel()
    # Create a tensor with type double (float64) on CUDA.
    x = torch.randn(100, device="cuda", dtype=torch.double)
    with pytest.raises(RuntimeError):
        # Expecting a failure because CHECK_CUDA and the kernel's pointer cast expects float.
        my_module.forward(x)

# Issue 2: Fixed active mask in warp-level operations may give incorrect results when the warp is partially active.
def test_partial_warp_activation():
    my_module = build_kernel()
    # Create an input tensor whose number of elements is not a multiple of the block dimension.
    # The kernel uses a fixed block size of 512 threads, so if we choose a size like 513,
    # the last warp is partially active and the fixed mask may erroneously bring in inactive lanes.
    # We'll use negative values to force the expensive path.
    n = 513
    x = torch.linspace(-1.0, 1.0, steps=n, device="cuda", dtype=torch.float32)
    # Expected result computed with PyTorch's own ELU.
    expected = F.elu(x, alpha=1.0)
    # Kernel result
    result = my_module.forward(x)
    torch.cuda.synchronize()
    # Although mathematically the expensive branch produces the correct result,
    # undefined behavior due to the fixed mask might cause differences.
    assert torch.allclose(result, expected, atol=1e-5), \
        f"Kernel output ({result}) differs from expected output ({expected})."

# Issue 3: Lack of error checking after kernel launch may hide errors in non-contiguous or CPU tensors.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by a transpose.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    x = x.t()  # Now non-contiguous.
    with pytest.raises(RuntimeError):
        my_module.forward(x)

def test_cpu_input():
    my_module = build_kernel()
    # Create a tensor on CPU.
    x = torch.randn(100, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        my_module.forward(x)
