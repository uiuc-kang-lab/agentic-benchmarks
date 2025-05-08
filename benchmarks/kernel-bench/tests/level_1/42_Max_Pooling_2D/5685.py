
import pytest
import torch
from torch.utils.cpp_extension import load

# Fixture to build and load our CUDA extension
@pytest.fixture(scope="module")
def cuda_module():
    module = load(
        name="custom_maxpool",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Double precision (float64) usage.
def test_double_input(cuda_module):
    # The kernel uses fmaxf, which is not defined for double.
    x = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # Expecting an error because of the double type usage.
        cuda_module.forward(x, 2, 2, 1, 3)
        
# Issue 2: Grid dimension overflow.
def test_grid_dimension_overflow(cuda_module):
    # Create input with a very large number of channels to force gridDim.z to be > 65535.
    # For instance, batch=1 and channels=70000 (70000 > 65535).
    x = torch.randn(1, 70000, 32, 32, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 2, 2, 1, 1)

# Issue 3: Non-contiguous input.
def test_non_contiguous_input(cuda_module):
    # Create a contiguous input tensor and then make it non‐contiguous.
    x = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float32)
    non_contig_x = x.transpose(2, 3)
    # Run our cuda kernel and the standard PyTorch MaxPool2d for reference.
    out_cuda = cuda_module.forward(non_contig_x, 2, 2, 1, 3)
    maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=3)
    out_ref = maxpool(non_contig_x)
    # We expect a mismatch because the kernel does not account for non‐contiguous layouts.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-5), \
        "Kernel output matches reference even though input is non-contiguous. Expected failure."

# Issue 4: Excessive shared memory requirement.
def test_excessive_shared_memory(cuda_module):
    # Choose parameters that force a huge shared memory allocation.
    # For a block of (16, 16), shared_width = 16*stride + (kernel_size-1)*dilation.
    # With kernel_size=20, dilation=10 and stride=1 the shared memory tile becomes very large.
    x = torch.randn(1, 1, 50, 50, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 20, 1, 0, 10)

# Issue 5: Lack of kernel launch error checking.
def test_kernel_launch_error_propagation(cuda_module):
    # In cases where the kernel fails (e.g. due to misconfiguration), the error is not explicitly caught.
    # We simulate such a failure by providing parameters that lead to an error.
    # Here we reuse the grid dimension overflow (issue 2) to see if the error propagates.
    x = torch.randn(1, 70000, 32, 32, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 2, 2, 1, 1)
