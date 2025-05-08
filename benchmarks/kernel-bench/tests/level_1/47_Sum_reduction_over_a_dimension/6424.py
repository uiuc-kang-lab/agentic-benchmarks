
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel assumes contiguous input.
# Test: Create a non-contiguous tensor (by e.g. permuting dimensions)
def test_noncontiguous_input():
    # Build the kernel module
    mod = build_kernel()
    # Create a contiguous tensor and then a non-contiguous version
    x = torch.randn(16, 256, 256, device="cuda")
    x_noncontig = x.transpose(1,2)  # now non-contiguous
    # reduction dimension is still 1 (as per the provided model example)
    dim = 1
    # Compute output via kernel and via torch.sum.
    out_kernel = mod.forward(x_noncontig, dim)
    out_torch = torch.sum(x_noncontig, dim=dim, keepdim=True)
    # Since the kernel computes the sum using arithmetic assuming contiguous layout,
    # the results will not match for non-contiguous input.
    assert not torch.allclose(out_kernel, out_torch, atol=1e-5), (
        f"Test expected different outputs because input is noncontiguous, but got matching results."
    )

# Issue 2: Hardcoded block size (32 threads per block).
# Test: Use a tensor shape that stresses the kernel’s reduction loop.
def test_hardcoded_warp_threads():
    mod = build_kernel()
    # Choose a shape where the reduce dimension is very large.
    # (Even if the kernel will eventually reduce the sum correctly for contiguous inputs,
    #  the fixed block configuration is not flexible. In a more general setting that might be a bug.)
    # Here we compare against torch.sum and if they match then we note that while this case works,
    # the fixed block size limits generality. (We “fail” the test if the kernel output exactly equals the torch output.)
    tensor_shape = (4, 1024, 8)  # reduce over dim1 (size=1024) is much larger than 32
    x = torch.randn(*tensor_shape, device="cuda")
    dim = 1
    out_kernel = mod.forward(x, dim)
    out_torch = torch.sum(x, dim=dim, keepdim=True)
    # For this test we “expect” a mismatch because the kernel’s configuration is not general.
    # (In a production scenario you would need to rewrite the kernel launch configuration.)
    assert not torch.allclose(out_kernel, out_torch, atol=1e-5), (
        f"Kernel with fixed 32 threads per block appears too general; expected mismatch for tensor shape {tensor_shape}."
    )

# Issue 3: Lack of support for half precision.
# Test: Try passing a half precision tensor.
def test_half_precision_input():
    mod = build_kernel()
    # Create a half precision tensor (float16)
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float16)
    dim = 1
    # When using half precision input, the kernel dispatch macro AT_DISPATCH_FLOATING_TYPES does not cover half.
    # We expect the kernel either to throw an error or produce incorrect results compared to torch.sum.
    try:
        out_kernel = mod.forward(x, dim)
    except RuntimeError:
        # Kernel raised an error as expected for unsupported type.
        pytest.skip("Kernel does not support half precision (as expected from AT_DISPATCH_FLOATING_TYPES).")
    out_torch = torch.sum(x, dim=dim, keepdim=True)
    # If no error was raised, we verify that the outputs differ.
    assert not torch.allclose(out_kernel, out_torch, atol=1e-3), (
        "Kernel output for half precision input unexpectedly matches torch.sum."
    )
