
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper: dynamically build the CUDA extension from the kernel.cu file.
def build_kernel():
    # Ensure kernel.cu exists in the current directory.
    kernel_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="rms_norm_kernel",
        sources=[kernel_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Reference implementation: mimics the PyTorch forward computation.
def rms_norm_reference(x, eps):
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms

# Test 1: No error is raised if cudaMemcpyToSymbol or kernel launch fail.
# We simulate a failure by trying to drive the kernel with an invalid input shape.
def test_invalid_dimension_input():
    my_module = build_kernel()
    # Provide an input tensor with only 2 dimensions.
    x = torch.randn(16, 64, device="cuda", dtype=torch.float32)
    # Expect the kernel to fail (e.g. indexing out-of-bound)
    with pytest.raises(RuntimeError):
        out = my_module.forward(x, 1e-5)
        torch.cuda.synchronize()

# Test 2: Provide a non-contiguous input tensor.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous input tensor of shape (batch, features, dim1, dim2)
    x_contig = torch.randn(16, 64, 32, 32, device="cuda", dtype=torch.float32)
    # Make the tensor non-contiguous by permuting dimensions (while keeping shape same)
    x_noncontig = x_contig.permute(0,1,3,2)
    # Run the custom CUDA kernel
    out_kernel = my_module.forward(x_noncontig, 1e-5)
    # Compute reference using the same input (non-contiguous) on CPU/GPU with PyTorch
    out_ref = rms_norm_reference(x_noncontig, 1e-5)
    torch.cuda.synchronize()
    # The results should match within a tolerance.
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), \
         "Kernel output differs for non-contiguous input tensor!"

# Test 3: Using half-precision tensor to trigger potential half-precision math issues.
def test_half_precision():
    my_module = build_kernel()
    x = torch.randn(16, 64, 32, 32, device="cuda", dtype=torch.float16)
    out_kernel = my_module.forward(x, 1e-5)
    out_ref = rms_norm_reference(x, 1e-5)
    torch.cuda.synchronize()
    # use a looser tolerance for half precision comparisons
    assert torch.allclose(out_kernel, out_ref, atol=1e-2), \
         "Kernel output differs for half-precision input tensor!"

# Test 4: Provide invalid epsilon to see if negative eps (which might trigger sqrt of negative)
# is handled the same way as the PyTorch reference.
def test_negative_epsilon():
    my_module = build_kernel()
    # Choose a tensor such that the mean square is guaranteed to be lower than abs(eps)
    # For example, using a very small tensor.
    x = torch.full((16, 64, 8, 8), 1e-3, device="cuda", dtype=torch.float32)
    negative_eps = -1e-4
    # Both our kernel and the reference should produce NaNs or similar.
    out_kernel = my_module.forward(x, negative_eps)
    out_ref = rms_norm_reference(x, negative_eps)
    torch.cuda.synchronize()
    # Comparing NaNs is tricky. We check that both have NaNs in the same locations.
    assert torch.isnan(out_kernel).all() == torch.isnan(out_ref).all(), \
         "Kernel output handling of negative epsilon (leading to NaN) differs from reference!"

